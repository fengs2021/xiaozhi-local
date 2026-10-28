"""小智本地控制中心：OTA 端点 + WebSocket 音频服务 + 网页控制台。"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from aiohttp import WSMsgType, web

from . import __version__
from .asr import build_asr
from .config import (
    DEFAULT_PERSONA,
    active_preset,
    delete_preset,
    enabled_tools,
    load_config,
    public_llm_view,
    public_persona_view,
    resolve_path,
    sanitize_persona,
    save_config,
    set_active,
    upsert_preset,
    weather_city,
)
from .history import ChatStore
from .schedule import (
    CST,
    ScheduleService,
    ScheduleStore,
    build_item_view,
    compute_next,
)
from datetime import datetime
from .limits import OutputCounter
from .personas import PersonaStore
from .memory import KIND_LABEL, KINDS, MemoryStore
from .memory_agent import MemoryAgent
from .music import MusicPlayer, MusicSource
from .playlists import PlaylistStore
from .llm import build_llm
from .session import DeviceSession
from .tools import TOOLS
from .tts import (
    EDGE_VOICES,
    MaoxiangPreview,
    PreviewTTS,
    build_tts,
    pcm_to_wav_bytes,
    voice_catalog,
)
from .vad import build_vad

log = logging.getLogger("server")

STATIC_DIR = Path(__file__).resolve().parent / "static"
# 猫剪音色表缓存
MAOXIANG_VOICES_CACHE = STATIC_DIR.parent.parent / "data" / "voices_maoxiang.json"


def tts_view(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """控制中心用的 tts 视图。"""
    t = cfg.get("tts") or {}
    return {
        "engine": t.get("engine", "edge"),
        "voice": t.get("voice", ""),
        "emotion": t.get("emotion", ""),
        "emotion_scale": t.get("emotion_scale", 0),
        "loudness": t.get("loudness", 0),
        "engines": [
            {"id": "maoxiang", "name": "火山豆包 / 猫剪（在线，音色多）"},
            {"id": "edge", "name": "微软 Edge（在线，稳定）"},
            {"id": "sapi", "name": "Windows SAPI（离线）"},
            {"id": "none", "name": "静音（不出声）"},
        ],
    }

# 公网反向代理（Cloudflare Tunnel / nginx）会带上这些头
_PROXY_HOST_HEADERS = ("X-Forwarded-Host", "X-Original-Host", "Host")


def _first_header(request: web.Request, names) -> str:
    for name in names:
        value = request.headers.get(name, "")
        if value:
            return value.split(",")[0].strip()
    return ""


def _is_loopback_or_private(hostname: str) -> bool:
    """判断主机名是否只能内网访问（局域网 IP / 回环 / mDNS 名）。"""
    if not hostname:
        return True
    h = hostname.strip("[]").lower()
    if h in ("localhost", "127.0.0.1", "::1") or h.endswith(".local"):
        return True
    parts = h.split(".")
    if len(parts) == 4 and all(p.isdigit() for p in parts):
        a, b = int(parts[0]), int(parts[1])
        if a == 10 or a == 127:
            return True
        if a == 192 and b == 168:
            return True
        if a == 172 and 16 <= b <= 31:
            return True
        if a == 169 and b == 254:
            return True
    return False


def build_ws_url(request: web.Request, server: Dict[str, Any], path: str) -> str:
    """按请求来源生成设备可用的 WebSocket 地址。

    优先级：
      1. config 里显式写的 server.public_ws_url（强制覆盖，留空即自动）
      2. 反代传来的 Host 头（公网域名 -> wss://域名/路径）
      3. server.public_host + port（局域网直连 -> ws://ip:port/路径）
    """
    forced = str(server.get("public_ws_url") or "").strip()
    if forced:
        return forced

    req_host = _first_header(request, _PROXY_HOST_HEADERS)
    if req_host.startswith("["):
        hostname = req_host[1:req_host.index("]")] if "]" in req_host else req_host
    elif ":" in req_host:
        hostname = req_host.rsplit(":", 1)[0]
    else:
        hostname = req_host

    forwarded_proto = _first_header(request, ("X-Forwarded-Proto",))
    scheme = forwarded_proto or request.scheme or "http"

    if hostname and not _is_loopback_or_private(hostname):
        ws_scheme = "wss" if scheme == "https" else "ws"
        return f"{ws_scheme}://{req_host}{path}"

    host = server.get("public_host") or "127.0.0.1"
    port = int(server.get("port", 8003))
    return f"ws://{host}:{port}{path}"


class Hub:
    """保存当前连接设备与会话，供网页展示。"""

    def __init__(self, store_path: Optional[Path] = None) -> None:
        self.sessions: Dict[str, DeviceSession] = {}
        self.devices: Dict[str, Dict[str, Any]] = {}
        self.logs: List[Dict[str, Any]] = []
        self.started_at = time.time()
        self._devices_path = store_path
        self._load_devices()

    def _load_devices(self) -> None:
        """从磁盘恢复设备历史（重启后仍能看到上次接入情况）。"""
        path = self._devices_path
        if not path or not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                self.devices = data
                for dev in self.devices.values():
                    dev["online"] = False
        except Exception as exc:
            log.warning("读取设备历史失败: %s", exc)

    def _save_devices(self) -> None:
        path = self._devices_path
        if not path:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(self.devices, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception as exc:
            log.warning("保存设备历史失败: %s", exc)

    def add_log(self, kind: str, text: str) -> None:
        self.logs.append({"t": time.time(), "kind": kind, "text": text})
        if len(self.logs) > 300:
            del self.logs[:100]

    def touch_device(self, device_id: str, client_id: str = "", online: bool = True) -> None:
        """记录设备历史：断线后仍保留，供网页展示最近连接情况。"""
        if not device_id:
            return
        now = time.time()
        dev = self.devices.setdefault(
            device_id,
            {
                "device_id": device_id,
                "client_id": client_id,
                "first_seen": now,
                "sessions": 0,
                "uplink_packets": 0,
                "downlink_packets": 0,
                "utterances": 0,
                "last_stt": "",
                "last_llm": "",
            },
        )
        if client_id:
            dev["client_id"] = client_id
        if online:
            dev["online"] = True
            dev["connected_at"] = now
            dev["sessions"] = int(dev.get("sessions", 0)) + 1
        else:
            dev["online"] = False
        dev["last_seen"] = now
        self._save_devices()

    def snapshot(self) -> Dict[str, Any]:
        live = [s.info() for s in self.sessions.values()]
        # 在线设备用实时统计刷新历史记录
        for info in live:
            dev = self.devices.get(info.get("device_id") or "")
            if dev:
                dev["uplink_packets"] = info.get("uplink_packets", 0)
                dev["downlink_packets"] = info.get("downlink_packets", 0)
                dev["utterances"] = info.get("utterances", 0)
                dev["last_stt"] = info.get("last_stt", "")
                dev["last_llm"] = info.get("last_llm", "")
                dev["state"] = info.get("state", "idle")
        devices = sorted(self.devices.values(), key=lambda d: d.get("last_seen", 0), reverse=True)
        return {
            "uptime": round(time.time() - self.started_at, 1),
            "sessions": live,
            "devices": devices,
            "logs": self.logs[-80:],
        }


class App:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.hub = Hub(Path(cfg["_root"]) / "data" / "devices.json")
        # 气象台站点索引落盘到 data/，避免每次重启都重新抓 34 个省
        try:
            from . import weather_cn

            weather_cn.INDEX.cache_path = Path(cfg["_root"]) / "data" / "nmc_stations.json"
        except Exception as exc:
            log.warning("气象台索引初始化失败: %s", exc)
        self.asr = build_asr(cfg, cfg["_root"])
        self.tts = build_tts(cfg)
        self.llm = self._make_llm(cfg)
        # 对话历史：写 data/chats/YYYY-MM-DD.jsonl
        self.store = ChatStore(cfg["_root"])
        # 长期记忆：写 data/memory.json，跨会话保留
        self.memory = MemoryStore(cfg["_root"])
        # 每日输出字数统计：儿童场景「聊太久就劝退」的计数，写 data/output_count.json
        self.counter = OutputCounter(cfg["_root"])
        # 人设库：多套人设存 data/personas.json，切换时写回 config.yaml 的 persona 段
        self.personas = PersonaStore(cfg["_root"])
        try:
            if self.personas.ensure_seed(cfg.get("persona") or {}):
                self.hub.add_log("persona", "已把当前人设收进人设库")
        except Exception as exc:
            log.warning("人设库初始化失败: %s", exc)
        # 在线音乐：搜歌 + 下载 + 解码成 opus 推给设备，mp3 缓存在 data/music/cache
        self.playlists = PlaylistStore(Path(cfg["_root"]))
        self.music_source: Optional[MusicSource] = None
        self.music_bitrate = 64000
        music_cfg = cfg.get("music") or {}
        if music_cfg.get("enabled", True):
            try:
                self.music_bitrate = int(music_cfg.get("bitrate", 64000) or 64000)
                self.music_source = MusicSource(
                    Path(cfg["_root"]) / "data" / "music" / "cache",
                    timeout=float(music_cfg.get("timeout", 12) or 12),
                    cache_max_mb=int(music_cfg.get("cache_max_mb", 200) or 200),
                    sources=music_cfg.get("sources", music_cfg.get("source")),
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("音乐源初始化失败: %s", exc)
        # 抽取记忆时的主语，跟随人设里的称呼
        try:
            self.memory.user_hint = str((cfg.get("persona") or {}).get("user_name") or "") or "孩子"
        except Exception:
            self.memory.user_hint = "孩子"
        # 后台记忆：每轮对话后异步调模型抽取画像与事实，不占用对话延迟
        self.memory_agent = MemoryAgent(
            self.memory,
            self._make_memory_llm,
            lambda: str((self.cfg.get("persona") or {}).get("user_name") or ""),
        )
        # 定时任务：提醒 / 闹钟，服务端时钟驱动，到点主动让设备说话
        self.schedules = ScheduleStore(cfg["_root"])
        self.schedule_svc = ScheduleService(
            self.schedules, self.hub, lambda: self.hub.sessions
        )
        self.services = {
            "asr": self.asr,
            "schedule": self.schedule_svc,
            "tts": self.tts,
            "llm": self.llm,
            "make_vad": lambda: build_vad(self.cfg),
            "hub": self.hub,
            "store": self.store,
            "memory": self.memory,
            "memory_agent": self.memory_agent,
            "counter": self.counter,
        }

    def _make_llm(self, cfg: Dict[str, Any]):
        """构造 LLMClient，并把当前启用的 MCP 工具名单注入（供 function calling 用）。"""
        preset = dict(active_preset(cfg))
        preset["tools"] = enabled_tools(cfg)
        # 意图快速通道：工具结果能直接说时，跳过第二次模型调用
        preset["direct_reply"] = bool((cfg.get("mcp") or {}).get("direct_reply", True))
        return build_llm(preset, cfg.get("persona"))

    def _make_memory_llm(self):
        """给后台记忆用的轻量客户端：不带工具、不思考、短输出。"""
        preset = dict(active_preset(self.cfg))
        preset["tools"] = []
        preset["reasoning_effort"] = "none"
        preset["max_tokens"] = 900
        preset["temperature"] = 0.1
        return build_llm(preset, {})

    def reload_llm(self) -> None:
        self.cfg = load_config()
        self.llm = self._make_llm(self.cfg)
        self.services["llm"] = self.llm
        log.info(
            "LLM 已切换为 %s，可用工具 %d 个",
            active_preset(self.cfg).get("name"),
            len(enabled_tools(self.cfg)),
        )

    def reload_tts(self) -> None:
        """音色/语速/音调改动后重建 TTS（新连接立即生效）。"""
        self.tts = build_tts(self.cfg)
        self.services["tts"] = self.tts
        log.info("TTS 已切换为 %s / %s", (self.cfg.get("tts") or {}).get("engine"), getattr(self.tts, "voice", "-"))

    # ---------- OTA ----------
    async def handle_ota(self, request: web.Request) -> web.Response:
        """固件启动/唤醒时来查版本，这里同时下发 WebSocket 连接信息。"""
        body = {}
        try:
            raw = await request.text()
            if raw:
                body = json.loads(raw)
        except Exception:
            pass
        device_id = request.headers.get("Device-Id", "")
        client_id = request.headers.get("Client-Id", "")
        ua = request.headers.get("User-Agent", "")
        self.hub.add_log("ota", f"OTA 请求 device={device_id} client={client_id} ua={ua}")

        server = self.cfg.get("server") or {}
        host = server.get("public_host") or "127.0.0.1"
        port = int(server.get("port", 8003))
        token = server.get("token", "")
        path = server.get("ws_path", "/xiaozhi/v1/")
        if not path.startswith("/"):
            path = "/" + path

        ws_url = build_ws_url(request, server, path)

        resp: Dict[str, Any] = {
            "server_time": {
                "timestamp": int(time.time() * 1000),
                "timezone_offset": 480,
            },
            "websocket": {
                "url": ws_url,
                "token": token,
            },
        }

        ota = self.cfg.get("ota") or {}
        if ota.get("offer_firmware") and ota.get("firmware_url"):
            resp["firmware"] = {
                "version": ota.get("firmware_version", "0.0.0"),
                "url": ota.get("firmware_url", ""),
            }

        log.info("OTA 响应 -> %s (host=%s)", ws_url, _first_header(request, _PROXY_HOST_HEADERS) or "-")
        return web.json_response(resp)

    # ---------- WebSocket ----------
    async def handle_ws(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse(heartbeat=30, max_msg_size=0)
        await ws.prepare(request)

        server = self.cfg.get("server") or {}
        token = server.get("token", "")
        auth = request.headers.get("Authorization", "")
        if token:
            got = auth.replace("Bearer ", "").strip() if auth else ""
            if got != token:
                log.warning("鉴权失败: Authorization=%r", auth)
                await ws.close(code=4003, message=b"unauthorized")
                return ws

        session = DeviceSession(ws, self.cfg, self.services)
        # 音乐播放器要持有 session 引用（推音频、看 _closed、被打断），只能在这里挂
        if self.music_source is not None:
            session.music = MusicPlayer(session, self.music_source, bitrate=self.music_bitrate)
        session.stats.device_id = request.headers.get("Device-Id", "")
        session.stats.client_id = request.headers.get("Client-Id", "")
        self.hub.sessions[session.stats.session_id] = session
        self.hub.touch_device(session.stats.device_id, session.stats.client_id, online=True)
        self.hub.add_log("ws", f"设备接入 {session.stats.device_id} ({session.stats.session_id[:8]})")

        # 断开原因诊断：close_code 能区分「设备主动关(1000/1001)」「网络断或心跳超时(1006)」
        last_rx = {"t": time.time(), "kind": "连接建立"}
        try:
            async for msg in ws:
                last_rx["t"] = time.time()
                last_rx["kind"] = msg.type.name
                if msg.type == WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                    except Exception:
                        continue
                    if data.get("type") == "hello":
                        audio_params = data.get("audio_params") or {}
                        self.hub.add_log(
                            "hello",
                            f"hello version={data.get('version')} 上行={audio_params.get('sample_rate')}Hz/{audio_params.get('frame_duration')}ms",
                        )
                        await session.send_json(
                            {
                                "type": "hello",
                                "transport": "websocket",
                                "session_id": session.stats.session_id,
                                "audio_params": {
                                    "format": "opus",
                                    "sample_rate": session.down_rate,
                                    "channels": 1,
                                    "frame_duration": session.down_frame_ms,
                                },
                            }
                        )
                    await session.on_json(data)
                elif msg.type == WSMsgType.BINARY:
                    await session.on_binary(msg.data)
                elif msg.type == WSMsgType.ERROR:
                    log.warning("WS 错误: %s", ws.exception())
                    break
        except Exception as exc:
            log.exception("会话异常: %s", exc)
        finally:
            quiet = time.time() - last_rx["t"]
            code = ws.close_code
            # 1000=正常关闭 1001=对端离开 1006=异常断开（网络断/心跳超时） 4003=鉴权失败
            log.info(
                "会话结束 %s close_code=%s 最后上行 %.1fs 前（%s）",
                session.stats.session_id[:8],
                code,
                quiet,
                last_rx["kind"],
            )
            await session.close()
            self.hub.sessions.pop(session.stats.session_id, None)
            self.hub.touch_device(session.stats.device_id, session.stats.client_id, online=False)
            self.hub.add_log("ws", f"设备断开 {session.stats.device_id}（code={code}）")
        return ws

    # ---------- 控制台 API ----------
    async def api_status(self, request: web.Request) -> web.Response:
        cfg = self.cfg
        preset = active_preset(cfg)
        return web.json_response(
            {
                "version": __version__,
                "config_path": str(Path(cfg["_root"]) / "config.yaml"),
                "server": cfg.get("server"),
                "llm": public_llm_view(cfg),
                "active_model": preset.get("model"),
                "asr": {"engine": (cfg.get("asr") or {}).get("engine"), "ready": type(self.asr).__name__ != "NullASR"},
                "tts": {
                    "engine": (cfg.get("tts") or {}).get("engine"),
                    "voice": (cfg.get("tts") or {}).get("voice"),
                },
                "persona": public_persona_view(cfg),
                "tools": enabled_tools(cfg),
                "weather_city": weather_city(cfg),
                **self.hub.snapshot(),
            }
        )

    # ---------- 人设库 ----------
    def _personas_view(self) -> Dict[str, Any]:
        return {
            "ok": True,
            "active": self.personas.active_id,
            "items": self.personas.list_items(),
            "persona": public_persona_view(self.cfg),
        }

    def _apply_persona(self, item: Dict[str, Any]) -> None:
        """把一条人设记录写进 config.yaml 的 persona 段，并重建 LLM / TTS。"""
        clean = sanitize_persona({k: item.get(k) for k in DEFAULT_PERSONA})
        cur = self.cfg.setdefault("persona", {})
        changed = False
        for key, value in clean.items():
            if cur.get(key) != value:
                cur[key] = value
                changed = True
        save_config(self.cfg)
        self.reload_llm()  # 内部会重新读一遍 config.yaml
        if changed:
            self.reload_tts()
        # 抽记忆时的主语跟着人设里的称呼走
        try:
            self.memory.user_hint = clean.get("user_name") or "孩子"
        except Exception:
            pass

    async def api_personas(self, request: web.Request) -> web.Response:
        """人设库：列表 / 新建 / 保存 / 复制 / 删除 / 设为当前。"""
        if request.method == "GET":
            return web.json_response(self._personas_view())

        data = await request.json()
        action = str(data.get("action") or "").strip()
        pid = str(data.get("id") or "").strip()
        item: Optional[Dict[str, Any]] = None
        try:
            if action == "create":
                item = self.personas.create(data)
                self.hub.add_log("persona", f"新建人设「{item.get('name')}」")
            elif action == "save":
                item = self.personas.update(pid, data)
                if pid == self.personas.active_id:
                    self._apply_persona(item)  # 改的就是当前在用的，立即生效
                self.hub.add_log("persona", f"人设「{item.get('name')}」已保存")
            elif action == "duplicate":
                item = self.personas.duplicate(pid)
                self.hub.add_log("persona", f"复制出人设「{item.get('name')}」")
            elif action == "delete":
                res = self.personas.delete(pid)
                switched = res.get("switched")
                if switched:
                    self._apply_persona(switched)
                    self.hub.add_log("persona", f"当前人设改为「{switched.get('name')}」")
                else:
                    self.hub.add_log("persona", "已删除一套人设")
            elif action == "activate":
                item = self.personas.set_active(pid)
                self._apply_persona(item)
                self.hub.add_log("persona", f"当前人设切换为「{item.get('name')}」")
            else:
                return web.json_response({"ok": False, "error": "未知操作"}, status=400)
        except KeyError:
            return web.json_response({"ok": False, "error": "找不到这套人设"}, status=404)
        except ValueError as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)

        out = self._personas_view()
        out["item"] = item
        return web.json_response(out)

    async def api_config(self, request: web.Request) -> web.Response:
        """读取/更新 LLM 配置。"""
        if request.method == "GET":
            return web.json_response(
                {
                    "llm": public_llm_view(self.cfg),
                    "persona": public_persona_view(self.cfg),
                    "tts": tts_view(self.cfg),
                    "voices": EDGE_VOICES,
                    "catalog": voice_catalog(self.cfg, str(MAOXIANG_VOICES_CACHE)),
                    "tools": enabled_tools(self.cfg),
                    "all_tools": [
                        {"name": t["name"], "description": t["description"]} for t in TOOLS
                    ],
                    "weather_city": weather_city(self.cfg),
                }
            )

        data = await request.json()
        llm = self.cfg.setdefault("llm", {})
        if "active" in data:
            llm["active"] = data["active"]
        if "system_prompt" in data:
            llm["system_prompt"] = data["system_prompt"]
        if "history_turns" in data:
            llm["history_turns"] = int(data["history_turns"])
        if "presets" in data:
            presets = llm.setdefault("presets", {})
            for name, p in data["presets"].items():
                cur = dict(presets.get(name) or {})
                for k, v in p.items():
                    if k == "api_key" and ("*" in str(v)):
                        continue  # 打码值不回写
                    cur[k] = v
                presets[name] = cur

        # 音色：引擎 / 音色 / 情绪，语速/音调仍走 persona
        tts_changed = False
        if "tts" in data and isinstance(data["tts"], dict):
            tts = self.cfg.setdefault("tts", {})
            incoming = data["tts"]
            engine = str(incoming.get("engine") or "").strip()
            if engine in ("maoxiang", "edge", "sapi", "none") and engine != tts.get("engine"):
                tts["engine"] = engine
                # 切引擎时音色不通用，清空让 build_tts 用该引擎默认值
                tts["voice"] = ""
                tts_changed = True
            voice = str(incoming.get("voice") or "").strip()
            if voice and voice != tts.get("voice"):
                tts["voice"] = voice
                tts_changed = True
            if "emotion" in incoming:
                emo = str(incoming.get("emotion") or "").strip()
                if emo != tts.get("emotion"):
                    tts["emotion"] = emo
                    tts_changed = True
            for key, lo, hi in (("emotion_scale", 0.0, 5.0), ("loudness", -50.0, 100.0)):
                if key in incoming:
                    try:
                        val = max(lo, min(hi, float(incoming[key])))
                    except (TypeError, ValueError):
                        continue
                    if val != tts.get(key):
                        tts[key] = val
                        tts_changed = True

        # 人设：只覆盖传入的字段，其余沿用现值（否则只改语速也会把人设清空）
        persona_changed = False
        if "persona" in data and isinstance(data["persona"], dict):
            incoming_persona = data["persona"]
            merged_persona = dict(self.cfg.get("persona") or {})
            for pkey in DEFAULT_PERSONA:
                if pkey in incoming_persona:
                    merged_persona[pkey] = incoming_persona[pkey]
            clean = sanitize_persona(merged_persona)
            cur_persona = self.cfg.setdefault("persona", {})
            for key, value in clean.items():
                if cur_persona.get(key) != value:
                    cur_persona[key] = value
                    persona_changed = True

        # MCP 工具开关与默认城市
        if "mcp" in data and isinstance(data["mcp"], dict):
            mcp = self.cfg.setdefault("mcp", {})
            if "enabled_tools" in data["mcp"] and isinstance(data["mcp"]["enabled_tools"], list):
                valid = {t["name"] for t in TOOLS}
                mcp["enabled_tools"] = [
                    str(n) for n in data["mcp"]["enabled_tools"] if str(n) in valid
                ]
            if "weather_city" in data["mcp"]:
                city = str(data["mcp"]["weather_city"] or "").strip()
                if city:
                    mcp["weather_city"] = city

        save_config(self.cfg)
        # 人设改动同步回人设库里当前使用的那条，避免两处不一致
        if persona_changed:
            try:
                self.personas.sync_active(self.cfg.get("persona") or {})
            except Exception as exc:
                log.warning("人设库同步失败: %s", exc)
        # 人设进了 system message，语音参数进了 TTS，分别重建
        self.reload_llm()
        if tts_changed or persona_changed:
            self.reload_tts()
        self.hub.add_log("config", f"配置已更新，当前模型 {active_preset(self.cfg).get('model')}")
        return web.json_response(
            {
                "ok": True,
                "llm": public_llm_view(self.cfg),
                "persona": public_persona_view(self.cfg),
                "tts": tts_view(self.cfg),
                "tools": enabled_tools(self.cfg),
                "weather_city": weather_city(self.cfg),
            }
        )

    async def api_presets(self, request: web.Request) -> web.Response:
        """预设的增 / 改 / 删 / 启用。"""
        data = await request.json()
        action = (data.get("action") or "").strip()
        name = (data.get("name") or "").strip()
        try:
            if action == "create" or action == "update":
                upsert_preset(
                    self.cfg,
                    name,
                    data.get("preset") or {},
                    old_name=(data.get("old_name") or "").strip(),
                )
                if data.get("activate"):
                    set_active(self.cfg, name)
            elif action == "delete":
                delete_preset(self.cfg, name)
            elif action == "activate":
                set_active(self.cfg, name)
            else:
                return web.json_response({"ok": False, "error": "未知操作"}, status=400)
            save_config(self.cfg)
            self.reload_llm()
            self.hub.add_log("config", f"预设 {action} -> {name}，当前 {active_preset(self.cfg).get('model')}")
            return web.json_response({"ok": True, "llm": public_llm_view(self.cfg)})
        except Exception as exc:
            log.warning("预设操作失败: %s", exc)
            return web.json_response({"ok": False, "error": str(exc)}, status=400)

    async def api_test_preset(self, request: web.Request) -> web.Response:
        """不切换当前模型，直接用某个预设试一句话。"""
        data = await request.json()
        name = (data.get("name") or "").strip()
        text = (data.get("text") or "你好，请用一句话介绍你自己").strip()
        presets = (self.cfg.get("llm") or {}).get("presets") or {}
        raw = dict(presets.get(name) or {})
        if not raw:
            return web.json_response({"ok": False, "error": "预设不存在"}, status=400)
        merged = dict(raw)
        merged["system_prompt"] = (self.cfg.get("llm") or {}).get("system_prompt", "")
        merged["history_turns"] = 0
        # 试一句要和真实会话一致：同样注入启用的工具，否则测不出 function calling
        merged["tools"] = enabled_tools(self.cfg)
        merged.setdefault("name", name)
        if not merged.get("api_key"):
            import os

            merged["api_key"] = os.environ.get("XIAOZHI_LLM_API_KEY", "")
        started = time.time()
        client = build_llm(merged, self.cfg.get("persona"))
        # 工具上下文：试一句时没有真实会话，hub 仍可注入
        client.set_tool_context(
            {"hub": self.hub, "session": None, "schedule": self.schedule_svc}
        )
        # 试一句也要带上长期记忆，否则测不出「记不记得我」这类效果
        try:
            client.set_memory_block(self.memory.prompt_block())
        except Exception:
            pass
        reply = await client.chat([], text)
        return web.json_response(
            {
                "ok": True,
                "reply": reply,
                "model": client.model,
                "base_url": client.base_url,
                "echo": client.is_echo,
                "elapsed_ms": int((time.time() - started) * 1000),
            }
        )

    async def api_test_llm(self, request: web.Request) -> web.Response:
        """在网页上试一句，验证模型连通性。"""
        data = await request.json()
        text = (data.get("text") or "你好，请自我介绍一下").strip()
        started = time.time()
        reply = await self.llm.chat([], text)
        return web.json_response(
            {"reply": reply, "elapsed_ms": int((time.time() - started) * 1000), "model": self.llm.model}
        )

    # ---------- 音色与试听 ----------
    def _preview_tts(self, engine: str = "") -> Any:
        """按当前配置造一个试听用的 TTS（语速/音调跟 persona 一致）。"""
        t = self.cfg.get("tts") or {}
        persona = self.cfg.get("persona") or {}
        sr = int((self.cfg.get("audio") or {}).get("downlink_sample_rate", 24000))
        if (engine or t.get("engine") or "edge") == "maoxiang":
            from .tts import pitch_value, speech_rate_factor

            return MaoxiangPreview(
                voice=t.get("voice") or "zh_female_vv_uranus_bigtts",
                sample_rate=sr,
                speech_rate=speech_rate_factor(self.cfg),
                pitch=pitch_value(self.cfg),
                loudness=float(t.get("loudness") or 0),
                emotion=str(t.get("emotion") or ""),
                emotion_scale=float(t.get("emotion_scale") or 0),
            )
        return PreviewTTS(
            t.get("voice") or "zh-CN-XiaoxiaoNeural",
            str(persona.get("speech_speed") or t.get("rate") or "+0%").strip(),
            t.get("volume", "+0%"),
            sr,
            str(persona.get("pitch") or t.get("pitch") or "+0Hz").strip(),
        )

    async def api_voices(self, request: web.Request) -> web.Response:
        """返回按引擎分组的音色目录、情绪表与当前选择。"""
        catalog = voice_catalog(self.cfg, str(MAOXIANG_VOICES_CACHE))
        return web.json_response(
            {
                "ok": True,
                **tts_view(self.cfg),
                "voices": EDGE_VOICES,
                "catalog": catalog,
            }
        )

    async def api_voices_refresh(self, request: web.Request) -> web.Response:
        """刷新音色表：maoxiang 重新下载，其余引擎只重新装配目录。"""
        engine = (self.cfg.get("tts") or {}).get("engine", "edge")
        try:
            data = await request.json()
        except Exception:
            data = {}
        if isinstance(data, dict) and data.get("engine"):
            engine = str(data["engine"])

        if engine == "maoxiang":
            from .tts_maoxiang import fetch_maoxiang_voices

            voices = await fetch_maoxiang_voices(MAOXIANG_VOICES_CACHE)
            self.hub.add_log("tts", f"猫剪音色表已刷新，共 {len(voices)} 个")
        else:
            voices = voice_catalog(self.cfg, str(MAOXIANG_VOICES_CACHE)).get(engine, {}).get("voices", [])
            self.hub.add_log("tts", f"{engine} 音色表已就绪，共 {len(voices)} 个")
        return web.json_response({"ok": True, "count": len(voices), "catalog": voice_catalog(self.cfg, str(MAOXIANG_VOICES_CACHE))})

    async def api_voices_test(self, request: web.Request) -> web.Response:
        """试听：合成一小段，返回 base64 的 WAV，浏览器可直接播。"""
        import base64

        try:
            data = await request.json()
        except Exception:
            data = {}
        if not isinstance(data, dict):
            data = {}
        voice = str(data.get("voice") or "").strip()
        engine = str(data.get("engine") or "").strip() or (self.cfg.get("tts") or {}).get("engine", "edge")
        catalog = voice_catalog(self.cfg, str(MAOXIANG_VOICES_CACHE))
        allowed = {v["id"] for v in (catalog.get(engine) or {}).get("voices", [])}
        if voice and allowed and voice not in allowed:
            return web.json_response(
                {"ok": False, "error": f"引擎 {engine} 不支持音色：{voice}"}, status=400
            )
        text = str(data.get("text") or "").strip() or "你好，我是小智，很高兴见到你。"
        if len(text) > 200:
            text = text[:200]

        started = time.time()
        try:
            tts = self._preview_tts(engine)
            pcm = await tts.synthesize_with(text, voice)
        except Exception as exc:
            log.warning("试听合成失败: %s", exc)
            return web.json_response({"ok": False, "error": f"合成失败：{exc}"}, status=500)
        if pcm is None or len(pcm) == 0:
            return web.json_response({"ok": False, "error": "合成结果为空"}, status=500)

        wav = pcm_to_wav_bytes(pcm, tts.sample_rate)
        duration = round(len(pcm) / float(tts.sample_rate), 2)
        self.hub.add_log("tts", f"试听 [{engine}] {voice or getattr(tts, 'voice', '')} {duration}s")
        return web.json_response(
            {
                "ok": True,
                "voice": voice or getattr(tts, "voice", ""),
                "engine": engine,
                "sample_rate": tts.sample_rate,
                "mime": "audio/wav",
                "audio_base64": base64.b64encode(wav).decode("ascii"),
                "duration_sec": duration,
                "elapsed_ms": int((time.time() - started) * 1000),
            }
        )

    # ---------- 对话历史 ----------
    # ---------- 定时任务 ----------
    def _schedule_payload(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """把控制台提交的提醒表单换算成 ScheduleStore.create 的参数。"""
        text = str(data.get("text") or "").strip()
        if not text:
            raise ValueError("要提醒什么？")
        if len(text) > 120:
            text = text[:120]
        repeat = str(data.get("repeat") or "once").strip().lower()
        if repeat not in ("once", "daily", "weekdays", "weekly"):
            repeat = "once"
        raw_when = str(data.get("when") or "").strip()
        if not raw_when:
            raise ValueError("要选个时间")
        try:
            base = datetime.fromisoformat(raw_when)
        except ValueError:
            raise ValueError("时间格式不对")
        if base.tzinfo is None:
            base = base.replace(tzinfo=CST)
        base = base.astimezone(CST)
        now = time.time()
        clock = ""
        weekday = None
        if repeat == "once":
            fire_at = base.timestamp()
            if fire_at <= now + 1:
                raise ValueError("这个时间已经过去了")
        else:
            clock = f"{base.hour:02d}:{base.minute:02d}"
            if repeat == "weekly":
                weekday = base.weekday()
            fire_at = compute_next(
                {
                    "repeat": repeat,
                    "clock": clock,
                    "weekday": weekday,
                    "fire_at": base.timestamp(),
                },
                now,
            )
        device_id = str(data.get("device_id") or "").strip()
        if not device_id:
            # 没指定就挂到最近在线的那台设备上
            try:
                for sess in self.hub.sessions.values():
                    sid = str(getattr(getattr(sess, "stats", None), "device_id", "") or "")
                    if sid:
                        device_id = sid
                        break
            except Exception:  # noqa: BLE001
                pass
        return {
            "text": text,
            "fire_at": fire_at,
            "repeat": repeat,
            "clock": clock,
            "weekday": weekday,
            "device_id": device_id,
            "kind": "alarm" if repeat != "once" else "remind",
            "created_by": "console",
        }

    async def api_schedules(self, request: web.Request) -> web.Response:
        """定时任务：GET 列表；POST 新建 / 删除 / 启停 / 立即试播。"""
        if request.method == "GET":
            items = [build_item_view(it) for it in self.schedules.list_items()]
            items.sort(key=lambda v: (not v["enabled"], v["fire_at"] or 0))
            return web.json_response(
                {"ok": True, "items": items, "status": self.schedule_svc.status()}
            )
        try:
            data = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "请求格式不对"}, status=400)
        action = str(data.get("action") or "create").strip()
        try:
            if action == "create":
                payload = self._schedule_payload(data)
                item = self.schedules.create(**payload)
                self.hub.add_log("schedule", f"新增提醒「{payload['text']}」")
                return web.json_response(
                    {
                        "ok": True,
                        "item": build_item_view(item),
                        "status": self.schedule_svc.status(),
                    }
                )
            if action == "delete":
                item = self.schedules.get(str(data.get("id") or ""))
                self.schedules.delete(str(data.get("id") or ""))
                self.hub.add_log("schedule", f"删除提醒「{item.get('text')}」")
                return web.json_response(
                    {"ok": True, "status": self.schedule_svc.status()}
                )
            if action == "toggle":
                tid = str(data.get("id") or "")
                self.schedules.set_enabled(tid, bool(data.get("enabled")))
                return web.json_response(
                    {"ok": True, "status": self.schedule_svc.status()}
                )
            if action == "test":
                text = str(data.get("text") or "").strip()
                if not text:
                    raise ValueError("要试播什么？")
                played = False
                for sess in list(self.hub.sessions.values()):
                    if getattr(sess, "_closed", True):
                        continue
                    try:
                        played = bool(await sess.speak_proactive(text, tag="test"))
                    except Exception as exc:  # noqa: BLE001
                        log.warning("试播失败: %s", exc)
                    break
                if not played:
                    return web.json_response(
                        {"ok": False, "error": "设备不在线，播不出去"}, status=409
                    )
                return web.json_response({"ok": True})
        except KeyError:
            return web.json_response({"ok": False, "error": "找不到这条提醒"}, status=404)
        except ValueError as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": False, "error": "未知操作"}, status=400)

    async def api_chats(self, request: web.Request) -> web.Response:
        """按日期分页读取对话历史。"""
        try:
            limit = int(request.query.get("limit", 100))
        except (TypeError, ValueError):
            limit = 100
        try:
            offset = int(request.query.get("offset", 0))
        except (TypeError, ValueError):
            offset = 0
        date = (request.query.get("date") or "").strip()
        try:
            result = self.store.list_items(date=date or None, limit=limit, offset=offset)
        except Exception as exc:
            log.warning("读取对话历史失败: %s", exc)
            return web.json_response({"ok": False, "error": str(exc)}, status=500)
        return web.json_response({"ok": True, **result})

    async def api_chat_sessions(self, request: web.Request) -> web.Response:
        """按会话聚合的对话历史：不带参数返回会话列表，带 session_id 返回该会话消息。"""
        sid = (request.query.get("session_id") or "").strip()
        try:
            if sid:
                messages = self.store.session_messages(sid)
                if not messages:
                    return web.json_response(
                        {"ok": False, "error": "没有这个会话"}, status=404
                    )
                for m in messages:  # 有原始录音的轮次给出播放地址
                    if m.get("audio"):
                        m["audio_url"] = f"/api/chats/{m.get('id', '')}/audio"
                return web.json_response(
                    {"ok": True, "session_id": sid, "messages": messages}
                )
            sessions = self.store.sessions()
        except Exception as exc:
            log.warning("读取会话列表失败: %s", exc)
            return web.json_response({"ok": False, "error": str(exc)}, status=500)
        return web.json_response(
            {"ok": True, "sessions": sessions, "total": len(sessions)}
        )

    async def api_chat_audio(self, request: web.Request) -> web.Response:
        """回放某一轮对话里用户当时的原始录音（wav）。"""
        item_id = (request.match_info.get("item_id") or "").strip()
        item = self.store.get_item(item_id) or {}
        path = self.store.audio_path(str(item.get("audio") or ""))
        if path is None:
            return web.json_response(
                {"ok": False, "error": "这条记录没有原始录音"}, status=404
            )
        return web.FileResponse(
            path, headers={"Cache-Control": "private, max-age=86400"}
        )

    async def api_chat_tts(self, request: web.Request) -> web.Response:
        """把一条历史消息重新合成语音，返回 base64 WAV 供网页播放。"""
        import base64

        try:
            data = await request.json()
        except Exception:
            data = {}
        if not isinstance(data, dict):
            data = {}
        text = str(data.get("text") or "").strip()
        item_id = str(data.get("id") or "").strip()
        if not text and item_id:
            item = self.store.get_item(item_id) or {}
            text = str(item.get("reply_text") or "").strip()
        if not text:
            return web.json_response({"ok": False, "error": "没有可合成的文本"}, status=400)
        if len(text) > 300:
            text = text[:300]

        engine = str((self.cfg.get("tts") or {}).get("engine") or "edge")
        if engine == "none":
            return web.json_response(
                {"ok": False, "error": "当前合成引擎为「不合成」"}, status=400
            )
        try:
            tts = self._preview_tts(engine)
            pcm = await tts.synthesize_with(text, "")
        except Exception as exc:
            log.warning("历史消息合成失败: %s", exc)
            return web.json_response({"ok": False, "error": f"合成失败：{exc}"}, status=500)
        if pcm is None or len(pcm) == 0:
            return web.json_response({"ok": False, "error": "合成结果为空"}, status=500)

        wav = pcm_to_wav_bytes(pcm, tts.sample_rate)
        duration = round(len(pcm) / float(tts.sample_rate), 2)
        self.hub.add_log("tts", f"历史回放合成 {duration}s（{len(text)} 字）")
        return web.json_response(
            {
                "ok": True,
                "wav": base64.b64encode(wav).decode("ascii"),
                "sample_rate": tts.sample_rate,
                "duration": duration,
            }
        )

    # ---------- 音乐 ----------

    def _music_session(self, device_id: str = "") -> Optional[Any]:
        """挑一台在线设备上的播放器。只有一个在线设备时不传 device_id 也行。"""
        players = [
            s
            for s in self.hub.sessions.values()
            if getattr(s, "music", None) is not None
        ]
        if device_id:
            for s in players:
                if str(getattr(s.stats, "device_id", "")) == device_id:
                    return s
            return None
        return players[0] if players else None

    def _music_devices(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for sid, s in self.hub.sessions.items():
            if getattr(s, "music", None) is None:
                continue
            out.append(
                {
                    "session_id": sid,
                    "device_id": str(getattr(s.stats, "device_id", "")),
                }
            )
        return out

    async def _json_body(self, request: web.Request) -> Dict[str, Any]:
        try:
            data = await request.json()
        except Exception:
            return {}
        return data if isinstance(data, dict) else {}

    async def api_music_status(self, request: web.Request) -> web.Response:
        device_id = (request.query.get("device_id") or "").strip()
        sess = self._music_session(device_id)
        return web.json_response(
            {
                "ok": True,
                "enabled": self.music_source is not None,
                "online": sess is not None,
                "devices": self._music_devices(),
                "player": sess.music.status() if sess is not None else None,
            }
        )

    async def api_music_search(self, request: web.Request) -> web.Response:
        if self.music_source is None:
            return web.json_response(
                {"ok": False, "error": "音乐功能没开（config.yaml 的 music.enabled）"},
                status=400,
            )
        data = await self._json_body(request)
        keyword = str(data.get("keyword") or "").strip()
        if not keyword:
            return web.json_response({"ok": False, "error": "关键词是空的"}, status=400)
        try:
            limit = int(data.get("limit") or 10)
        except (TypeError, ValueError):
            limit = 10
        limit = max(1, min(limit, 20))
        try:
            songs = await self.music_source.search(keyword, limit=limit)
        except Exception as exc:  # noqa: BLE001
            log.warning("搜索歌曲失败: %s", exc)
            return web.json_response(
                {"ok": False, "error": f"搜索失败：{exc}"}, status=502
            )
        return web.json_response({"ok": True, "keyword": keyword, "songs": songs})

    async def api_music_play(self, request: web.Request) -> web.Response:
        """控制中心点播：一首歌 / 一个关键词 / 整个歌单。"""
        if self.music_source is None:
            return web.json_response(
                {"ok": False, "error": "音乐功能没开（config.yaml 的 music.enabled）"},
                status=400,
            )
        data = await self._json_body(request)
        sess = self._music_session(str(data.get("device_id") or "").strip())
        if sess is None:
            return web.json_response(
                {"ok": False, "error": "设备不在线，先唤醒小智说句话连上再点播"},
                status=409,
            )
        player = sess.music
        songs: List[Dict[str, Any]] = []
        pl_id = str(data.get("playlist_id") or "").strip()
        song = data.get("song")
        keyword = str(data.get("keyword") or "").strip()
        if pl_id:
            pl = self.playlists.get(pl_id)
            if pl is None:
                return web.json_response(
                    {"ok": False, "error": "找不到这个歌单"}, status=404
                )
            songs = list(pl["songs"])
            if not songs:
                return web.json_response(
                    {"ok": False, "error": "这个歌单还是空的"}, status=400
                )
        elif isinstance(song, dict) and song.get("id"):
            songs = [song]
        elif keyword:
            try:
                songs = await self.music_source.search(keyword, limit=6)
            except Exception as exc:  # noqa: BLE001
                log.warning("点播搜索失败: %s", exc)
                return web.json_response(
                    {"ok": False, "error": f"搜索失败：{exc}"}, status=502
                )
            if not songs:
                return web.json_response(
                    {"ok": False, "error": f"没搜到「{keyword}」"}, status=404
                )
        else:
            return web.json_response(
                {"ok": False, "error": "要么给一首歌，要么给关键词，要么给歌单"},
                status=400,
            )
        try:
            index = int(data.get("index") or 0)
        except (TypeError, ValueError):
            index = 0
        index = max(0, min(index, len(songs) - 1))
        prepared = await player.prepare_song(songs[index])
        if prepared is None:
            return web.json_response(
                {"ok": False, "error": "这首歌拿不到音频（可能没版权），换一首试试"},
                status=502,
            )
        if getattr(sess, "_speaking", False):
            try:
                await sess.abort_speaking("控制中心点播")
            except Exception:  # noqa: BLE001
                pass
        # abort_speaking 会把 session._abort 置 True 让 TTS 立刻停发，但音乐要接着用
        # 同一条下行通道。这个标记不清掉，send_audio 会把开头几百毫秒的帧直接丢掉
        # （设备上就是点播后先静音一下才出声），推流循环也会以为被取消直接退出。
        sess._abort = False
        sess._clear_music_resume("控制中心点播")
        player.start(prepared, queue=songs, index=index)
        title = str((prepared.get("song") or {}).get("title") or "")
        self.hub.add_log("music", f"控制中心点播：{title}")
        return web.json_response({"ok": True, "player": player.status()})

    async def api_music_control(self, request: web.Request) -> web.Response:
        """暂停 / 继续 / 停止 / 上一首 / 下一首。"""
        data = await self._json_body(request)
        action = str(data.get("action") or "").strip()
        sess = self._music_session(str(data.get("device_id") or "").strip())
        if sess is None:
            return web.json_response(
                {"ok": False, "error": "设备不在线"}, status=409
            )
        player = sess.music
        if action == "pause":
            if not player.pause():
                return web.json_response(
                    {"ok": False, "error": "现在没在放歌"}, status=400
                )
            # 界面上的暂停态由推流循环的暂停分支下发（它会同时发恢复），这里不重复
        elif action == "resume":
            if not player.resume():
                # 暂停超时已经被当成「停止」了：用记下的进度重新起播，
                # 不然控制中心的播放键会变成死路（按了只弹「现在没在放歌」）
                if not await sess._restart_music(player, "控制中心继续"):
                    return web.json_response(
                        {"ok": False, "error": "现在没在放歌"}, status=400
                    )

        elif action == "stop":
            player.stop_nowait()
            # 刚语音点歌、话还没说完时这首歌挂在 pending 上，不清掉的话
            # 这轮 TTS 一结束 _start_pending_music 又把它放起来
            sess.pending_music = None
            sess._clear_music_resume("控制中心停止")
        elif action == "mode":
            mode = str(data.get("mode") or "").strip()
            if not player.set_mode(mode):
                return web.json_response(
                    {"ok": False, "error": f"不认识的播放模式：{mode}"}, status=400
                )
            # 模式对「下一首」的判断是即时的；正在放的那首不打断，播完自然按新模式走
        elif action in ("next", "prev"):
            if not player.queue:
                return web.json_response(
                    {"ok": False, "error": "现在没有播放列表"}, status=400
                )
            # 循环 / 随机 的边界判断都在 player 里，这里不要自己夹下标
            prepared = (
                await player.next_track() if action == "next" else await player.prev_track()
            )
            if prepared is None:
                return web.json_response(
                    {"ok": False, "error": "已经到头了" if action == "next" else "已经是最前面一首了"},
                    status=400,
                )
            sess._abort = False
            sess._clear_music_resume("控制中心换歌")
            player.start(prepared, queue=list(player.queue), index=player.index)
        else:
            return web.json_response(
                {"ok": False, "error": "未知操作"}, status=400
            )
        return web.json_response({"ok": True, "player": player.status()})

    async def api_music_playlists(self, request: web.Request) -> web.Response:
        return web.json_response({"ok": True, "items": self.playlists.list_items()})

    async def api_music_playlist(self, request: web.Request) -> web.Response:
        data = await self._json_body(request)
        action = str(data.get("action") or "").strip()
        pid = str(data.get("id") or "").strip()
        msg = ""
        item: Optional[Dict[str, Any]] = None
        try:
            if action == "create":
                item = self.playlists.create(
                    str(data.get("name") or ""), data.get("songs")
                )
                msg = f"新建歌单「{item['name']}」"
            elif action == "rename":
                item = self.playlists.rename(pid, str(data.get("name") or ""))
                msg = f"歌单改名为「{item['name']}」"
            elif action == "delete":
                self.playlists.delete(pid)
                msg = "删掉一个歌单"
            elif action == "add":
                item = self.playlists.add_song(pid, data.get("song") or {})
                msg = f"往歌单里加了「{(data.get('song') or {}).get('title')}」"
            elif action == "remove":
                item = self.playlists.remove_song(pid, str(data.get("song_id") or ""))
                msg = "从歌单里去掉一首"
            else:
                return web.json_response(
                    {"ok": False, "error": "未知操作"}, status=400
                )
        except KeyError:
            return web.json_response(
                {"ok": False, "error": "找不到这个歌单"}, status=404
            )
        except ValueError as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        if msg:
            self.hub.add_log("music", msg)
        return web.json_response(
            {"ok": True, "items": self.playlists.list_items(), "item": item}
        )

    async def api_chat_detail(self, request: web.Request) -> web.Response:
        """读取单条对话详情。"""
        item_id = (request.match_info.get("item_id") or "").strip()
        item = self.store.get_item(item_id)
        if item is None:
            return web.json_response({"ok": False, "error": "没有这条对话"}, status=404)
        return web.json_response({"ok": True, "item": item})

    # ---------- 长期记忆 ----------
    async def api_memory(self, request: web.Request) -> web.Response:
        """记忆库快照：画像 + 条目 + 统计 + 设置 + 后台状态。"""
        kind = (request.query.get("kind") or "").strip()
        keyword = (request.query.get("q") or "").strip()
        try:
            snap = self.memory.snapshot()
            if kind or keyword:
                snap["items"] = self.memory.list(kind=kind, keyword=keyword)
        except Exception as exc:
            log.warning("读取记忆失败: %s", exc)
            return web.json_response({"ok": False, "error": str(exc)}, status=500)
        snap["ok"] = True
        snap["total"] = len(snap["items"])
        snap["agent"] = self.memory_agent.status()
        return web.json_response(snap)

    async def api_memory_preview(self, request: web.Request) -> web.Response:
        """预览：这句话会带哪些记忆进模型（不写 last_used_at）。"""
        q = (request.query.get("q") or "").strip()
        try:
            block = self.memory.build_context(q, mark=False)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=500)
        return web.json_response(
            {"ok": True, "query": q, "block": block, "chars": len(block)}
        )

    async def api_memory_change(self, request: web.Request) -> web.Response:
        """记忆库写操作。action=add|update|delete|clear|profile|profile_bulk|reflect|settings"""
        try:
            data = await request.json()
        except Exception:
            data = {}
        action = str((data or {}).get("action") or "add").strip()

        if action == "add":
            text = str(data.get("text") or "").strip()
            if not text:
                return web.json_response({"ok": False, "error": "记忆内容不能为空"}, status=400)
            kind = str(data.get("kind") or "fact").strip()
            try:
                importance = float(data.get("importance", 0.6))
            except (TypeError, ValueError):
                importance = 0.6
            item = self.memory.add(text, kind=kind, importance=importance, source="manual")
            return web.json_response(
                {"ok": True, "item": item, "duplicate": item is None}
            )

        if action == "update":
            mid = str(data.get("id") or "").strip()
            if not mid:
                return web.json_response({"ok": False, "error": "缺少 id"}, status=400)
            item = self.memory.update(
                mid,
                text=data.get("text"),
                kind=data.get("kind"),
                importance=data.get("importance"),
            )
            if item is None:
                return web.json_response({"ok": False, "error": "没有这条记忆"}, status=404)
            return web.json_response({"ok": True, "item": item})

        if action == "delete":
            mid = str(data.get("id") or "").strip()
            if not mid:
                return web.json_response({"ok": False, "error": "缺少 id"}, status=400)
            ok = self.memory.remove(mid)
            return web.json_response({"ok": ok, "error": "" if ok else "没有这条记忆"})

        if action == "clear":
            keep_profile = bool(data.get("keep_profile"))
            n = self.memory.clear(keep_profile=keep_profile)
            return web.json_response({"ok": True, "removed": n})

        if action == "profile":
            field = str(data.get("field") or "").strip()
            value = str(data.get("value") or "").strip()
            changed = self.memory.set_profile_field(field, value, source="manual")
            return web.json_response(
                {"ok": True, "changed": bool(changed), "profile": self.memory.profile_items()}
            )

        if action == "profile_bulk":
            mapping = data.get("profile") or {}
            if not isinstance(mapping, dict):
                return web.json_response({"ok": False, "error": "profile 必须是对象"}, status=400)
            self.memory.set_profile(mapping, source="manual")
            return web.json_response({"ok": True, "profile": self.memory.profile_items()})

        if action == "reflect":
            if not self.memory.settings.get("auto", True):
                return web.json_response(
                    {"ok": False, "error": "自动记忆已关闭，先在设置里打开"}, status=400
                )
            queued = self.memory_agent.submit_reflect()
            return web.json_response(
                {"ok": True, "queued": bool(queued), "agent": self.memory_agent.status()}
            )

        if action == "settings":
            wanted = [k for k in ("auto", "reflect_every", "inject_max_chars") if k in data]
            if not wanted:
                return web.json_response({"ok": False, "error": "没有要改的设置"}, status=400)
            updated = {}
            for key in wanted:
                if self.memory.set_setting(key, data.get(key)):
                    updated[key] = self.memory.settings.get(key)
            return web.json_response(
                {"ok": True, "settings": updated, "saved": dict(self.memory.settings)}
            )

        return web.json_response({"ok": False, "error": f"未知操作：{action}"}, status=400)

    async def api_index(self, request: web.Request) -> web.StreamResponse:
        return web.FileResponse(STATIC_DIR / "index.html")


def create_app(cfg: Dict[str, Any]) -> web.Application:
    app_ctx = App(cfg)
    a = web.Application(client_max_size=16 * 1024 * 1024)
    a["ctx"] = app_ctx

    server = cfg.get("server") or {}
    ws_path = server.get("ws_path", "/xiaozhi/v1/")
    if not ws_path.startswith("/"):
        ws_path = "/" + ws_path

    a.router.add_post("/xiaozhi/ota/", app_ctx.handle_ota)
    a.router.add_get("/xiaozhi/ota/", app_ctx.handle_ota)
    a.router.add_get(ws_path, app_ctx.handle_ws)
    a.router.add_get("/api/status", app_ctx.api_status)
    a.router.add_route("*", "/api/config", app_ctx.api_config)
    a.router.add_post("/api/presets", app_ctx.api_presets)
    a.router.add_post("/api/test-preset", app_ctx.api_test_preset)
    a.router.add_post("/api/test-llm", app_ctx.api_test_llm)
    a.router.add_get("/api/voices", app_ctx.api_voices)
    a.router.add_post("/api/voices/test", app_ctx.api_voices_test)
    a.router.add_post("/api/voices/refresh", app_ctx.api_voices_refresh)
    a.router.add_get("/api/memory", app_ctx.api_memory)
    a.router.add_post("/api/memory", app_ctx.api_memory_change)
    a.router.add_get("/api/memory/preview", app_ctx.api_memory_preview)
    a.router.add_get("/api/personas", app_ctx.api_personas)
    a.router.add_post("/api/personas", app_ctx.api_personas)
    a.router.add_get("/api/schedules", app_ctx.api_schedules)
    a.router.add_post("/api/schedules", app_ctx.api_schedules)
    a.router.add_get("/api/music/status", app_ctx.api_music_status)
    a.router.add_post("/api/music/search", app_ctx.api_music_search)
    a.router.add_post("/api/music/play", app_ctx.api_music_play)
    a.router.add_post("/api/music/control", app_ctx.api_music_control)
    a.router.add_get("/api/music/playlists", app_ctx.api_music_playlists)
    a.router.add_post("/api/music/playlist", app_ctx.api_music_playlist)
    a.router.add_get("/api/chats", app_ctx.api_chats)
    a.router.add_get("/api/chat_sessions", app_ctx.api_chat_sessions)
    a.router.add_post("/api/chat_tts", app_ctx.api_chat_tts)
    a.router.add_get("/api/chats/{item_id}/audio", app_ctx.api_chat_audio)
    a.router.add_get("/api/chats/{item_id}", app_ctx.api_chat_detail)
    a.router.add_get("/", app_ctx.api_index)
    a.router.add_static("/static/", STATIC_DIR)

    async def _warm_voice_cache(app_: web.Application) -> None:
        """启动后台刷新猫剪音色表（失败不影响主服务）。"""
        from .tts_maoxiang import fetch_maoxiang_voices

        await asyncio.sleep(1.5)
        await fetch_maoxiang_voices(MAOXIANG_VOICES_CACHE)

    a.on_startup.append(_warm_voice_cache)

    async def _start_memory_agent(app_: web.Application) -> None:
        """启动后台记忆工作线程（每轮对话后异步抽取）。"""
        try:
            app_ctx.memory_agent.start()
        except Exception as exc:  # noqa: BLE001
            log.warning("后台记忆启动失败: %s", exc)

    a.on_startup.append(_start_memory_agent)

    async def _warm_bedbook(app_: web.Application) -> None:
        """启动后台把故事 MCP 子进程拉起来（冷启动 1.2 秒，别让孩子等）。"""
        try:
            from . import bedbook

            await asyncio.sleep(3.0)
            await bedbook.client().stories()
            log.info("故事服务已就绪")
        except Exception as exc:  # noqa: BLE001
            log.warning("故事服务预热失败: %s", exc)

    a.on_startup.append(_warm_bedbook)

    async def _stop_bedbook(app_: web.Application) -> None:
        try:
            from . import bedbook

            await bedbook.client().close()
        except Exception:  # noqa: BLE001
            pass

    a.on_cleanup.append(_stop_bedbook)

    async def _warm_story_audio(app_: web.Application) -> None:
        """后台预热 B 站客户端（首页 buvid + gaia 激活 + wbi key，冷启动约 1 秒）。"""
        try:
            from . import story_audio

            await asyncio.sleep(6.0)
            await story_audio.client().warmup()
            log.info("故事音频源已就绪")
        except Exception as exc:  # noqa: BLE001
            log.warning("故事音频预热失败: %s", exc)

    a.on_startup.append(_warm_story_audio)

    async def _stop_story_audio(app_: web.Application) -> None:
        try:
            from . import story_audio

            await story_audio.client().close()
        except Exception:  # noqa: BLE001
            pass

    a.on_cleanup.append(_stop_story_audio)

    async def _stop_memory_agent(app_: web.Application) -> None:
        try:
            await app_ctx.memory_agent.stop()
        except Exception:  # noqa: BLE001
            pass

    a.on_cleanup.append(_stop_memory_agent)

    async def _start_schedule(app_: web.Application) -> None:
        """启动定时任务扫描：每 5 秒看一次有没有到点的提醒。"""
        try:
            app_ctx.schedule_svc.start()
        except Exception as exc:  # noqa: BLE001
            log.warning("定时任务服务启动失败: %s", exc)

    a.on_startup.append(_start_schedule)

    async def _stop_schedule(app_: web.Application) -> None:
        try:
            await app_ctx.schedule_svc.stop()
        except Exception:  # noqa: BLE001
            pass

    a.on_cleanup.append(_stop_schedule)

    async def _close_tts_pool(app_: web.Application) -> None:
        """退出时把 TTS 预连接池里的连接关掉，别挂在服务端。"""
        closer = getattr(app_ctx.tts, "aclose", None)
        if callable(closer):
            try:
                await closer()
            except Exception:  # noqa: BLE001
                pass

    a.on_cleanup.append(_close_tts_pool)

    async def _close_music(app_: web.Application) -> None:
        """退出时关掉音乐源的 HTTP 会话。"""
        src = getattr(app_ctx, "music_source", None)
        closer = getattr(src, "close", None)
        if callable(closer):
            try:
                await closer()
            except Exception:  # noqa: BLE001
                pass

    a.on_cleanup.append(_close_music)
    return a


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)-8s %(message)s",
        datefmt="%H:%M:%S",
    )
    cfg = load_config()
    server = cfg.get("server") or {}
    host = server.get("host", "0.0.0.0")
    port = int(server.get("port", 8003))
    log.info("小智本地控制中心 v%s 启动于 http://%s:%d", __version__, host, port)
    log.info("OTA 端点: http://%s:%d/xiaozhi/ota/", server.get("public_host", host), port)
    web.run_app(create_app(cfg), host=host, port=port, print=None)


if __name__ == "__main__":
    main()