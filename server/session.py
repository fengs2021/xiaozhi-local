"""单个设备的 WebSocket 会话：音频上下行 + ASR/LLM/TTS 管线。

协议依据（xiaozhi-esp32 固件 main/protocols/websocket_protocol.cc）：
- 握手头 Authorization / Protocol-Version / Device-Id / Client-Id
- 设备 hello -> 服务器 hello（必须带 transport:"websocket"）
- version_=1 时二进制帧是裸 Opus 包，无头
- 设备在 Listening 状态持续推流，VAD 只在设备端驱动 LED，不自动停止
  -> 所以必须由服务器自己判断一句话结束
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .asr import is_filler
from .audio import OpusDownlinkEncoder, OpusUplinkDecoder
from .config import enabled_tools
from .llm import strip_for_speech
from .tools import TOOLS, call_tool

log = logging.getLogger("session")

# 上行音频固定 16kHz / int16 / 单声道（固件推流与 ASR 都按这个采样率）
UPLINK_SAMPLE_RATE = 16000

# 固定话术（对应上游 max_output_size.wav / close_connection_no_voice_time 的默认文案）
LIMIT_BYE_TEXT = "不好意思，我今天有点事情要忙，明天这个时候我们再聊吧，拜拜。"
# 听到这些词就别再续播音乐了（LLM 一般会调 stop_music，这里是兜底）
MUSIC_STOP_WORDS = ("别放", "不要放", "不听了", "关掉音乐", "关音乐", "停止播放", "别唱", "停一下歌")

# 语音切歌/改模式：ASR 出文本后先匹配这组词，命中就直接执行，不走模型（省一轮两秒）。
# 只在「刚被音乐打断的 30 秒内」或者音乐正在放的时候才认，免得孩子讲故事里冒出
# 一句「下一首」被吞掉。
MUSIC_CMD_WINDOW = 30.0
_MUSIC_CMD_STOP = MUSIC_STOP_WORDS + ("别放了", "停了吧", "关了吧")
_MUSIC_CMD_NEXT = ("下一首", "下首", "换一首", "换首歌", "换一个", "切歌", "换个歌")
_MUSIC_CMD_PREV = ("上一首", "前一首", "回到上一首")
_MUSIC_CMD_RESUME = ("继续播放", "接着放", "继续放", "接着听", "继续听")
_MUSIC_CMD_ONE = ("单曲循环", "循环这一首", "循环这首", "一直放这首", "重复播放")
_MUSIC_CMD_ALL = ("列表循环", "循环播放", "循环放", "全部循环")
_MUSIC_CMD_SHUFFLE = ("随机播放", "随便放", "乱序播放", "随机放")
_MUSIC_CMD_OFF = ("顺序播放", "放完就停", "不要循环", "取消循环")
_MUSIC_MODE_TEXT = {
    "off": "顺序播放",
    "one": "单曲循环",
    "all": "列表循环",
    "shuffle": "随机播放",
}


# 讲故事：分段合成、按序下发。每段一句话，句子之间留一点停顿，
# 听起来像在讲而不是在念稿。TTS 并发 3 条（与对话主链路同一个上限）。
STORY_GAP_SECONDS = 0.12
STORY_TTS_CONCURRENCY = 3
STORY_MAX_CHARS = 4000
_STORY_SENT_RE = re.compile(r"[^。！？!?；;\n]+[。！？!?；;]?")


def split_story_text(text: str, max_chars: int = 48) -> List[str]:
    """把故事正文切成适合逐句合成的短句。

    整篇一次丢给 TTS 要等它把两千字都合成完才出声（几十秒），孩子会以为
    设备没反应。这里优先按句末标点切；单句超过 max_chars 的再按逗号/顿号切，
    保证每段都能很快合成完、立刻开播。
    """
    out: List[str] = []
    for raw in _STORY_SENT_RE.findall(str(text or "")):
        seg = raw.strip()
        if not seg:
            continue
        while len(seg) > max_chars:
            cut = max(
                seg.rfind("，", 0, max_chars),
                seg.rfind("、", 0, max_chars),
                seg.rfind(",", 0, max_chars),
                seg.rfind("：", 0, max_chars),
            )
            if cut <= 4:
                cut = max_chars - 1
            head = seg[: cut + 1].strip()
            if head:
                out.append(head)
            seg = seg[cut + 1 :].strip()
        if seg:
            out.append(seg)
    return out


def _norm_music_cmd(text: Any) -> str:
    s = str(text or "")
    for ch in " \t\u3000，。！？、,.!?～~…—":
        s = s.replace(ch, "")
    return s


def match_music_command(text: str) -> Optional[tuple]:
    """把一句话解析成音乐指令：(action, value)；认不出返回 None。

    只在短句里认（≤10 个字）：「你能帮我放下一首歌吗」这种长句照常交给模型。
    """
    s = _norm_music_cmd(text)
    if not s or len(s) > 10:
        return None
    if any(w in s for w in _MUSIC_CMD_STOP):
        return ("stop", "")
    if any(w in s for w in _MUSIC_CMD_NEXT):
        return ("next", "")
    if any(w in s for w in _MUSIC_CMD_PREV):
        return ("prev", "")
    if any(w in s for w in _MUSIC_CMD_RESUME):
        return ("resume", "")
    if any(w in s for w in _MUSIC_CMD_ONE):
        return ("mode", "one")
    if any(w in s for w in _MUSIC_CMD_SHUFFLE):
        return ("mode", "shuffle")
    if any(w in s for w in _MUSIC_CMD_ALL):
        return ("mode", "all")
    if any(w in s for w in _MUSIC_CMD_OFF):
        return ("mode", "off")
    return None
IDLE_BYE_TEXT = "时间过得真快，那我先去忙啦，明天见。"


@dataclass
class SessionStats:
    session_id: str
    device_id: str = ""
    client_id: str = ""
    connected_at: float = field(default_factory=time.time)
    uplink_packets: int = 0
    downlink_packets: int = 0
    utterances: int = 0
    last_stt: str = ""
    last_llm: str = ""
    state: str = "idle"


class DeviceSession:
    """一个设备连接的生命周期。"""

    def __init__(self, ws, cfg: Dict[str, Any], services: Dict[str, Any]):
        self.ws = ws
        self.cfg = cfg
        self.services = services
        self.asr = services["asr"]
        self.tts = services["tts"]
        self.llm = services["llm"]
        self.make_vad = services["make_vad"]
        self.hub = services.get("hub")
        self.store = services.get("store")
        self.memory = services.get("memory")
        self.memory_agent = services.get("memory_agent")
        # 每日输出字数统计（儿童使用保护），由 app.py 注入
        self.counter = services.get("counter")

        audio_cfg = cfg.get("audio") or {}
        self.down_rate = int(audio_cfg.get("downlink_sample_rate", 24000))
        self.down_frame_ms = int(audio_cfg.get("downlink_frame_ms", 20))
        self.down_bitrate = int(audio_cfg.get("downlink_bitrate", 32000))

        self.stats = SessionStats(session_id=str(uuid.uuid4()))
        self.decoder = OpusUplinkDecoder(16000)
        self.encoder = OpusDownlinkEncoder(self.down_rate, self.down_frame_ms, self.down_bitrate)
        self.vad = self.make_vad()
        self.history: List[Dict[str, str]] = []

        self._listening = False
        self._speaking = False
        self._closed = False
        self._tasks: List[asyncio.Task] = []
        self._pipeline_lock = asyncio.Lock()
        self._abort = False
        self._last_audio_at = 0.0
        self._silence_task: Optional[asyncio.Task] = None
        # 正在跑/排队中的「ASR -> LLM -> TTS -> 下发」任务，打断时要能全部取消
        self._utterance_tasks: set = set()
        self._abort_reason = ""

        # 使用限制（limits 段，0 表示不限制）
        limits = cfg.get("limits") or {}
        self.max_chars_per_day = int(limits.get("max_output_chars_per_day", 0) or 0)
        self.idle_bye_seconds = float(limits.get("idle_bye_seconds", 0) or 0)
        self.wake_guard_seconds = float(limits.get("wake_guard_seconds", 0) or 0)
        self.wake_guard_min_samples = int(
            float(limits.get("wake_guard_min_ms", 0) or 0) * UPLINK_SAMPLE_RATE / 1000
        )
        self._listen_at = 0.0
        self._bye_sent = False

        # 停顿宽容：Silero 一旦听到 min_silence 的静音就判句末，孩子一句话中间的
        # 停顿比成人长得多，「今天的天气」和「合肥」会被切成两句。这里收到段之后
        # 先攒一小会儿（merge_grace_ms），期间又有新段就拼起来，等真的安静了再起管线。
        vad_cfg = cfg.get("vad") or {}
        self.merge_grace_ms = int(vad_cfg.get("merge_grace_ms", 0) or 0)
        self.merge_max_seconds = float(vad_cfg.get("merge_max_seconds", 10.0) or 10.0)
        self.merge_skip_seconds = float(vad_cfg.get("merge_skip_seconds", 3.0) or 3.0)
        self._pending_pcm: List[np.ndarray] = []
        self._pending_timer: Optional[asyncio.Task] = None

        # 在线音乐：MusicPlayer 要持有 session 引用，所以不在这里建，由 app 装配时挂上来。
        # pending_music 存「已经搜好、等这句话说完再放」的曲子。
        self.music: Any = None
        self.pending_music: Any = None
        # 刚停掉的歌：用户说「别放了」时 listen start 已经先把音乐停了，
        # stop_music 工具要靠这个才知道该回答「好，不放了」而不是「没在放歌」。
        self._music_recent: Optional[Dict[str, Any]] = None
        # 被打断时记下这首歌唱到哪儿了：这轮对话答完再从断点续播（孩子问完还想接着听）。
        # 说「别放了」或者点了别的歌就把这份记录清掉，不再续。
        self._music_resume: Optional[Dict[str, Any]] = None
        self._music_resume_task: Optional[asyncio.Task] = None
        self.music_resume_delay = 8.0

        # 讲故事：bedbook 故事 MCP 取回的长正文先挂在这里（pending_story），
        # 等「好呀，我们来讲《X》」这句 TTS 播完再分段合成、按序下发。
        # 整篇丢给 TTS 会先合成几十秒才开始出声，孩子以为没反应。
        self.pending_story: Any = None
        self._story_task: Optional[asyncio.Task] = None

        # 倒计时界面：可能同时有好几条在倒数（「3 分钟」里再补一句「10 分钟」）。
        # 屏幕永远显示最早到期的那条，它到点后把界面交还给下一条 —— 只存一条的话
        # 后来的会顶掉前面的，前面那条到点还会把后面那条的字一起收掉。
        self._countdowns: Dict[str, Dict[str, Any]] = {}
        self._countdown_shown: str = ""

    # ---------- 下行 ----------
    async def send_json(self, obj: Dict[str, Any]) -> None:
        try:
            await self.ws.send_str(json.dumps(obj, ensure_ascii=False))
        except Exception as exc:
            log.debug("send_json 失败: %s", exc)

    # ---------- 屏幕 UI 指令（倒计时）----------
    async def show_countdown(self, seconds: int, text: str = "", item_id: str = "") -> None:
        """让设备切到倒计时界面。纯显示，设备侧仍能监听唤醒词、随时插话。"""
        try:
            secs = int(seconds)
        except (TypeError, ValueError):
            return
        if secs <= 0:
            return
        key = str(item_id or "") or "anon-%.3f" % time.time()
        self._countdowns[key] = {
            "fire_at": time.time() + secs,
            "text": str(text or ""),
        }
        await self._refresh_countdown()

    async def hide_countdown(self, item_id: str = "") -> None:
        """某条倒计时结束或被取消：从账上划掉，界面交给还剩下的最早那条。

        传 item_id 只动那一条（别的提醒到点不会把这条的界面收掉）；
        不传就整批收掉，留给外部直接清屏的场景。
        """
        if item_id:
            self._countdowns.pop(str(item_id), None)
        else:
            self._countdowns.clear()
        await self._refresh_countdown()

    async def _refresh_countdown(self) -> None:
        """按账上剩下的倒计时重画屏幕：永远显示最早到期的那条。"""
        now = time.time()
        for key in [k for k, v in self._countdowns.items() if v["fire_at"] <= now]:
            # 到点这件事由 schedule 负责播报，这里只把过期的账划掉
            self._countdowns.pop(key, None)
        if not self._countdowns:
            if self._countdown_shown:
                self._countdown_shown = ""
                await self.send_json({"type": "ui", "action": "countdown_hide"})
            return
        key = min(self._countdowns, key=lambda k: self._countdowns[k]["fire_at"])
        entry = self._countdowns[key]
        remaining = max(1, int(round(entry["fire_at"] - now)))
        self._countdown_shown = key
        await self.send_json({
            "type": "ui",
            "action": "countdown",
            "seconds": remaining,
            "text": entry["text"],
            "id": key,
        })

    # ---------- 屏幕 UI 指令（音乐）----------
    async def show_music(self, title: str, artist: str = "", duration: int = 0,
                         position: int = 0, paused: bool = False,
                         mode: str = "") -> None:
        """让设备把屏切到「正在放歌」。纯显示，不碰音频链路。"""
        await self.send_json({
            "type": "ui",
            "action": "music_show",
            "title": str(title or ""),
            "artist": str(artist or ""),
            "duration": int(duration or 0),
            "position": int(position or 0),
            "paused": bool(paused),
            "mode": str(mode or ""),
            "mode_text": _MUSIC_MODE_TEXT.get(str(mode or ""), ""),
        })

    async def update_music(self, position: int = 0, duration: int = 0,
                           paused: bool = False) -> None:
        """只校准进度/暂停态，不动歌名。"""
        await self.send_json({
            "type": "ui",
            "action": "music_update",
            "duration": int(duration or 0),
            "position": int(position or 0),
            "paused": bool(paused),
        })

    async def hide_music(self) -> None:
        await self.send_json({"type": "ui", "action": "music_hide"})

    async def send_audio(self, payload: bytes) -> None:
        if self._abort:
            return
        try:
            await self.ws.send_bytes(payload)
            self.stats.downlink_packets += 1
        except Exception as exc:
            log.debug("send_audio 失败: %s", exc)

    # ---------- 上行 ----------
    def _uplink_step(self, data: bytes) -> list:
        """上行一步到位：Opus 解码 + VAD 判定在同一线程内完成。

        旧代码 decode 和 vad.push 各走一次线程池往返（每包两次调度），
        设备 60ms 一包时这些调度开销会挤占 VAD 判定的时间精度。
        """
        pcm = self.decoder.decode(data)
        if pcm.size == 0:
            return []
        return self.vad.push(pcm)

    async def on_binary(self, data: bytes) -> None:
        if not data:
            return
        self.stats.uplink_packets += 1
        self._last_audio_at = time.time()
        segments = await asyncio.get_running_loop().run_in_executor(
            None, self._uplink_step, data
        )
        if segments and self.wake_guard_seconds > 0:
            # 刚进入聆听的两秒内，设备唤醒提示音、机身共振和房间回声很容易被 VAD
            # 判成一句「语音」。这段时间里只丢掉过短的段，用户正常开口（段更长）不受影响。
            if time.time() - self._listen_at < self.wake_guard_seconds:
                kept = [s for s in segments if s.size >= self.wake_guard_min_samples]
                dropped = len(segments) - len(kept)
                if dropped:
                    log.info("[%s] 唤醒免疫：丢弃 %d 段过短语音", self.stats.session_id[:8], dropped)
                segments = kept
        if segments and self.music is not None:
            # 真听到人声才停音乐。设备的 auto 模式在 TTS 结束后会立刻发 listen start，
            # 那里不能停——否则刚点好的歌一启动就被掐掉（14:03:55 实测就是这个）。
            self._stop_music("听到说话")
        for seg in segments:
            self._queue_segment(seg)

    async def on_json(self, msg: Dict[str, Any]) -> None:
        mtype = msg.get("type")
        if mtype == "listen":
            state = msg.get("state")
            if state == "start":
                self._listening = True
                self.stats.state = "listening"
                self._abort = False
                self._abort_reason = ""
                self._last_audio_at = time.time()
                self._listen_at = time.time()
                self._bye_sent = False
                self._drop_pending()
                self.vad.reset()
                self._start_watchdog()
                log.info("[%s] 开始聆听 mode=%s", self.stats.session_id[:8], msg.get("mode"))
            elif state == "stop":
                self._listening = False
                self.stats.state = "idle"
                self._cancel_watchdog()
                # 设备主动停止：把残留语音也处理掉
                segs = await asyncio.get_running_loop().run_in_executor(None, self.vad.flush)
                for seg in segs:
                    self._queue_segment(seg)
                self._flush_pending("停止聆听")
                log.info("[%s] 停止聆听", self.stats.session_id[:8])
        elif mtype == "abort":
            log.info("[%s] 收到 abort: %s", self.stats.session_id[:8], msg.get("reason"))
            await self.abort_speaking(str(msg.get("reason") or ""))
        elif mtype == "mcp":
            await self._handle_mcp(msg.get("payload") or {})
        elif mtype == "hello":
            pass
        else:
            log.debug("[%s] 未处理消息: %s", self.stats.session_id[:8], mtype)

    async def _handle_mcp(self, payload: Dict[str, Any]) -> None:
        """MCP 应答：initialize / tools/list / tools/call。"""
        method = payload.get("method")
        req_id = payload.get("id")

        # 通知类消息没有 id，也不需要回复
        if method in ("notifications/initialized", "initialized", "notifications/cancelled"):
            return
        if req_id is None and method and method.startswith("notifications/"):
            return

        if method == "initialize":
            await self._send_mcp_result(
                req_id,
                {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "xiaozhi-local", "version": "0.1.0"},
                },
            )
            return

        if method == "tools/list":
            # 按 config.yaml 的 mcp.enabled_tools 过滤，缺省全开
            allowed = set(enabled_tools(self.cfg))
            tools = [t for t in TOOLS if t.get("name") in allowed]
            await self._send_mcp_result(req_id, {"tools": tools})
            return

        if method == "tools/call":
            params = payload.get("params") or {}
            name = str(params.get("name") or "").strip()
            args = params.get("arguments") or {}
            if not isinstance(args, dict):
                args = {}
            allowed = set(enabled_tools(self.cfg))
            if name not in allowed:
                await self._send_mcp_result(
                    req_id,
                    {
                        "content": [{"type": "text", "text": f"工具 {name} 未启用。"}],
                        "isError": True,
                    },
                )
                return
            # 注入运行上下文，get_device_status 需要 hub/session
            call_args = dict(args)
            call_args["_ctx"] = {"hub": self.hub, "session": self}
            result = await call_tool(name, call_args)
            if self.hub is not None:
                self.hub.add_log("mcp", f"调用工具 {name}: {str(result.get('text'))[:60]}")
            await self._send_mcp_result(
                req_id,
                {
                    "content": [{"type": "text", "text": str(result.get("text", ""))}],
                    "isError": bool(result.get("is_error")),
                },
            )
            return

        # 未知方法按 JSON-RPC 规范返回 -32601
        await self.send_json(
            {
                "type": "mcp",
                "payload": {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32601, "message": "Method not found"},
                },
            }
        )

    async def _send_mcp_result(self, req_id: Any, result: Dict[str, Any]) -> None:
        await self.send_json(
            {"type": "mcp", "payload": {"jsonrpc": "2.0", "id": req_id, "result": result}}
        )

    # ---------- 静音看门狗 ----------
    def _start_watchdog(self) -> None:
        """设备端 VAD 只驱动 LED，不会发 listen stop；
        若用户说完后麦克风安静下来，靠 VAD 自己收句；
        若 VAD 因故没收到足够静音（例如设备掉包），这里兜底 flush。"""
        self._cancel_watchdog()
        self._silence_task = asyncio.create_task(self._watchdog())

    def _cancel_watchdog(self) -> None:
        if self._silence_task is not None:
            self._silence_task.cancel()
            self._silence_task = None

    async def _watchdog(self) -> None:
        try:
            while not self._closed:
                await asyncio.sleep(0.5)
                if not self._listening or self._pipeline_lock.locked():
                    continue
                idle = time.time() - self._last_audio_at
                # 长时间没人说话：说一句告别话再断开（儿童场景，避免设备一直挂着）。
                # 但如果这台设备上还压着没触发的提醒/闹钟，绝不能告别——
                # 夜里没人说话两分钟就断线的话，早上 7 点的闹钟永远等不到。
                if self.idle_bye_seconds > 0 and not self._bye_sent and not self._speaking:
                    quiet = time.time() - max(self._last_audio_at, self.stats.connected_at)
                    if quiet >= self.idle_bye_seconds:
                        if self._has_pending_schedule():
                            # 每 0.5 秒走到这里，用 DEBUG 免得刷屏
                            log.debug(
                                "[%s] 静默 %.0fs，但有未触发的定时任务，保持在线",
                                self.stats.session_id[:8],
                                quiet,
                            )
                        else:
                            self._bye_sent = True
                            log.info(
                                "[%s] 已静默 %.0fs，主动告别",
                                self.stats.session_id[:8],
                                quiet,
                            )
                            self._spawn(self._say_and_hangup(IDLE_BYE_TEXT, "空闲告别"))
                            return
                # 1.2 秒没有任何上行包，且 VAD 认为还在说话 -> 强制收句
                if idle >= 1.2 and self.vad.is_speech:
                    segs = await asyncio.get_running_loop().run_in_executor(
                        None, self.vad.flush
                    )
                    if segs:
                        log.info("[%s] 静音超时，强制收句 %d 段", self.stats.session_id[:8], len(segs))
                    for seg in segs:
                        self._queue_segment(seg)
                    self._flush_pending("静音超时")
        except asyncio.CancelledError:
            pass

    # ---------- 停顿合并 ----------
    def _queue_segment(self, seg) -> None:
        """收到一段语音：先攒着，等一小段安静再起管线。

        merge_grace_ms <= 0 时退化为原来的「来一段处理一段」。
        """
        if seg is None or seg.size == 0:
            return
        if self.merge_grace_ms <= 0:
            self._spawn(self._handle_utterance(seg))
            return
        self._pending_pcm.append(seg)
        total = sum(s.size for s in self._pending_pcm) / UPLINK_SAMPLE_RATE
        # 攒太长了就别再等，否则用户说完半天没反应
        if total >= self.merge_max_seconds:
            self._flush_pending("已到合并上限")
            return
        # 单独一段就已经说了挺久：这种长度后面跟的停顿基本是真说完了，
        # 不用再等合并窗口，直接处理，别让长句子白等 600ms。
        if len(self._pending_pcm) == 1 and total >= self.merge_skip_seconds:
            self._flush_pending()
            return
        if self._pending_timer is not None:
            self._pending_timer.cancel()
        self._pending_timer = asyncio.create_task(self._pending_flush())

    async def _pending_flush(self) -> None:
        try:
            await asyncio.sleep(self.merge_grace_ms / 1000.0)
        except asyncio.CancelledError:
            return
        self._pending_timer = None
        self._flush_pending()

    def _flush_pending(self, reason: str = "") -> None:
        """把攒着的段拼成一句，立刻起管线。"""
        if self._pending_timer is not None:
            self._pending_timer.cancel()
            self._pending_timer = None
        if not self._pending_pcm:
            return
        parts = self._pending_pcm
        self._pending_pcm = []
        seg = parts[0] if len(parts) == 1 else np.concatenate(parts)
        if len(parts) > 1:
            log.info(
                "[%s] 停顿合并 %d 段为一句（%.2fs%s）",
                self.stats.session_id[:8],
                len(parts),
                seg.size / UPLINK_SAMPLE_RATE,
                "，" + reason if reason else "",
            )
        self._spawn(self._handle_utterance(seg))

    def _drop_pending(self) -> None:
        """丢掉攒着的段（打断、重连时用，避免旧语音冒出来）。"""
        if self._pending_timer is not None:
            self._pending_timer.cancel()
            self._pending_timer = None
        self._pending_pcm = []

    # ---------- 管线 ----------
    def _spawn(self, coro) -> None:
        t = asyncio.create_task(coro)
        self._tasks.append(t)
        t.add_done_callback(lambda x: self._tasks.remove(x) if x in self._tasks else None)

    # ---------- 固定话术 / 收尾 ----------
    def _limit_reached(self) -> bool:
        """今天的输出字数是否已经到顶。"""
        if self.counter is None or self.max_chars_per_day <= 0:
            return False
        try:
            return self.counter.used(self.stats.device_id) >= self.max_chars_per_day
        except Exception as exc:  # noqa: BLE001
            log.warning("[%s] 读取输出计数失败: %s", self.stats.session_id[:8], exc)
            return False

    async def _speak_text(self, text: str) -> None:
        """不经 LLM，直接把一句话合成下发（告别、限额提示这类固定话术）。"""
        if self._closed:
            return
        self._speaking = True
        self.stats.state = "speaking"
        # 必须按帧节拍推：一次性把 2 秒音频灌给设备，它会当成缓冲溢出丢掉，
        # 听感就是「只说了一个字」。与正常对话的 play_all 用同一口径 20ms/帧。
        interval = self.down_frame_ms / 1000.0
        next_at = time.monotonic()
        try:
            await self.send_json({"type": "tts", "state": "start"})
            await self.send_json({"type": "llm", "emotion": "neutral", "text": text})
            await self.send_json({"type": "tts", "state": "sentence_start", "text": text})
            async for chunk in self.tts.stream(text):
                if self._closed or self._abort:
                    break
                if chunk is None or getattr(chunk, "size", 0) == 0:
                    continue
                for payload in self.encoder.encode(chunk):
                    if self._closed or self._abort:
                        break
                    await self.send_audio(payload)
                    next_at += interval
                    delay = next_at - time.monotonic()
                    if delay > 0:
                        await asyncio.sleep(delay)
                    else:
                        next_at = time.monotonic()
            if not (self._closed or self._abort):
                for payload in self.encoder.flush():
                    await self.send_audio(payload)
                    next_at += interval
                    delay = next_at - time.monotonic()
                    if delay > 0:
                        await asyncio.sleep(delay)
        finally:
            self._speaking = False
            if self.stats.state == "speaking":
                self.stats.state = "idle"
            try:
                await self.send_json({"type": "tts", "state": "stop"})
            except Exception:  # noqa: BLE001
                pass
            self._start_pending_music()

    def _has_pending_schedule(self) -> bool:
        """这台设备上还有没有没触发的提醒/闹钟。"""
        svc = self.services.get("schedule")
        if svc is None:
            return False
        try:
            return bool(svc.store.has_pending_for(self.stats.device_id))
        except Exception:  # noqa: BLE001
            return False

    def _stop_music(self, reason: str = "") -> None:
        """停掉正在放的歌（打断、开始聆听、关会话时调）。"""
        player = self.music
        pending, self.pending_music = self.pending_music, None
        if player is None:
            return
        try:
            current = getattr(player, "current", None)
            if getattr(player, "playing", False) or current or pending:
                song = current or (pending or {}).get("song")
                # 被打断的那首记下进度，等这轮对话说完再从断点接着放。
                # 只记 3 秒之后的：歌才刚开头就被叫停，多半是换主意了。
                pos = float(getattr(player, "position", 0.0) or 0.0)
                if reason == "打断":
                    # 连队列一起记：续播以后按「下一首」还能接着走原歌单，
                    # 只记一首的话续播完就成单曲了。
                    self._music_resume = (
                        {
                            "song": dict(song),
                            "position": pos,
                            "queue": list(getattr(player, "queue", []) or []),
                            "index": int(getattr(player, "index", 0) or 0),
                        }
                        if song and pos >= 3.0
                        else None
                    )
                else:
                    self._music_resume = None
                if song:
                    self._music_recent = {"song": song, "at": time.time()}
                player.stop_nowait()
                log.info("[%s] 停止音乐：%s", self.stats.session_id[:8], reason or "-")
        except Exception as exc:  # noqa: BLE001
            log.debug("停止音乐失败: %s", exc)

    def record_music_resume(self, position: float) -> None:
        """暂停太久被当成「停止」时记下进度：之后说「继续播放」还能接上。"""
        player = self.music
        song = getattr(player, "current", None) if player is not None else None
        if not song:
            return
        self._music_resume = {
            "song": dict(song),
            "position": max(0.0, float(position or 0.0)),
            "queue": list(getattr(player, "queue", []) or []),
            "index": int(getattr(player, "index", 0) or 0),
        }
        self._music_recent = {"song": dict(song), "at": time.time()}

    async def _music_reply_fallback(self, reply: str, llm: Any) -> None:
        """模型只口头答应「这就给你放《X》」却没调 play_music 时，按回复文本补一次。

        实测 24 次点歌里 6 次模型漏调工具（工具清单涨到 22 个之后更容易漏），
        这时歌名已经说给孩子听了，不响就是「点了没反应」。这里按文本兜底：
        只在本轮确实没调过 play_music、且回复里已经报了歌名时才补。
        """
        if self.music is None or self.pending_music is not None:
            return
        if self._abort or self._closed:
            return
        if "play_music" not in (getattr(llm, "tool_names", ()) or ()):
            return
        if "play_music" in (getattr(llm, "last_tool_names", []) or []):
            return
        text = str(reply or "")
        if not re.search(r"播放|放一首|放首|来一首|来点|这就放|接着放|给你放|给你播", text):
            return
        # 优先认书名号/引号里的歌名：「这就给你放《橡皮》」，原样取用——
        # 《加油呀》的「呀」就是歌名的一部分，不能当语气词削掉。
        match = re.search(r"[《「\"“]([^》」\"”]{1,30})[》」\"”]", text)
        if match is not None:
            title = match.group(1).strip()
        else:
            # 再退一步认紧跟「播放/放首」的裸歌名：「播放稻香」
            bare = re.search(
                r"(?:播放|放一首|放首|来一首|来首)([^，。！？、,.\s《》「」\"”]{1,20})", text
            )
            title = bare.group(1).strip().rstrip("吧呢啊呀哦嘛啦") if bare else ""
            # 裸歌名容易误抓：「现在播放时间到了」→「时间到了」、「放点什么」→「点什么」
            if title.startswith(("点", "时间", "完", "结束", "起来", "什么", "啥")):
                title = ""
            if title in {"歌", "首歌", "歌曲", "曲子", "音乐", "点儿"}:
                title = ""
        if not title:
            return
        log.info(
            "[%s] 模型没调 play_music，按回复补播：%s", self.stats.session_id[:8], title
        )
        try:
            result = await call_tool(
                "play_music", {"keyword": title, "_ctx": {"session": self}}
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("兜底点歌失败: %s", exc)
            return
        if result.get("is_error") or self.pending_music is None:
            log.warning("兜底点歌没生效: %s", result.get("text"))

    def _start_pending_music(self) -> None:
        """这句话已经说完，把刚才点好的歌接上。

        点歌时不能立刻开播：工具返回的「好，给你放稻香」还要走 TTS 下发，
        两路 opus 同时推设备会互相盖掉，所以先挂在 pending_music 上，
        等 TTS 收尾再真正开始。
        """
        prepared, self.pending_music = self.pending_music, None
        if not prepared or self.music is None:
            return
        # 已经开始放新的歌了，之前那份「续播进度」作废
        self._clear_music_resume("换了新歌")
        try:
            # 关键词点歌时 prepared 里带着搜索出来的整张候选表（queue）和命中位置，
            # 交给播放器，「下一首」才能顺着候选表往下走。
            self.music.start(
                prepared,
                queue=list(prepared.get("queue") or []) or None,
                index=int(prepared.get("index") or 0),
            )
            log.info(
                "[%s] 开始放歌：%s",
                self.stats.session_id[:8],
                (prepared.get("song") or {}).get("title"),
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("启动音乐失败: %s", exc)

    # ---------- 讲故事（bedbook 故事 MCP） ----------
    def queue_story(self, story: Dict[str, Any]) -> None:
        """工具取回故事后挂在这里：等「好呀，我们来讲《X》」播完再开讲。"""
        title = str((story or {}).get("title") or "").strip()
        text = str((story or {}).get("text") or "").strip()
        if not text:
            return
        self.pending_story = {"title": title, "text": text}
        log.info(
            "[%s] 待讲故事：%s（%d 字）",
            self.stats.session_id[:8],
            title or "-",
            len(text),
        )

    def _start_pending_story(self) -> None:
        """这句话已经说完，把挂着的故事分段播出去。"""
        story, self.pending_story = self.pending_story, None
        if not story or self._closed:
            return
        task = asyncio.create_task(self._play_story(story), name="story")
        self._story_task = task
        self._tasks.append(task)
        task.add_done_callback(lambda x: self._tasks.remove(x) if x in self._tasks else None)

    def _stop_story(self, reason: str = "") -> bool:
        """停掉正在播的故事（打断、插话问问题、关会话时调）。"""
        task, self._story_task = self._story_task, None
        pending, self.pending_story = self.pending_story, None
        if task is not None and not task.done():
            task.cancel()
            log.info("[%s] 停止讲故事：%s", self.stats.session_id[:8], reason or "-")
            return True
        if pending is not None:
            log.info("[%s] 取消待讲故事：%s", self.stats.session_id[:8], reason or "-")
            return True
        return False

    async def _play_story(self, story: Dict[str, Any]) -> None:
        """把一篇长故事分段合成、按顺序下发。

        与对话主链路同一套「并行合成 + 按序播放」口径：每句一个队列，合成端
        最多 3 句并行往队列里推，播放端按原顺序消费，第一句合成完就能出声。
        孩子说唤醒词或插话时任务被 cancel，残留句子在 finally 里一起取消。
        """
        title = str((story or {}).get("title") or "").strip()
        text = str((story or {}).get("text") or "").strip()
        sentences = [
            s for s in (strip_for_speech(x).strip() for x in split_story_text(text[:STORY_MAX_CHARS])) if s
        ]
        if not sentences:
            return
        sid = self.stats.session_id[:8]
        self._speaking = True
        self.stats.state = "speaking"
        interval = self.down_frame_ms / 1000.0
        tts_pool = asyncio.Semaphore(STORY_TTS_CONCURRENCY)
        queues: List[asyncio.Queue] = [asyncio.Queue() for _ in sentences]
        tts_started = False
        first_audio_at: Optional[float] = None
        total_packets = 0
        total_samples = 0
        t0 = time.time()

        async def _synth(idx: int, sent: str) -> None:
            queue = queues[idx]
            async with tts_pool:
                try:
                    async for chunk in self.tts.stream(sent):
                        if self._closed:
                            break
                        if getattr(chunk, "size", 0):
                            queue.put_nowait(chunk)
                except Exception as exc:  # noqa: BLE001
                    log.warning("[%s] 故事分段合成失败: %s", sid, exc)
                finally:
                    queue.put_nowait(None)

        tasks = [asyncio.create_task(_synth(i, s)) for i, s in enumerate(sentences)]
        try:
            log.info("[%s] 开始讲故事：%s（%d 句）", sid, title or "-", len(sentences))
            for idx, sent in enumerate(sentences):
                queue = queues[idx]
                sent_started = False
                next_at = time.monotonic()
                while True:
                    if self._closed or self._abort:
                        return
                    try:
                        # 带超时轮询：合成端可能卡在信号量上，不能死等
                        chunk = await asyncio.wait_for(queue.get(), timeout=0.25)
                    except asyncio.TimeoutError:
                        continue
                    if chunk is None:
                        break
                    if not sent_started:
                        sent_started = True
                        if not tts_started:
                            tts_started = True
                            await self.send_json({"type": "tts", "state": "start"})
                        await self.send_json(
                            {"type": "tts", "state": "sentence_start", "text": sent}
                        )
                        next_at = time.monotonic()
                    for payload in self.encoder.encode(chunk):
                        if self._closed or self._abort:
                            return
                        await self.send_audio(payload)
                        total_packets += 1
                        total_samples += int(getattr(chunk, "size", 0) or 0)
                        if first_audio_at is None:
                            first_audio_at = time.time()
                        next_at += interval
                        delay = next_at - time.monotonic()
                        if delay > 0:
                            await asyncio.sleep(delay)
                        else:
                            next_at = time.monotonic()
                # 句子收尾：把不足一帧的残余补零发出去
                if not (self._closed or self._abort):
                    for payload in self.encoder.flush():
                        await self.send_audio(payload)
                        total_packets += 1
                if STORY_GAP_SECONDS > 0:
                    await asyncio.sleep(STORY_GAP_SECONDS)
        except asyncio.CancelledError:
            log.info("[%s] 故事被打断：%s", sid, title or "-")
            raise
        except Exception as exc:  # noqa: BLE001
            log.warning("[%s] 讲故事失败: %s", sid, exc)
        finally:
            for t in tasks:
                if not t.done():
                    t.cancel()
            if tts_started:
                try:
                    await self.send_json({"type": "tts", "state": "stop"})
                except Exception:  # noqa: BLE001
                    pass
            self._speaking = False
            if self.stats.state == "speaking":
                self.stats.state = "idle"
            if self._story_task is asyncio.current_task():
                self._story_task = None
            log.info(
                "[%s] 讲完故事：%s（共 %d 包 %.2fs 音频，用时 %.1fs）",
                sid,
                title or "-",
                total_packets,
                total_samples / float(self.down_rate or 1),
                time.time() - t0,
            )

    # ---------- 打断后的续播 ----------
    def _cancel_music_resume(self) -> None:
        """只取消等待中的任务，进度留着 —— 这轮对话结束后还会再排一次。"""
        task, self._music_resume_task = self._music_resume_task, None
        if task is not None and not task.done():
            task.cancel()

    def _clear_music_resume(self, reason: str = "") -> None:
        """用户明确改主意了（点了别的歌、说别放了、关了音乐），别再续播。"""
        if self._music_resume is not None:
            log.info(
                "[%s] 取消续播（%s）",
                self.stats.session_id[:8],
                reason or "-",
            )
        self._music_resume = None
        self._cancel_music_resume()

    def _schedule_music_resume(self) -> None:
        """一轮对话答完时调用：等 music_resume_delay 秒没人说话，就从断点续播。"""
        self._cancel_music_resume()
        if not self._music_resume or self._closed or self.music is None:
            return
        try:
            self._music_resume_task = asyncio.create_task(
                self._resume_music_later(), name="music-resume"
            )
        except RuntimeError:  # 没有事件循环（关会话途中），忽略
            self._music_resume_task = None

    async def _resume_music_later(self) -> None:
        try:
            await asyncio.sleep(float(self.music_resume_delay))
            if not self._music_resume or self._closed or self.music is None:
                return
            # 这期间又开了新一轮对话：数据留着，等这轮答完 _schedule_music_resume 再排一次
            if self._speaking or self._pipeline_lock.locked():
                return
            data, self._music_resume = self._music_resume, None
            player = self.music
            if getattr(player, "playing", False):
                return
            song = data.get("song") or {}
            prepared = await player.prepare_song(song)
            if not prepared:
                log.info("[%s] 续播失败：拿不到音频「%s」", self.stats.session_id[:8],
                         song.get("title"))
                return
            queue = list(data.get("queue") or []) or [song]
            try:
                index = int(data.get("index") or 0)
            except (TypeError, ValueError):
                index = 0
            player.start(
                prepared,
                queue=queue,
                index=index,
                offset_seconds=float(data.get("position") or 0.0),
            )
            log.info(
                "[%s] 从 %.1fs 处续播「%s」",
                self.stats.session_id[:8],
                float(data.get("position") or 0.0),
                song.get("title"),
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            log.warning("[%s] 续播失败: %s", self.stats.session_id[:8], exc)

    # ---------- 语音切歌 / 改模式 ----------
    def _fill_queue_from_recent(self, player: Any) -> None:
        """打断后队列通常还在；续播成单曲的情形下，把最近那首补回队列。"""
        if getattr(player, "queue", None):
            return
        song = (self._music_recent or {}).get("song") or {}
        if song:
            player.queue = [song]
            player.index = 0

    async def _restart_music(self, player: Any, reason: str) -> bool:
        """用打断时记下的进度（没有就用最近那首）重新起播。"""
        data = self._music_resume
        self._music_resume = None
        self._cancel_music_resume()
        song = (data or {}).get("song") or (self._music_recent or {}).get("song") or {}
        if not song:
            return False
        prepared = await player.prepare_song(song)
        if prepared is None:
            return False
        queue = list((data or {}).get("queue") or []) or [song]
        try:
            index = int((data or {}).get("index") or 0)
        except (TypeError, ValueError):
            index = 0
        position = float((data or {}).get("position") or 0.0)
        self._abort = False
        player.start(prepared, queue=queue, index=index, offset_seconds=position)
        log.info(
            "[%s] 音乐续播（%s）：%s 从 %.1fs",
            self.stats.session_id[:8],
            reason,
            song.get("title"),
            position,
        )
        return True

    async def _handle_music_command(self, text: str) -> bool:
        """语音切歌 / 改播放模式：命中就自己执行完，返回 True 让这轮不进模型。"""
        player = self.music
        cmd = match_music_command(text)
        if player is None or cmd is None:
            return False
        recent = self._music_recent or {}
        fresh = bool(recent.get("song")) and (
            time.time() - float(recent.get("at") or 0.0) < MUSIC_CMD_WINDOW
        )
        if not (fresh or getattr(player, "playing", False) or self._music_resume):
            # 没在放歌、也刚没放过：这句话多半是闲聊，别抢它
            return False
        action, value = cmd
        sid = self.stats.session_id[:8]
        if action == "stop":
            self._music_recent = None
            self._clear_music_resume("语音停止")
            self._stop_music("用户停止")
            log.info("[%s] 音乐指令：停止", sid)
            await self._speak_text("好，不放了。")
            return True
        if action in ("next", "prev"):
            self._fill_queue_from_recent(player)
            prepared = (
                await player.next_track() if action == "next" else await player.prev_track()
            )
            if prepared is None:
                log.info("[%s] 音乐指令：%s 已到头", sid, action)
                await self._speak_text(
                    "这已经是最后一首啦。" if action == "next" else "已经是最前面一首啦。"
                )
                return True
            self._clear_music_resume("语音切歌")
            self._abort = False
            player.start(prepared, queue=list(player.queue), index=player.index)
            log.info(
                "[%s] 音乐指令：%s → %s",
                sid,
                action,
                (prepared.get("song") or {}).get("title"),
            )
            return True
        if action == "mode":
            if not player.set_mode(str(value)):
                return False
            mode_text = _MUSIC_MODE_TEXT.get(str(value), "")
            # 先说一句再放：两路 opus 同时推会互相盖掉
            await self._speak_text(f"好，{mode_text}。")
            ok = await self._restart_music(player, "改模式")
            log.info("[%s] 音乐指令：模式 %s（续播=%s）", sid, mode_text, ok)
            return True
        if action == "resume":
            ok = await self._restart_music(player, "继续播放")
            log.info("[%s] 音乐指令：继续播放（%s）", sid, ok)
            if not ok:
                await self._speak_text("现在没有可以接着放的歌。")
            return True
        return False

    async def speak_proactive(self, text: str, tag: str = "remind") -> bool: 
        """服务端主动说一句话（定时提醒 / 闹钟），返回是否真的播出去了。

        跟 _speak_text 的差别：
        - 正在对话或正在播报时先打断：提醒不该排在别人后面等
        - 播完不算一轮对话，不写历史、不动 last_stt
        - 播完刷新 _last_audio_at，避免刚播完立刻被判空闲
        """
        if self._closed:
            return False
        text = str(text or "").strip()
        if not text:
            return False
        if self._speaking or self._pipeline_lock.locked():
            log.info("[%s] 主动播报打断当前对话（%s）", self.stats.session_id[:8], tag)
            try:
                await self.abort_speaking(f"主动播报:{tag}")
            except Exception as exc:  # noqa: BLE001
                log.debug("打断当前对话失败（忽略）: %s", exc)
        # 打断会把 _abort 置 True，send_audio 见它就直接丢帧；主动播报开始前必须复位，
        # 否则这次提醒只会「响一下」或者干脆没声。
        self._abort = False
        self._abort_reason = ""
        log.info("[%s] 主动播报（%s）：%s", self.stats.session_id[:8], tag, text)
        try:
            await self._speak_text(text)
        except Exception as exc:  # noqa: BLE001
            log.warning("[%s] 主动播报失败: %s", self.stats.session_id[:8], exc)
            return False
        self._last_audio_at = time.time()
        self._bye_sent = False
        # 提醒打断了正在放的歌：播完接着放
        self._schedule_music_resume()
        return True

    async def _say_and_hangup(self, text: str, reason: str) -> None:
        """说一句收尾话，等音频发完再断开连接。"""
        log.info("[%s] 收尾：%s", self.stats.session_id[:8], reason)
        try:
            await self._speak_text(text)
        except Exception as exc:  # noqa: BLE001
            log.warning("[%s] 收尾话术播放失败: %s", self.stats.session_id[:8], exc)
        # 用独立任务关连接：直接 await close() 会取消到当前任务自己
        asyncio.create_task(self._delayed_close(1.5, reason))

    async def _delayed_close(self, delay: float, reason: str) -> None:
        await asyncio.sleep(delay)
        if self._closed:
            return
        log.info("[%s] 结束会话：%s", self.stats.session_id[:8], reason)
        await self.close()

    async def _handle_utterance(self, pcm: np.ndarray) -> None:
        """一句完整语音 -> ASR -> LLM -> TTS -> 下发。"""
        if pcm.size < 16000 * 0.25:
            return
        # 有人说话了：等待中的续播先撤掉，等这轮答完再重新排
        self._cancel_music_resume()
        # 上一轮还没答完（模型还在想，或者正在播报）时又听到一句新的：
        # 如果只是排队等锁，等第一轮播完再跑第二轮，孩子会连着听到两份回答。
        # 这里先把旧的那轮掐掉，只答最新这句。
        # 注意必须在把当前任务登记进 _utterance_tasks 之前做，否则 abort 会连自己一起取消。
        if self._pipeline_lock.locked() and not self._abort:
            log.info("[%s] 上一轮还没答完又来了新的一句，先打断旧的", self.stats.session_id[:8])
            await self.abort_speaking("听到新的问题")
        me = asyncio.current_task()
        self._utterance_tasks.add(me)
        try:
            await self._run_utterance(pcm)
        except asyncio.CancelledError:
            # 被 abort 取消：安静收尾，不要往上抛，否则 aiohttp 会打一堆 traceback
            log.info("[%s] 本轮已被打断，管线中止", self.stats.session_id[:8])
        finally:
            self._utterance_tasks.discard(me)
            if not self._utterance_tasks:
                self._speaking = False
                if self.stats.state == "speaking":
                    self.stats.state = "idle"

    async def abort_speaking(self, reason: str = "") -> None:
        """设备打断（说唤醒词或按键）：立刻停下发、取消在跑的 LLM/TTS、清空 VAD 残留。

        以前只置 _abort 标记，要等 LLM 吐出下一个 chunk 才会发现；
        模型卡住时用户说完话要等十几秒才被识别。这里直接 cancel 任务。
        """
        self._abort = True
        self._abort_reason = reason
        self._drop_pending()
        self._stop_music("打断")
        self._stop_story("打断")
        pending = [t for t in list(self._utterance_tasks) if not t.done()]
        for t in pending:
            t.cancel()
        if pending:
            # 等它们真正退出，避免和接下来的 listen start 抢 _pipeline_lock
            await asyncio.wait(pending, timeout=1.0)
        # 清掉 VAD 里还没成句的残音，避免把打断前的半句话当成新问题
        try:
            await asyncio.get_running_loop().run_in_executor(None, self.vad.reset)
        except Exception:
            pass
        # 无条件补一条 tts stop。
        # 设备端 Application::AbortSpeaking() 只把 aborted_ 置 true 并上报 abort，
        # 真正离开 speaking 状态要等收到 {"type":"tts","state":"stop"}。而正常收尾
        # 那条 tts stop 在 _run_utterance 末尾，任务被 cancel 时根本走不到——
        # 于是设备就永远停在「说话中」：麦克风关着、唤醒词进不了 listening，
        # 再按按键还是发 abort，形成死循环（用户反馈的「按按钮就卡死」）。
        # 重复发对设备无副作用：它只在 speaking 状态才响应。
        try:
            await self.send_json({"type": "tts", "state": "stop"})
        except Exception as exc:  # noqa: BLE001
            log.debug("[%s] 补发 tts stop 失败：%s", self.stats.session_id[:8], exc)
        self._speaking = False
        self.stats.state = "idle"
        log.info(
            "[%s] 打断完成 reason=%s，管线已取消，VAD 已重置",
            self.stats.session_id[:8],
            reason or "-",
        )

    async def _run_utterance(self, pcm: np.ndarray) -> None:
        """真正的一轮管线，被 _handle_utterance 包一层以便取消。"""
        async with self._pipeline_lock:
            # 每日输出字数到顶：不再陪聊，说一句劝退话并断开
            if self._limit_reached():
                log.info(
                    "[%s] 今日输出已达上限 %d 字，结束会话",
                    self.stats.session_id[:8],
                    self.max_chars_per_day,
                )
                await self._say_and_hangup(LIMIT_BYE_TEXT, "每日输出限额")
                return
            self.stats.utterances += 1
            secs = pcm.size / 16000
            log.info("[%s] 收到语音 %.2fs，开始识别", self.stats.session_id[:8], secs)

            t_asr = time.time()
            text = await asyncio.get_running_loop().run_in_executor(None, self.asr.transcribe, pcm)
            text = (text or "").strip()
            asr_ms = int((time.time() - t_asr) * 1000)
            if not text:
                log.info("[%s] 识别为空，忽略", self.stats.session_id[:8])
                return
            if is_filler(text):
                # 纯语气词（「嗯」「啊啊」）：孩子随口一声、环境噪音、VAD 切到半个字
                # 都会产生这种结果，不值得为它跑一轮两秒多的 LLM 往返。
                log.info("[%s] 识别为语气词，忽略: %s", self.stats.session_id[:8], text)
                return
            self.stats.last_stt = text
            log.info("[%s] ASR: %s", self.stats.session_id[:8], text)
            await self.send_json({"type": "stt", "text": text})
            # 「别放了」这类话：LLM 一般会调 stop_music，但万一它只是回一句话，
            # 8 秒后音乐又自己放回来会很怪。按关键词先兜一层。
            if any(word in text for word in MUSIC_STOP_WORDS):
                self._clear_music_resume("听到停止指令")
            # 孩子插话问问题：故事先停下，回答问题比讲完更重要。
            # 只在这里停（识别出真话之后），不跟 VAD 段挂钩——否则故事自己的
            # 声音被麦克风收到一点就把故事掐了。
            self._stop_story("听到新的问题")

            self._abort = False
            # 语音切歌/改模式：短句命中就直接执行并回一句，不进模型（省一轮两秒多）
            if await self._handle_music_command(text):
                return
            t_llm = time.time()
            # 趁 LLM 还在思考，先把 TTS 的 WSS 握手做掉（每条连接只能跑一个任务，
            # 所以按预计句数预握手几条；每句省约 340ms 握手开销）。
            prewarm = getattr(self.tts, "prewarm", None)
            if callable(prewarm):
                try:
                    prewarm(2)
                except Exception as exc:  # noqa: BLE001
                    log.debug("TTS 预连接失败（忽略）: %s", exc)
            llm = self.services.get("llm") or self.llm
            # 注入工具运行上下文，get_device_status 等工具需要 hub/session
            if hasattr(llm, "set_tool_context"):
                llm.set_tool_context(
                    {
                        "hub": self.services.get("hub"),
                        "session": self,
                        "schedule": self.services.get("schedule"),
                    }
                )

            # 流式分句：LLM 一边生成，TTS 一边合成，合成完一句就立刻下发，
            # 首句音频的延迟从「LLM 全部生成 + TTS 全部合成」降到「首句生成 + 首句合成」。
            self._speaking = True
            self.stats.state = "speaking"
            tts_started = False
            first_audio_at: Optional[float] = None
            total_packets = 0
            total_samples = 0
            full_text: List[str] = []
            tts_ms_total = 0
            buf = ""

            # TTS 并发合成、按序播放。
            # 每句 TTS 都有 600~800ms 固定握手+合成开销，旧代码一句一等（三句 2.3s）；
            # 现在句子一凑够就丢进合成池（最多 3 句并行），播放端按原顺序 await，
            # 第 2 句的握手与第 1 句的播放重叠，三句实测降到 0.85s。
            tts_pool = asyncio.Semaphore(3)
            tts_tasks: List[asyncio.Task] = []
            sent_texts: List[str] = []
            # 每句一个队列：合成端边收边往里推，播放端按句子顺序消费，
            # 既保留「多句并行合成」，又让每句的第一块音频一到就能开播。
            chunk_queues: List[asyncio.Queue] = []
            new_sentence = asyncio.Event()
            producer_done = False

            async def _synth(sent: str, queue: asyncio.Queue) -> None:
                """把一句的音频块流式推进它自己的队列，结束时放 None 收尾。"""
                nonlocal tts_ms_total
                async with tts_pool:
                    t0 = time.time()
                    try:
                        async for chunk in self.tts.stream(sent):
                            if self._abort or self._closed:
                                break
                            if getattr(chunk, "size", 0):
                                queue.put_nowait(chunk)
                    except Exception as exc:  # noqa: BLE001
                        log.warning(
                            "[%s] TTS 合成失败: %s", self.stats.session_id[:8], exc
                        )
                    finally:
                        tts_ms_total += int((time.time() - t0) * 1000)
                        queue.put_nowait(None)

            def enqueue(sentence: str) -> None:
                sent = strip_for_speech(sentence).strip()
                if not sent:
                    return
                q: asyncio.Queue = asyncio.Queue()
                chunk_queues.append(q)
                tts_tasks.append(asyncio.create_task(_synth(sent, q)))
                sent_texts.append(sent)
                new_sentence.set()
                # 池里的连接被取走后按差额补足，下一句仍能免握手
                if callable(prewarm):
                    try:
                        prewarm(3)
                    except Exception:  # noqa: BLE001
                        pass

            async def play_all() -> None:
                nonlocal tts_started, first_audio_at, total_packets, total_samples, tts_ms_total
                idx = 0
                interval = self.down_frame_ms / 1000.0
                try:
                    while True:
                        if self._abort or self._closed:
                            break
                        if idx < len(chunk_queues):
                            queue = chunk_queues[idx]
                            sent = sent_texts[idx]
                            idx += 1
                            sent_started = False
                            got_chunk = False
                            while True:
                                if self._abort or self._closed:
                                    break
                                # 带超时轮询：被打断时 producer 可能还卡在信号量上，
                                # 不能指望它一定把 None 送进来。
                                try:
                                    chunk = await asyncio.wait_for(queue.get(), timeout=0.25)
                                except asyncio.TimeoutError:
                                    continue
                                if chunk is None:
                                    break
                                got_chunk = True
                                if not sent_started:
                                    sent_started = True
                                    if not tts_started:
                                        tts_started = True
                                        await self.send_json({"type": "tts", "state": "start"})
                                    await self.send_json(
                                        {"type": "tts", "state": "sentence_start", "text": sent}
                                    )
                                next_at = time.monotonic()
                                for pkt in self.encoder.encode(chunk):
                                    if self._abort or self._closed:
                                        break
                                    await self.send_audio(pkt)
                                    total_packets += 1
                                    if first_audio_at is None:
                                        first_audio_at = time.time()
                                    next_at += interval
                                    delay = next_at - time.monotonic()
                                    if delay > 0:
                                        await asyncio.sleep(delay)
                                    else:
                                        next_at = time.monotonic()
                                total_samples += int(getattr(chunk, "size", 0))
                            if not got_chunk:
                                log.warning(
                                    "[%s] 单句 TTS 无输出: %s",
                                    self.stats.session_id[:8],
                                    sent[:20],
                                )
                                continue
                            # 句子收尾：把不足一帧的残余补零发出去
                            if not (self._abort or self._closed):
                                for pkt in self.encoder.flush():
                                    await self.send_audio(pkt)
                                    total_packets += 1
                                    if first_audio_at is None:
                                        first_audio_at = time.time()
                            continue
                        if producer_done:
                            break
                        new_sentence.clear()
                        if idx < len(chunk_queues):
                            continue
                        await new_sentence.wait()
                finally:
                    for t in tts_tasks[idx:]:
                        if not t.done():
                            t.cancel()

            player = asyncio.create_task(play_all())

            # 每轮对话前按「这句话」检索相关记忆再注入，比无脑塞前几条更贴题
            if self.memory is not None:
                try:
                    llm.set_memory_block(self.memory.build_context(text))
                except Exception as exc:
                    log.warning("记忆检索失败: %s", exc)

            async for piece in llm.chat_stream(self.history, text):
                if self._abort or self._closed:
                    break
                full_text.append(piece)
                buf += piece
                # 切句策略：句末标点立刻切；逗号类攒够 9 字也切，让首句更快送到 TTS
                while True:
                    cut = -1
                    for i, ch in enumerate(buf):
                        if ch in "\u3002\uff01\uff1f\uff1b\n!?;":
                            cut = i
                            break
                        if ch in "\uff0c\u3001\uff1a," and i >= 9:
                            cut = i
                            break
                    if cut < 0:
                        break
                    sentence, buf = buf[: cut + 1], buf[cut + 1 :]
                    if sentence.strip():
                        enqueue(sentence)
            # LLM 生成到此结束。计时口径必须在这里断开：这一刻之前是模型吐字，
            # 之后全是 TTS 合成与音频下发，混在一起会把播放时长算进模型耗时。
            t_llm_done = time.time()
            if buf.strip() and not self._abort:
                enqueue(buf)
            producer_done = True
            new_sentence.set()

            # 文本回显先于音频播完发出：旧代码要等全部音频下发完才发 llm 消息，
            # 前端字幕因此滞后整段播放时长。
            reply = strip_for_speech("".join(full_text)).strip() or "我没想好怎么回答。"
            self.stats.last_llm = reply
            log.info("[%s] LLM: %s", self.stats.session_id[:8], reply)
            await self.send_json({"type": "llm", "emotion": "neutral", "text": reply})

            await self._music_reply_fallback(reply, llm)

            await player
            t_play_done = time.time()

            self.history.append({"role": "user", "content": text})
            self.history.append({"role": "assistant", "content": reply})
            # 交给后台记忆线程：异步调模型抽取画像与事实，不占这一轮对话的延迟
            if self.memory_agent is not None:
                try:
                    self.memory_agent.submit_turn(text, reply, self.stats.session_id)
                except Exception as exc:
                    log.warning("记忆入队失败: %s", exc)
            elif self.memory is not None:
                try:
                    self.memory.extract_from_turn(text, reply)
                except Exception as exc:
                    log.warning("记忆抽取失败: %s", exc)
            # 三段耗时各自独立，不再混算：
            #   llm_ms         模型生成：t_llm → 最后一个字吐完
            #   first_audio_ms 首包：t_llm → 第一块音频下发（用户实际感知的等待）
            #   play_ms        音频下发：第一块 → 最后一块（约等于音频本身时长）
            #   total_ms       整轮：t_llm → 播放结束（不含 ASR）
            llm_ms = int((t_llm_done - t_llm) * 1000)
            first_audio_ms = int((first_audio_at - t_llm) * 1000) if first_audio_at else 0
            play_ms = int((t_play_done - first_audio_at) * 1000) if first_audio_at else 0
            total_ms = int((t_play_done - t_llm) * 1000)

            if not tts_started:
                log.warning("[%s] TTS 全程无输出，仅下发文本", self.stats.session_id[:8])
                await self.send_json({"type": "tts", "state": "start"})
                await self.send_json({"type": "tts", "state": "stop"})

            await self.send_json({"type": "tts", "state": "stop"})
            self._speaking = False
            self.stats.state = "idle"
            self._start_pending_music()
            # 讲故事工具已经把正文挂在 pending_story 上，这句话说完就开讲
            self._start_pending_story()
            # 刚才被唤醒词/按键打断的歌，安静 8 秒后从断点接上
            self._schedule_music_resume()
            # 这一轮已经答完，「刚停过歌」的记录不再有参考价值
            self._music_recent = None
            log.info(
                "[%s] 耗时拆分：ASR %dms · 模型 %dms · 首包 %dms · 播放 %dms · 整轮 %dms"
                "（共 %d 包 %.2fs 音频）",
                self.stats.session_id[:8],
                asr_ms,
                llm_ms,
                first_audio_ms,
                play_ms,
                total_ms,
                total_packets,
                total_samples / self.down_rate,
            )
            # 累计今天说了多少字，用于每日上限判断
            if self.counter is not None and reply:
                try:
                    total = self.counter.add(self.stats.device_id, len(reply))
                    if self.max_chars_per_day > 0 and total >= self.max_chars_per_day:
                        log.info(
                            "[%s] 今日累计输出 %d 字，已达上限 %d",
                            self.stats.session_id[:8],
                            total,
                            self.max_chars_per_day,
                        )
                except Exception as exc:  # noqa: BLE001
                    log.warning("[%s] 累计输出字数失败: %s", self.stats.session_id[:8], exc)
            self._record_chat(
                text,
                reply,
                asr_ms,
                llm_ms,
                tts_ms_total,
                pcm,
                first_audio_ms=first_audio_ms,
                play_ms=play_ms,
                total_ms=total_ms,
            )

    def _record_chat(
        self,
        user_text: str,
        reply_text: str,
        asr_ms: int,
        llm_ms: int,
        tts_ms: int,
        pcm: Optional[np.ndarray] = None,
        first_audio_ms: int = 0,
        play_ms: int = 0,
        total_ms: int = 0,
    ) -> None:
        """把这一轮对话写入历史文件（含用户原始录音）；失败不影响主流程。"""
        if self.store is None:
            return
        # 上行是 16kHz int16 单声道，原样存成 wav，历史页就能回放用户本来的声音
        wav: Optional[bytes] = None
        audio_secs = 0.0
        if pcm is not None and getattr(pcm, "size", 0):
            try:
                from .tts import pcm_to_wav_bytes

                wav = pcm_to_wav_bytes(pcm, UPLINK_SAMPLE_RATE)
                audio_secs = float(pcm.size) / float(UPLINK_SAMPLE_RATE)
            except Exception as exc:  # noqa: BLE001
                log.warning("[%s] 用户录音编码失败: %s", self.stats.session_id[:8], exc)
                wav = None
        try:
            self.store.append(
                session_id=self.stats.session_id,
                device_id=self.stats.device_id,
                user_text=user_text,
                reply_text=reply_text,
                asr_ms=asr_ms,
                llm_ms=llm_ms,
                tts_ms=tts_ms,
                first_audio_ms=first_audio_ms,
                play_ms=play_ms,
                total_ms=total_ms,
                audio=wav,
                audio_secs=audio_secs,
            )
        except Exception as exc:
            log.warning("[%s] 写对话历史失败: %s", self.stats.session_id[:8], exc)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._cancel_watchdog()
        self._cancel_music_resume()
        for t in list(self._tasks):
            t.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        # 主动关 ws，让 handle_ws 里的 async for 退出（空闲告别 / 限额劝退要用）
        try:
            await self.ws.close()
        except Exception:  # noqa: BLE001
            pass

    def info(self) -> Dict[str, Any]:
        return {
            "session_id": self.stats.session_id,
            "device_id": self.stats.device_id,
            "client_id": self.stats.client_id,
            "connected_seconds": round(time.time() - self.stats.connected_at, 1),
            "uplink_packets": self.stats.uplink_packets,
            "downlink_packets": self.stats.downlink_packets,
            "utterances": self.stats.utterances,
            "last_stt": self.stats.last_stt,
            "last_llm": self.stats.last_llm,
            "state": self.stats.state,
        }
