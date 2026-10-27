"""猫剪（火山豆包）直连 TTS 通道。

协议来源：第三方阅读器插件「猫剪豆问」逆向结果（_ref_v3.js），要点：
- WS 地址 wss://audio5-normal-hl.myparallelstory.com/internal/api/v1/ws?ssmix=&aid=&device_id=
  其中 aid / device_id **必须是 13 位纯数字**，否则服务端回
  {"event":"TaskFailed","status_code":40200001,"statusText":"InvalidAccess:EmptyToken"}
  （这个错误文案具有误导性，与 token 无关，改 ID 格式即可）。
- StartTask -> TaskStarted -> 发 text 帧 -> FinishTask -> 收音频 -> TaskFinished。
- audio_config.format=pcm 时返回 24kHz s16le 单声道裸 PCM；mp3 时返回 MP3。
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import random
import socket
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

log = logging.getLogger("tts.maoxiang")

MAOXIANG_WS = "wss://audio5-normal-hl.myparallelstory.com/internal/api/v1/ws"
MAOXIANG_APPKEY = "WQuVLKMGRo"
MAOXIANG_NAMESPACE = "BidirectionalTTS"
MAOXIANG_SAMPLE_RATE = 24000

# 已知不支持情绪参数的音色（首次失败后记录，避免每次都多花一次往返）
_EMOTION_UNSUPPORTED: set = set()

# 预连接空闲多久后视为过期（服务端实测空闲 20s 仍可用，留足余量）
IDLE_TTL = 15.0


def _ws_host() -> str:
    """猫剪 WS 的主机名，握手时既要做 SNI 也要按它解析 IPv4。"""
    return "audio5-normal-hl.myparallelstory.com"


_ORIG_GETADDRINFO = socket.getaddrinfo


def _ipv4_only_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    """强制只解析 IPv4。

    本机 DNS 被 Clash fake-ip 接管，猫剪域名会同时解析出 IPv6 2001:2::94
    （黑洞，TCP 直接超时 5 秒）和 IPv4 198.18.0.152（TUN 可达）。websockets
    默认按 getaddrinfo 顺序先试 IPv6，一挂就是整个 open_timeout，表现为
    "timed out during opening handshake"。把 IPv6 从解析结果里剔掉即可。
    """
    if host == _ws_host() and family in (0, socket.AF_UNSPEC):
        try:
            return _ORIG_GETADDRINFO(host, port, socket.AF_INET, type, proto, flags)
        except socket.gaierror:
            pass
    return _ORIG_GETADDRINFO(host, port, family, type, proto, flags)


# 只影响猫剪域名，其他域名的解析行为完全不变
socket.getaddrinfo = _ipv4_only_getaddrinfo


def _ws_open(ws) -> bool:
    """websockets 15.x 没有 .closed 属性，用 state 判断（1 = OPEN）。"""
    state = getattr(ws, "state", None)
    if state is None:
        return True
    try:
        return int(state) == 1
    except (TypeError, ValueError):
        return str(state).lower().endswith("open")


async def _safe_close(ws) -> None:
    try:
        await ws.close()
    except Exception:  # noqa: BLE001
        pass

# 音色表远程地址（插件里用的同一个 CDN 仓库）；本地有缓存时优先用缓存。
MAOXIANG_VOICES_URL = "https://cnb.cool/xiatian.ktn/tts/-/git/raw/main/maojiandouwentts.json"

# 兜底音色（网络不可用时至少能选）
FALLBACK_VOICES = [
    {"id": "zh_female_vv_uranus_bigtts", "name": "薇薇", "gender": "女", "style": "通用女声，清亮"},
    {"id": "ICL_5561786db01b", "name": "豆问默认", "gender": "女", "style": "插件默认音色"},
    {"id": "zh_male_yunzhou_jupiter_bigtts", "name": "云舟", "gender": "男", "style": "通用男声"},
]

# 插件内置的情绪枚举（音频侧原生支持时走 audio_config.emotion，否则降级为提示词）
EMOTIONS = [
    {"id": "", "name": "默认"},
    {"id": "happy", "name": "开心"},
    {"id": "excited", "name": "兴奋"},
    {"id": "tender", "name": "温柔"},
    {"id": "comfort", "name": "安慰"},
    {"id": "sad", "name": "悲伤"},
    {"id": "angry", "name": "生气"},
    {"id": "surprised", "name": "惊讶"},
    {"id": "shy", "name": "害羞"},
    {"id": "news", "name": "播报"},
    {"id": "story", "name": "讲故事"},
    {"id": "neutral", "name": "中性"},
]


def _rand_id() -> str:
    """13 位纯数字，模拟插件 generateId()。"""
    return str(random.randint(10**12, 10**13 - 1))


def _load_voice_file(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out: List[Dict[str, Any]] = []
    for item in data:
        vid = str(item.get("voice_id") or "").strip()
        if not vid:
            continue
        tags = []
        if item.get("is_pro"):
            tags.append("高级")
        if item.get("is_emotion"):
            tags.append("情绪")
        if item.get("is_singing"):
            tags.append("唱歌")
        out.append(
            {
                "id": vid,
                "name": str(item.get("name") or vid),
                "gender": str(item.get("gender") or ""),
                "style": "/".join(tags) or str(item.get("category") or ""),
                "emotions": item.get("emotions") or [],
            }
        )
    return out


def load_maoxiang_voices(cache_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """读取本地缓存的音色表；没有就返回兜底列表。"""
    if cache_path and cache_path.exists():
        try:
            voices = _load_voice_file(cache_path)
            if voices:
                return voices
        except Exception as exc:  # noqa: BLE001
            log.warning("读取猫剪音色表失败: %s", exc)
    return list(FALLBACK_VOICES)


async def fetch_maoxiang_voices(cache_path: Path) -> List[Dict[str, Any]]:
    """下载音色表并落盘缓存（启动时调用一次）。"""
    import aiohttp

    try:
        timeout = aiohttp.ClientTimeout(total=20)
        async with aiohttp.ClientSession(timeout=timeout) as sess:
            async with sess.get(MAOXIANG_VOICES_URL) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"HTTP {resp.status}")
                raw = await resp.read()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(raw)
        voices = _load_voice_file(cache_path)
        log.info("猫剪音色表已更新：%d 个音色", len(voices))
        return voices
    except Exception as exc:  # noqa: BLE001
        log.warning("下载猫剪音色表失败（沿用本地缓存）：%s", exc)
        return load_maoxiang_voices(cache_path)


class MaoxiangTTS:
    """猫剪直连 TTS：返回 sample_rate 的 int16 单声道 PCM。"""

    def __init__(
        self,
        voice: str = "zh_female_vv_uranus_bigtts",
        sample_rate: int = MAOXIANG_SAMPLE_RATE,
        speech_rate: float = 1.0,
        pitch: float = 0.0,
        loudness: float = 0.0,
        emotion: str = "",
        emotion_scale: float = 0.0,
        timeout: float = 20.0,
        open_timeout: float = 5.0,
    ):
        self.voice = voice or "zh_female_vv_uranus_bigtts"
        self.sample_rate = int(sample_rate)
        self.speech_rate = max(0.2, min(3.0, float(speech_rate or 1.0)))
        self.pitch = max(-5.0, min(5.0, float(pitch or 0.0)))
        self.loudness = max(-50.0, min(100.0, float(loudness or 0.0)))
        self.emotion = (emotion or "").strip()
        self.emotion_scale = max(0.0, min(5.0, float(emotion_scale or 0.0)))
        self.timeout = timeout
        # 握手单独设短超时：猫剪握手正常 0.3~0.5s，偶发长时间不回包。
        # 用 20s 的合成超时去等握手会让一句 TTS 白等 20 秒，重试两次就是 40 秒。
        self.open_timeout = max(1.0, float(open_timeout))
        # 预握手连接池：[(websocket, 建立时刻)]
        self._idle: List[Any] = []
        self._warm_tasks: set = set()
        self._closed = False

    def _build_url(self) -> str:
        return f"{MAOXIANG_WS}?ssmix=&aid={_rand_id()}&device_id={_rand_id()}"

    def _build_payload(self) -> Dict[str, Any]:
        audio_config: Dict[str, Any] = {
            "format": "pcm",
            "sample_rate": MAOXIANG_SAMPLE_RATE,
            "loudness_rate": int(self.loudness),
        }
        if self.emotion:
            audio_config["emotion"] = self.emotion
            if self.emotion_scale:
                audio_config["emotion_scale"] = self.emotion_scale
        return {
            "audio_config": audio_config,
            "extra": {
                "post_process": {"pitch": self.pitch, "speech_rate": self.speech_rate},
                "max_length_to_filter_parenthesis": 0,
            },
            "speaker": self.voice,
        }

    # ---------- 预连接池 ----------
    # 服务端在 TaskFinished 之后会主动关掉连接（实测第二句复用返回 0 字节 / 1000 OK），
    # 所以一条连接只能跑一个任务；但连接建好后空闲 20s 仍可用。
    # 于是改成「提前握手、用完即弃」：LLM 还在出字时就把连接握好，
    # 句子到了直接发任务，每句省掉约 340ms 的 WSS 握手。

    def prewarm(self, count: int = 1) -> None:
        """提前握手，保证池里至少有 count 条可用连接（失败静默，走冷连接兜底）。"""
        if self._closed:
            return
        alive = self._purge_idle()
        for _ in range(max(0, int(count) - alive)):
            task = asyncio.create_task(self._warm_one())
            self._warm_tasks.add(task)
            task.add_done_callback(self._warm_tasks.discard)

    def _purge_idle(self) -> int:
        """清掉过期/已关的闲置连接，返回剩余可用条数。"""
        now = time.time()
        kept = []
        for ws, born in self._idle:
            if not _ws_open(ws):
                continue
            if now - born > IDLE_TTL:
                asyncio.create_task(_safe_close(ws))
                continue
            kept.append((ws, born))
        self._idle = kept
        return len(kept)

    async def _warm_one(self) -> None:
        import websockets

        try:
            ws = await self._connect_race(2)
        except Exception as exc:  # noqa: BLE001
            log.debug("猫剪预连接失败（忽略，改用冷连接）: %s", exc)
            return
        if self._closed:
            await _safe_close(ws)
            return
        self._idle.append((ws, time.time()))
        # 兜底：闲置超时的连接自己关掉，别挂在服务端
        asyncio.create_task(self._expire(ws))

    async def _expire(self, ws) -> None:
        try:
            await asyncio.sleep(IDLE_TTL)
        except asyncio.CancelledError:
            return
        for i, (w, _) in enumerate(self._idle):
            if w is ws:
                self._idle.pop(i)
                await _safe_close(ws)
                return

    def _take_idle(self):
        """取一条已握手的连接；过期或已关的顺手丢掉。"""
        now = time.time()
        while self._idle:
            ws, born = self._idle.pop()
            if not _ws_open(ws):
                continue
            if now - born > IDLE_TTL:
                asyncio.create_task(_safe_close(ws))
                continue
            return ws
        return None

    async def aclose(self) -> None:
        """关掉闲置连接与预热任务（会话结束时调用）。"""
        self._closed = True
        for task in list(self._warm_tasks):
            task.cancel()
        self._warm_tasks.clear()
        while self._idle:
            ws, _ = self._idle.pop()
            await _safe_close(ws)

    async def synthesize(self, text: str) -> np.ndarray:
        if not text or not text.strip():
            return np.zeros(0, dtype=np.int16)
        # 部分音色不支持情绪参数，带了情绪会静默返回 0 字节音频。
        # 这种情况自动退回无情绪重试一次，并把该音色标记为「无情绪」。
        pcm = await self._synthesize_once(text)
        if pcm.size == 0 and self.emotion:
            global _EMOTION_UNSUPPORTED
            if self.voice not in _EMOTION_UNSUPPORTED:
                _EMOTION_UNSUPPORTED.add(self.voice)
                log.warning("音色 %s 不支持情绪 %s，已自动退回无情绪", self.voice, self.emotion)
            old = self.emotion
            self.emotion = ""
            try:
                pcm = await self._synthesize_once(text)
            finally:
                self.emotion = old
        return pcm

    async def stream(self, text: str):
        """流式合成：收到一块音频就产出一块，首包从 1.1~1.3s 降到 0.3s 左右。

        豆包 bigtts 本身是边合成边推送的，旧实现把所有块攒进 buf 等 TaskFinished
        才返回，等于把「首包」拖成了「整句」。这里改成边收边产出，播放端拿到第一块
        就能开始编码下发，剩下的在播放期间陆续到达。
        """
        if not text or not text.strip():
            return
        # 需要重采样时退回整句：逐块重采样在块边界会有瑕疵，而当前配置
        # （downlink_sample_rate = 24000 = MAOXIANG_SAMPLE_RATE）不会走到这里。
        if self.sample_rate != MAOXIANG_SAMPLE_RATE:
            pcm = await self.synthesize(text)
            if pcm.size:
                yield pcm
            return

        got_any = False
        try:
            async for chunk in self._stream_once(text):
                got_any = True
                yield chunk
        except RuntimeError:
            # 部分音色不支持情绪参数，带了情绪会静默返回 0 字节，退回无情绪重试一次
            if got_any or not self.emotion:
                raise
            global _EMOTION_UNSUPPORTED
            if self.voice not in _EMOTION_UNSUPPORTED:
                _EMOTION_UNSUPPORTED.add(self.voice)
                log.warning("音色 %s 不支持情绪 %s，已自动退回无情绪", self.voice, self.emotion)
            old = self.emotion
            self.emotion = ""
            try:
                async for chunk in self._stream_once(text):
                    yield chunk
            finally:
                self.emotion = old

    async def _stream_once(self, text: str):
        """真正干活的流式合成：逐块 yield int16 PCM。"""
        started = time.time()
        tail = b""
        total = 0
        failed: Optional[str] = None
        first_at: Optional[float] = None
        ws = await self._open_ws()
        await ws.send(
            json.dumps(
                {
                    "appkey": MAOXIANG_APPKEY,
                    "event": "StartTask",
                    "namespace": MAOXIANG_NAMESPACE,
                    "payload": json.dumps(self._build_payload(), ensure_ascii=False),
                }
            )
        )
        text_sent = False
        try:
            async with asyncio.timeout(self.timeout):
                async for msg in ws:
                    raw = b""
                    if isinstance(msg, (bytes, bytearray)):
                        raw = bytes(msg)
                    else:
                        try:
                            data = json.loads(msg)
                        except Exception:  # noqa: BLE001
                            continue
                        if data.get("type") == 3 and data.get("buffer"):
                            try:
                                raw = base64.b64decode(data["buffer"])
                            except Exception:  # noqa: BLE001
                                raw = b""
                        else:
                            event = str(data.get("event") or "")
                            if event == "TaskFailed":
                                failed = str(
                                    data.get("status_text") or data.get("statusText") or "unknown"
                                )
                                break
                            if event == "TaskStarted" and not text_sent:
                                text_sent = True
                                await ws.send(
                                    json.dumps(
                                        {"payload": json.dumps({"text": text}, ensure_ascii=False)},
                                        ensure_ascii=False,
                                    )
                                )
                                await ws.send(
                                    json.dumps(
                                        {
                                            "appkey": MAOXIANG_APPKEY,
                                            "event": "FinishTask",
                                            "namespace": MAOXIANG_NAMESPACE,
                                        }
                                    )
                                )
                            elif event == "TaskFinished":
                                break
                            continue
                    if not raw:
                        continue
                    # int16 是 2 字节对齐，网络分片可能切在半个样本上，余字节留到下一块
                    raw = tail + raw
                    if len(raw) % 2:
                        tail = raw[-1:]
                        raw = raw[:-1]
                    else:
                        tail = b""
                    if not raw:
                        continue
                    pcm = np.frombuffer(raw, dtype=np.int16).copy()
                    total += pcm.size
                    if first_at is None:
                        first_at = time.time()
                        log.info(
                            "猫剪首包 %.0fms（音色 %s）",
                            (first_at - started) * 1000.0,
                            self.voice,
                        )
                    yield pcm
        except (TimeoutError, asyncio.TimeoutError):
            failed = "timeout"
        finally:
            await _safe_close(ws)

        if failed and total == 0:
            raise RuntimeError(f"猫剪 TTS 失败：{failed}")
        if total:
            log.info(
                "猫剪合成完成 %.2fs 音频（用时 %dms，音色 %s）",
                total / float(self.sample_rate),
                int((time.time() - started) * 1000),
                self.voice,
            )

    async def _connect_race(self, n: int = 3):
        """并发 n 路握手，取第一条成功的，其余取消。

        猫剪走火山 CDN（volc-dcdn），握手偶发完全不回包：实测同一时刻
        手写 Upgrade 请求 5/5 成功，而 websockets 串行重试会连续失败。
        并发几路后基本必中，最坏耗时也只是单次 open_timeout 而不是 n 倍。
        """
        import websockets

        url = self._build_url()
        ot = self.open_timeout

        async def _one():
            # websockets 15.x 的 connect() 返回的是可等待对象而非 coroutine，
            # create_task 不接受，所以包一层真正的协程。
            return await websockets.connect(url, max_size=None, open_timeout=ot)

        tasks = [asyncio.create_task(_one()) for _ in range(max(1, n))]
        winner = None
        errors: List[BaseException] = []
        pending = set(tasks)
        try:
            while pending and winner is None:
                done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                for t in done:
                    if t.cancelled():
                        continue
                    exc = t.exception()
                    if exc is not None:
                        errors.append(exc)
                        continue
                    ws = t.result()
                    if winner is None:
                        winner = ws
                    else:
                        await _safe_close(ws)
        finally:
            for t in pending:
                t.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
        if winner is None:
            raise (errors[-1] if errors else RuntimeError("猫剪握手全部失败"))
        return winner

    async def _open_ws(self):
        """优先取预握手连接，没有就并发握手现连。"""
        ws = self._take_idle()
        if ws is not None:
            return ws
        return await self._connect_race(3)

    async def _synthesize_once(self, text: str) -> np.ndarray:
        started = time.time()
        buf = bytearray()
        failed: Optional[str] = None
        ws = await self._open_ws()
        if True:
            await ws.send(
                json.dumps(
                    {
                        "appkey": MAOXIANG_APPKEY,
                        "event": "StartTask",
                        "namespace": MAOXIANG_NAMESPACE,
                        "payload": json.dumps(self._build_payload(), ensure_ascii=False),
                    }
                )
            )
            text_sent = False
            try:
                async with asyncio.timeout(self.timeout):
                    async for msg in ws:
                        if isinstance(msg, (bytes, bytearray)):
                            buf.extend(msg)
                            continue
                        try:
                            data = json.loads(msg)
                        except Exception:  # noqa: BLE001
                            continue
                        if data.get("type") == 3 and data.get("buffer"):
                            try:
                                buf.extend(base64.b64decode(data["buffer"]))
                            except Exception:  # noqa: BLE001
                                pass
                            continue
                        event = str(data.get("event") or "")
                        if event == "TaskFailed":
                            failed = str(data.get("status_text") or data.get("statusText") or "unknown")
                            break
                        if event == "TaskStarted" and not text_sent:
                            text_sent = True
                            await ws.send(
                                json.dumps(
                                    {"payload": json.dumps({"text": text}, ensure_ascii=False)},
                                    ensure_ascii=False,
                                )
                            )
                            await ws.send(
                                json.dumps(
                                    {
                                        "appkey": MAOXIANG_APPKEY,
                                        "event": "FinishTask",
                                        "namespace": MAOXIANG_NAMESPACE,
                                    }
                                )
                            )
                        elif event == "TaskFinished":
                            break
            except (TimeoutError, asyncio.TimeoutError):
                failed = "timeout"
            finally:
                await _safe_close(ws)

        if failed and not buf:
            raise RuntimeError(f"猫剪 TTS 失败：{failed}")
        if not buf:
            return np.zeros(0, dtype=np.int16)
        pcm = np.frombuffer(bytes(buf), dtype=np.int16).copy()
        if self.sample_rate != MAOXIANG_SAMPLE_RATE:
            from .audio import resample

            pcm = resample(pcm, MAOXIANG_SAMPLE_RATE, self.sample_rate)
        log.info(
            "猫剪合成完成 %.2fs 音频（%d 字节，用时 %dms，音色 %s）",
            pcm.size / float(self.sample_rate),
            len(buf),
            int((time.time() - started) * 1000),
            self.voice,
        )
        return pcm
