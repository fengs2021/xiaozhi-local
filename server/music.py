"""在线音乐：搜歌、拿直链、解码、按帧下发到设备。

设计取舍（为什么不用洛雪 / B 站直连）：
- 洛雪音乐（LX Music）是 Electron 桌面应用，音源是跑在渲染进程里的 JS 脚本，
  没有对外 HTTP 接口。要复用它得再起一个 Node 服务常驻，多一个进程和一个失效点。
  它的音源脚本本质上就是调各音乐平台的公开接口，这里直接做同一件事，省掉中间层。
- B 站 web 搜索接口现在一律 412（风控要 wbi 签名 + buvid cookie），yt-dlp 的
  bilisearch 同样过不去；就算过了，视频音轨里 MV、翻唱、直播切片混在一起，
  按歌名点歌命中率很差。签名这一套在 server/bili.py 里做了，但只给「讲故事」
  用（那里分P天然就是单集），点歌还是走网易云/酷我。
- 网易云搜索接口和 song/media/outer/url 直链不需要登录，实测大部分歌能直接拿到
  128k mp3；拿不到（版权/VIP）会返回 html，这里按 Content-Type 过滤，自动试下一首。

音频链路：mp3/aac 字节 -> PyAV 解码重采样成 24kHz int16 单声道 -> libopus 20ms 帧
-> 按帧节奏 send_audio。音质上限由设备的 opus 下行采样率决定（当前 24kHz 单声道
64kbps，大约是调频广播的水平），不是这里能改的。
"""
from __future__ import annotations

import ast
import asyncio
import io
import json
import logging
import math
import os
import random
import re
import time
import urllib.parse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .audio import OpusDownlinkEncoder

try:
    import av
except Exception as exc:  # pragma: no cover
    raise RuntimeError("需要安装 PyAV：pip install av") from exc

log = logging.getLogger("music")

NETEASE_SEARCH = "https://music.163.com/api/search/get/web"
NETEASE_OUTER = "https://music.163.com/song/media/outer/url?id={}.mp3"

# 网易云拿不到（VIP/版权/下架）时用酷我兜底。酷我属腾讯音乐娱乐集团（TME），
# 跟 QQ 音乐、酷狗同系；搜索和直链都不需要登录，实测儿歌命中率高。
# 两个坑：搜索返回的是 Python 字面量（true/false/null 不是 JSON 写法），
# 直链接口返回的是裸 URL 字符串（不是 JSON）。
KUWO_SEARCH = "https://search.kuwo.cn/r.s"
KUWO_URL = "https://antiserver.kuwo.cn/anti.s"
# 酷我对付费歌只给 11 秒左右试听；这种片段按体积看不出来，必须数 mp3 帧
MIN_PLAY_SECONDS = 45.0

# 溯音（oiapi.net）两个代理：不负责搜索，只在直连拿不到时兜底换链。
# 实测能绕开两家直连的限制：网易云对我们 404 的《孤勇者》它能给 256 秒完整曲；
# 酷我只给 11 秒试听的付费原唱它也能给整首。
# 但很不稳定（403 / 410 / 1 秒垃圾流都遇到过），所以必须照样做体积和时长校验，
# 并且放在音源链最末尾。
SUYIN_163 = "https://oiapi.net/api/Music_163"
SUYIN_KUWO = "https://oiapi.net/api/Kuwo"
_SUYIN_LINK_RE = re.compile(r"音乐链接[：:]\s*(\S+)")
_SUYIN_NAME_RE = re.compile(r"歌名[：:]\s*(\S+)")
_SUYIN_SINGER_RE = re.compile(r"歌手[：:]\s*(\S+)")

_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
_HEADERS = {"Referer": "https://music.163.com", "User-Agent": _UA}
KUWO_HEADERS = {"Referer": "https://www.kuwo.cn", "User-Agent": _UA}

# 播放模式：off 顺序（播完就停）/ one 单曲循环 / all 列表循环 / shuffle 随机
MUSIC_MODES = ("off", "one", "all", "shuffle")
MUSIC_MODE_DEFAULT = "all"
MUSIC_MODE_TEXT = {
    "off": "顺序播放",
    "one": "单曲循环",
    "all": "列表循环",
    "shuffle": "随机播放",
}

# 暂停超过这么久就按「停止」处理：暂停态设备仍停在 speaking，
# 屏幕上只要音乐界面还在就不省电（20% 亮度亮到天亮），而且孩子早就不听了。
# 停之前把进度记进 session，说「继续播放」还能接着听。
PAUSE_STOP_SECONDS = 120.0

# 单曲下载上限，防止拿到奇怪的大文件把内存吃穿
MAX_DOWNLOAD_BYTES = 16 * 1024 * 1024
# 小于这个体积基本是错误页或者试听片段
MIN_VALID_BYTES = 200 * 1024


def _norm(text: Any) -> str:
    """去掉空白和常见分隔符，方便比标题。"""
    s = str(text or "").lower()
    for ch in " \t\u3000·-—_/\\|()（）[]【】<>《》":
        s = s.replace(ch, "")
    return s


def _rank(keyword: str, song: Dict[str, Any]) -> float:
    """给一首搜索结果打分：标题越像关键词、歌手对得上、播放量越大，分越高。"""
    kw = _norm(keyword)
    title = _norm(song.get("title"))
    artist = _norm(song.get("artist"))
    score = 0.0
    if kw and title:
        if title == kw:
            score += 1000
        elif title.startswith(kw):
            score += 600
        elif kw in title:
            score += 400
        elif title in kw:
            score += 300
    if kw and artist:
        if artist in kw:
            score += 300
        else:
            for part in str(song.get("artist") or "").replace("、", " ").replace(",", " ").split():
                if part and _norm(part) in kw:
                    score += 300
                    break
    pc = int(song.get("play_count") or 0)
    if pc > 0:
        score += min(200.0, math.log10(pc + 1) * 25)
    dur = int(song.get("duration") or 0)
    if dur and dur < 45:
        # 铃声/试听片段，不是完整歌
        score -= 400
    # 伴奏、纯音乐、铃声、DJ 版这些孩子点歌时基本不是想要的
    low_title = str(song.get("title") or "").lower()
    if kw:
        for bad in ("纯音乐", "钢琴版", "伴奏", "弹鼓", "铃声", "试听", "片段", "dj"):
            if bad in low_title and bad not in kw:
                score -= 250
                break
        # 「第01集 遵守交规…」这种是儿童广播剧分集，不是歌
        if "集" in low_title and "第" in low_title and "第" not in kw:
            score -= 250
    return score


_MP3_BITRATES = [0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320]
_MP3_RATES = [44100, 48000, 32000]
_KUWO_TRUE_RE = (re.compile(r"\btrue\b"), re.compile(r"\bfalse\b"), re.compile(r"\bnull\b"))


def _parse_sources(value: Any) -> List[str]:
    """把配置里的音源链解析成 provider 列表，只留认识的、去重保序。"""
    if value is None:
        return ["netease", "kuwo"]
    if isinstance(value, (list, tuple)):
        items = [str(x or "") for x in value]
    else:
        items = str(value).replace(";", ",").split(",")
    out: List[str] = []
    for item in items:
        name = str(item).strip().lower()
        if name in ("netease", "kuwo", "suyin") and name not in out:
            out.append(name)
    return out or ["netease", "kuwo"]


def _mp3_seconds(data: bytes) -> float:
    """数 mp3 帧算时长；不是 mp3（或者解析不出来）返回 0。"""
    n = len(data)
    i = 0
    if data[:3] == b"ID3" and n > 10:
        i = 10 + (
            ((data[6] & 0x7F) << 21)
            | ((data[7] & 0x7F) << 14)
            | ((data[8] & 0x7F) << 7)
            | (data[9] & 0x7F)
        )
    secs = 0.0
    frames = 0
    while i + 4 <= n:
        if data[i] != 0xFF or (data[i + 1] & 0xE0) != 0xE0:
            i += 1
            continue
        ver = (data[i + 1] >> 3) & 3
        layer = (data[i + 1] >> 1) & 3
        bitrate = (data[i + 2] >> 4) & 0xF
        rate = (data[i + 2] >> 2) & 3
        padding = (data[i + 2] >> 1) & 1
        if ver != 3 or layer != 1 or bitrate in (0, 15) or rate == 3:
            i += 1
            continue
        size = int(144 * _MP3_BITRATES[bitrate] * 1000 / _MP3_RATES[rate]) + padding
        if size <= 4:
            i += 1
            continue
        secs += 1152.0 / _MP3_RATES[rate]
        frames += 1
        i += size
    return secs if frames else 0.0


_BAD_VERSION_WORDS = (
    "片段", "合集", "串烧", "伴奏", "纯音乐", "铃声", "试听", "钢琴版", "dj版", "remix",
)
_BRACKET_RE = re.compile(r"[（(\[【][^）)\]】]*[）)\]】]")


def _clean_title(text: Any) -> str:
    """把「稻香(Live)」「孤勇者（热血epic版）」里的括号内容去掉再比，
    这样「同一首歌的不同版本」算同一首，而「稻香老友」「小跳蛙+拔萝卜串烧」不算。"""
    return _norm(_BRACKET_RE.sub("", str(text or "")))


def _clean_kuwo(text: Any) -> str:
    """酷我的标题里混着 &nbsp; 之类的实体，清一下省得念出来奇怪。"""
    s = str(text or "")
    for a, b in (("&nbsp;", " "), ("&amp;", "&"), ("&quot;", '"'), ("&#39;", "'")):
        s = s.replace(a, b)
    # 酷我在字符串里塞了 JSON 风格转义（\u0026 就是 &），ast 解析后还带着反斜杠
    if "\\u" in s:
        s = re.sub(
            r"\\u([0-9a-fA-F]{4})", lambda m: chr(int(m.group(1), 16)), s
        )
    return re.sub(r"\s+", " ", s).strip()


def decode_bytes(data: bytes, sample_rate: int) -> np.ndarray:
    """把音频文件字节解成指定采样率的 int16 单声道 PCM。"""
    chunks: List[np.ndarray] = []
    try:
        with av.open(io.BytesIO(data)) as container:
            if not container.streams.audio:
                return np.zeros(0, dtype=np.int16)
            stream = container.streams.audio[0]
            resampler = av.AudioResampler(format="s16", layout="mono", rate=sample_rate)
            for frame in container.decode(stream):
                for r in resampler.resample(frame):
                    arr = r.to_ndarray()
                    if arr.dtype != np.int16:
                        arr = (np.clip(arr, -1.0, 1.0) * 32767).astype(np.int16)
                    chunks.append(arr.reshape(-1))
            for r in resampler.resample(None):
                arr = r.to_ndarray()
                if arr.dtype != np.int16:
                    arr = (np.clip(arr, -1.0, 1.0) * 32767).astype(np.int16)
                chunks.append(arr.reshape(-1))
    except Exception as exc:  # noqa: BLE001
        log.warning("解码失败: %s", exc)
        return np.zeros(0, dtype=np.int16)
    if not chunks:
        return np.zeros(0, dtype=np.int16)
    return np.concatenate(chunks)


def polish(pcm: np.ndarray, sample_rate: int) -> np.ndarray:
    """响度对齐 + 首尾淡入淡出，避免音乐比 TTS 小声、切歌时爆音。"""
    if pcm.size == 0:
        return pcm
    peak = int(np.abs(pcm).max())
    if peak > 0:
        gain = min(6.0, 0.88 * 32767.0 / peak)
        if gain > 1.02:
            pcm = np.clip(pcm.astype(np.float32) * gain, -32768, 32767).astype(np.int16)
    fade_in = min(int(sample_rate * 0.15), pcm.size // 4)
    fade_out = min(int(sample_rate * 0.40), pcm.size // 4)
    if fade_in > 1:
        ramp = np.linspace(0.0, 1.0, fade_in, dtype=np.float32)
        pcm[:fade_in] = (pcm[:fade_in].astype(np.float32) * ramp).astype(np.int16)
    if fade_out > 1:
        ramp = np.linspace(1.0, 0.0, fade_out, dtype=np.float32)
        pcm[-fade_out:] = (pcm[-fade_out:].astype(np.float32) * ramp).astype(np.int16)
    return pcm


class MusicSource:
    """网易云搜索 + 直链下载 + 本地缓存。"""

    def __init__(
        self,
        cache_dir: Path,
        timeout: float = 12.0,
        cache_max_mb: int = 200,
        sources: Any = None,
    ) -> None:
        # 音源链：按顺序试，前一个拿不到才问下一个
        self.sources = _parse_sources(sources)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = float(timeout)
        self.cache_max_bytes = int(cache_max_mb) * 1024 * 1024
        self._session: Any = None

    @property
    def _state_path(self) -> Path:
        # 和缓存放同一层（data/music/），换歌不丢、重启不丢
        return self.cache_dir.parent / "music_state.json"

    def load_mode(self) -> str:
        """读回上次选的播放模式；文件坏了就当没存过。"""
        try:
            data = json.loads(self._state_path.read_text(encoding="utf-8"))
            mode = str((data or {}).get("mode") or "").strip().lower()
            if mode in MUSIC_MODES:
                return mode
        except Exception:  # noqa: BLE001
            pass
        return MUSIC_MODE_DEFAULT

    def save_mode(self, mode: str) -> None:
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            self._state_path.write_text(
                json.dumps({"mode": mode}, ensure_ascii=False), encoding="utf-8"
            )
        except Exception as exc:  # noqa: BLE001
            log.debug("保存播放模式失败: %s", exc)

    async def _http(self) -> Any:
        import aiohttp

        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers=_HEADERS,
                timeout=aiohttp.ClientTimeout(total=self.timeout),
            )
        return self._session

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            try:
                await self._session.close()
            except Exception:  # noqa: BLE001
                pass
        self._session = None

    # ---------- 搜索 ----------
    async def search(self, keyword: str, limit: int = 8) -> List[Dict[str, Any]]:
        """给界面用的搜索：网易云优先，它一条都没有时用酷我顶上。"""
        keyword = str(keyword or "").strip()
        if not keyword:
            return []
        songs = await self._netease_search(keyword, limit)
        if not songs and "kuwo" in self.sources:
            songs = await self._kuwo_search(keyword, limit)
        return songs

    async def _netease_search(self, keyword: str, limit: int = 8) -> List[Dict[str, Any]]:
        keyword = str(keyword or "").strip()
        if not keyword:
            return []
        url = NETEASE_SEARCH + "?" + urllib.parse.urlencode(
            {"s": keyword, "type": 1, "limit": max(1, min(20, limit))}
        )
        try:
            session = await self._http()
            async with session.get(url) as resp:
                if resp.status != 200:
                    log.warning("搜索返回 %s", resp.status)
                    return []
                data = await resp.json(content_type=None)
        except Exception as exc:  # noqa: BLE001
            log.warning("搜索失败: %s", exc)
            return []
        songs = ((data.get("result") or {}).get("songs")) or []
        out: List[Dict[str, Any]] = []
        for item in songs:
            try:
                artists = ", ".join(
                    str(a.get("name") or "") for a in (item.get("artists") or [])
                )
                out.append(
                    {
                        "id": str(item.get("id") or ""),
                        "title": str(item.get("name") or ""),
                        "artist": artists,
                        "duration": int((item.get("duration") or 0) / 1000),
                        "play_count": int(item.get("playCount") or 0),
                        "provider": "netease",
                    }
                )
            except Exception:  # noqa: BLE001
                continue
        return [s for s in out if s["id"]]

    # ---------- 直链 ----------
    def _cache_path(self, song_id: str) -> Path:
        return self.cache_dir / f"{song_id}.mp3"

    async def fetch(self, song: Dict[str, Any]) -> Optional[bytes]:
        """拿到这首歌的 mp3 字节；拿不到返回 None。"""
        sid = str(song.get("id") or "")
        if not sid:
            return None
        path = self._cache_path(sid)
        try:
            if path.exists() and path.stat().st_size >= MIN_VALID_BYTES:
                # 淘汰是按 mtime 挑最旧的，命中时不刷新的话，
                # 常听的那几首反而会先被删掉
                os.utime(path, None)
                return path.read_bytes()
        except Exception:  # noqa: BLE001
            pass
        provider = str(song.get("provider") or "netease")
        if provider == "kuwo":
            return await self._fetch_kuwo(song, path)
        if provider == "suyin163":
            return await self._fetch_suyin163(song, path)
        if provider == "suyinkw":
            return await self._fetch_suyin_kw(song, path)
        return await self._fetch_netease(song, path)

    def _save_cache(self, path: Path, data: bytes, truncated: bool) -> None:
        try:
            if truncated:
                log.info(
                    "「%s」超过 %dMB 被截断，本次可放但不写缓存",
                    path.stem,
                    MAX_DOWNLOAD_BYTES // 1024 // 1024,
                )
            else:
                path.write_bytes(data)
            self._trim_cache()
        except Exception as exc:  # noqa: BLE001
            log.debug("写缓存失败: %s", exc)

    async def _fetch_netease(self, song: Dict[str, Any], path: Path) -> Optional[bytes]:
        sid = str(song.get("id") or "")
        url = NETEASE_OUTER.format(sid)
        data = b""
        truncated = False
        try:
            session = await self._http()
            async with session.get(url) as resp:
                ctype = str(resp.headers.get("Content-Type") or "").lower()
                if "audio" not in ctype:
                    # 版权受限时会返回一个 html 错误页
                    return None
                buf = bytearray()
                async for chunk in resp.content.iter_chunked(65536):
                    buf.extend(chunk)
                    if len(buf) > MAX_DOWNLOAD_BYTES:
                        # 超上限大概率是长音频或者服务器给错了东西。这份数据还能放，
                        # 但绝不能写进缓存，否则以后每次放都是同一段残曲。
                        truncated = True
                        break
                data = bytes(buf)
        except Exception as exc:  # noqa: BLE001
            log.warning("下载失败 %s: %s", sid, exc)
            return None
        if len(data) < MIN_VALID_BYTES:
            return None
        self._save_cache(path, data, truncated)
        return data

    async def _fetch_kuwo(self, song: Dict[str, Any], path: Path) -> Optional[bytes]:
        """酷我：先用 rid 换直链再下载。试听片段一律当无效。"""
        rid = str(song.get("rid") or "")
        if not rid:
            return None
        url = ""
        data = b""
        truncated = False
        try:
            session = await self._http()
            api = KUWO_URL + "?" + urllib.parse.urlencode(
                {
                    "type": "convert_url3",
                    "rid": f"MUSIC_{rid}",
                    "format": "mp3",
                    "response": "url",
                }
            )
            async with session.get(api, headers=KUWO_HEADERS) as resp:
                if resp.status != 200:
                    log.info("酷我直链返回 %s（%s）", resp.status, rid)
                    return None
                text = (await resp.text()).strip()
            if text.startswith("{"):
                try:
                    payload = json.loads(text)
                except Exception:  # noqa: BLE001
                    payload = {}
                url = str(
                    payload.get("url") or (payload.get("data") or {}).get("url") or ""
                )
            elif text.startswith("http"):
                url = text
            if not url.startswith("http"):
                log.info("酷我没给出直链（%s）：%s", rid, text[:80])
                return None
            async with session.get(url, headers=KUWO_HEADERS) as resp:
                ctype = str(resp.headers.get("Content-Type") or "").lower()
                if "audio" not in ctype and "octet-stream" not in ctype:
                    log.info("酷我下到的不是音频（%s）：%s", rid, ctype)
                    return None
                buf = bytearray()
                async for chunk in resp.content.iter_chunked(65536):
                    buf.extend(chunk)
                    if len(buf) > MAX_DOWNLOAD_BYTES:
                        truncated = True
                        break
                data = bytes(buf)
        except Exception as exc:  # noqa: BLE001
            log.warning("酷我下载失败 %s: %s", rid, exc)
            return None
        if len(data) < MIN_VALID_BYTES:
            return None
        # 酷我对付费歌只给十来秒试听，体积也有 180KB 左右，光看体积拦不住
        secs = _mp3_seconds(data)
        declared = int(song.get("duration") or 0)
        if secs and declared >= 90 and secs < max(MIN_PLAY_SECONDS, declared * 0.5):
            log.info(
                "酷我「%s」只给了 %.1f 秒试听（标称 %d 秒），跳过",
                song.get("title") or rid,
                secs,
                declared,
            )
            return None
        self._save_cache(path, data, truncated)
        return data

    # ---------- 溯音兜底（不搜索，只换链）----------
    def _accept(
        self, song: Dict[str, Any], data: bytes, truncated: bool, path: Path, tag: str
    ) -> Optional[bytes]:
        """溯音这类第三方代理会把试听片段、甚至 1 秒的垃圾流也返回来，
        所以体积和数帧时长两道都要过，过不了直接丢。"""
        if len(data) < MIN_VALID_BYTES:
            log.info("%s「%s」只给了 %d 字节，跳过", tag, song.get("title"), len(data))
            return None
        secs = _mp3_seconds(data)
        declared = int(song.get("duration") or 0)
        limit = max(MIN_PLAY_SECONDS, declared * 0.5) if declared >= 90 else MIN_PLAY_SECONDS
        if secs < limit:
            log.info(
                "%s「%s」只有 %.1f 秒（标称 %d 秒），跳过",
                tag,
                song.get("title") or song.get("id"),
                secs,
                declared,
            )
            return None
        self._save_cache(path, data, truncated)
        return data

    async def _suyin_download(
        self, song: Dict[str, Any], url: str, tag: str
    ) -> Optional[Tuple[bytes, bool]]:
        if not url.startswith("http"):
            return None
        data = b""
        truncated = False
        session = await self._http()
        async with session.get(url, headers={"User-Agent": _UA}) as resp:
            ctype = str(resp.headers.get("Content-Type") or "").lower()
            if "audio" not in ctype and "octet-stream" not in ctype:
                log.info("%s 下到的不是音频（%s）：%s", tag, ctype, song.get("title"))
                return None
            buf = bytearray()
            async for chunk in resp.content.iter_chunked(65536):
                buf.extend(chunk)
                if len(buf) > MAX_DOWNLOAD_BYTES:
                    truncated = True
                    break
            data = bytes(buf)
        return data, truncated

    async def _fetch_suyin163(self, song: Dict[str, Any], path: Path) -> Optional[bytes]:
        """溯音网易云代理：拿网易云 id 换直链，专治我们直连 404 的付费原唱。"""
        sid = str(song.get("id") or "").replace("sy163", "")
        if not sid:
            return None
        try:
            session = await self._http()
            async with session.get(
                SUYIN_163 + "?" + urllib.parse.urlencode({"id": sid}),
                headers={"User-Agent": _UA},
            ) as resp:
                if resp.status != 200:
                    log.info("溯音163 返回 %s（%s）", resp.status, sid)
                    return None
                payload = await resp.json(content_type=None)
            items = payload.get("data") or []
            if isinstance(items, dict):
                items = [items]
            url = str((items[0] if items else {}).get("url") or "")
            if int(payload.get("code") or 0) != 0 or not url:
                log.info("溯音163 没给链接（%s）：%s", sid, str(payload)[:80])
                return None
            got = await self._suyin_download(song, url, "溯音163")
        except Exception as exc:  # noqa: BLE001
            log.warning("溯音163 失败 %s: %s", sid, exc)
            return None
        if not got:
            return None
        return self._accept(song, got[0], got[1], path, "溯音163")

    async def _fetch_suyin_kw(self, song: Dict[str, Any], path: Path) -> Optional[bytes]:
        """溯音酷我代理：按「歌名 歌手」问，直链藏在 message 文本里。"""
        title = str(song.get("title") or "").strip()
        if not title:
            return None
        artist = str(song.get("artist") or "").replace("、", ",").split(",")[0].strip()
        msg = f"{title} {artist}".strip()
        try:
            session = await self._http()
            async with session.get(
                SUYIN_KUWO + "?" + urllib.parse.urlencode({"msg": msg, "n": 1, "br": 7}),
                headers={"User-Agent": _UA},
            ) as resp:
                if resp.status != 200:
                    log.info("溯音酷我返回 %s（%s）", resp.status, msg)
                    return None
                payload = await resp.json(content_type=None)
            text = str(payload.get("message") or "")
            link = _SUYIN_LINK_RE.search(text)
            if not link:
                log.info("溯音酷我没给链接（%s）：%s", msg, text[:80])
                return None
            hit = _SUYIN_NAME_RE.search(text)
            hit_name = _norm(hit.group(1)) if hit else ""
            if hit_name and _norm(title) not in hit_name:
                log.info("溯音酷我回的是别的歌（%s → %s），跳过", msg, hit.group(1))
                return None
            got = await self._suyin_download(song, link.group(1), "溯音酷我")
        except Exception as exc:  # noqa: BLE001
            log.warning("溯音酷我失败 %s: %s", msg, exc)
            return None
        if not got:
            return None
        return self._accept(song, got[0], got[1], path, "溯音酷我")

    async def _pick_suyin(
        self, keyword: str, tries: int = 4
    ) -> Tuple[Optional[Dict[str, Any]], Optional[bytes]]:
        """两家直连都失败时，拿网易云/酷我的候选去溯音问一遍。

        先试溯音163（有网易云 id，命中率更高、返回是结构化 JSON），
        再试溯音酷我（要歌名+歌手，直链塞在文本里）。
        """
        songs = await self._netease_search(keyword, limit=10)
        if songs:
            songs = sorted(songs, key=lambda s: _rank(keyword, s), reverse=True)
            for song in songs[: max(1, tries)]:
                cand = dict(song)
                cand["provider"] = "suyin163"
                cand["id"] = f"sy163{song.get('id')}"
                data = await self.fetch(cand)
                if data:
                    return cand, data
        candidates = await self._kuwo_search(keyword, limit=8)
        for song in candidates[: max(1, tries)]:
            cand = dict(song)
            cand["provider"] = "suyinkw"
            cand["id"] = f"sykw{song.get('id')}"
            data = await self.fetch(cand)
            if data:
                return cand, data
        return None, None

    async def _kuwo_search(self, keyword: str, limit: int = 8) -> List[Dict[str, Any]]:
        url = KUWO_SEARCH + "?" + urllib.parse.urlencode(
            {
                "all": keyword,
                "ft": "music",
                "itemset": "web_2013",
                "client": "kt",
                "pn": "0",
                "rn": str(max(1, min(20, limit))),
                "rformat": "json",
                "encoding": "utf8",
                "vipver": "1",
            }
        )
        try:
            session = await self._http()
            async with session.get(url, headers=KUWO_HEADERS) as resp:
                if resp.status != 200:
                    log.warning("酷我搜索返回 %s", resp.status)
                    return []
                raw = (await resp.text()).strip()
        except Exception as exc:  # noqa: BLE001
            log.warning("酷我搜索失败: %s", exc)
            return []
        data: Any = {}
        try:
            text = _KUWO_TRUE_RE[0].sub("True", raw)
            text = _KUWO_TRUE_RE[1].sub("False", text)
            text = _KUWO_TRUE_RE[2].sub("None", text)
            data = ast.literal_eval(text)
        except Exception as exc:  # noqa: BLE001
            log.warning("酷我搜索解析失败: %s", exc)
            return []
        if not isinstance(data, dict):
            return []
        out: List[Dict[str, Any]] = []
        for item in data.get("abslist") or []:
            if not isinstance(item, dict):
                continue
            rid = str(item.get("MUSICRID") or "").replace("MUSIC_", "").strip()
            if not rid:
                continue
            out.append(
                {
                    # 加前缀，避免和网易云的数字 id 撞车（缓存文件名就是 id）
                    "id": f"kuw{rid}",
                    "rid": rid,
                    "provider": "kuwo",
                    "title": _clean_kuwo(item.get("SONGNAME")),
                    "artist": _clean_kuwo(item.get("ARTIST")),
                    "duration": int(float(item.get("DURATION") or 0)),
                    "play_count": 0,
                }
            )
        return out

    def _trim_cache(self) -> None:
        """缓存超过上限时按最后访问时间删最旧的。"""
        try:
            files = [p for p in self.cache_dir.glob("*.mp3") if p.is_file()]
            total = sum(p.stat().st_size for p in files)
            if total <= self.cache_max_bytes:
                return
            files.sort(key=lambda p: p.stat().st_mtime)
            for p in files:
                if total <= self.cache_max_bytes:
                    break
                try:
                    size = p.stat().st_size
                    p.unlink()
                    total -= size
                except Exception:  # noqa: BLE001
                    pass
        except Exception:  # noqa: BLE001
            pass

    async def pick(
        self, keyword: str, tries: int = 3
    ) -> Tuple[Optional[Dict[str, Any]], Optional[bytes]]:
        """按音源链逐个试：先网易云，拿不到再问酷我。"""
        keyword = str(keyword or "").strip()
        if not keyword:
            return None, None
        for provider in self.sources:
            if provider == "suyin":
                continue  # 溯音不搜索，只兜底，等直连都试完再走
            if provider == "netease":
                song, data = await self._pick_netease(keyword, tries)
            else:
                song, data = await self._pick_kuwo(keyword, tries=max(4, tries + 3))
            if song and data:
                if provider != "netease":
                    log.info(
                        "「%s」网易云没拿到，改用酷我：%s / %s",
                        keyword,
                        song.get("title"),
                        song.get("artist"),
                    )
                return song, data
        if "suyin" in self.sources:
            song, data = await self._pick_suyin(keyword, tries=tries + 1)
            if song and data:
                log.info(
                    "「%s」直连都没拿到，溯音补到：%s / %s",
                    keyword,
                    song.get("title"),
                    song.get("artist"),
                )
                return song, data
        return None, None

    async def _pick_netease(
        self, keyword: str, tries: int = 3
    ) -> Tuple[Optional[Dict[str, Any]], Optional[bytes]]:
        """网易云：搜索并按顺序试前几首，返回第一首真能拿到音频的。"""
        songs = await self._netease_search(keyword, limit=10)
        if not songs:
            return None, None
        # 网易云的默认排序会先把翻唱、DJ 版顶上来（搜「稻香」第一条是「稻香(深情版)/Lucky小爱」），
        # 这里按「标题是否就是这个词 + 歌手是否在词里 + 播放量」重新打分，
        # 让原唱排在翻唱前面。
        songs = sorted(songs, key=lambda s: _rank(keyword, s), reverse=True)
        top = songs[0]
        data = await self.fetch(top)
        if data:
            return top, data
        # 网易云「最像的那首」直链 404，多半是 VIP 原唱（搜「孤勇者」最高分是陈奕迅，
        # 但网易云给我们 404）。先用溯音代理把这首原唱捞回来——它给的还是同一个
        # 网易云 id，只是换了条链接——捞不到再退到酷我同名版，最后才让翻唱顶上。
        if "suyin" in self.sources:
            cand = dict(top)
            cand["provider"] = "suyin163"
            cand["id"] = f"sy163{top.get('id')}"
            data = await self.fetch(cand)
            if data:
                log.info(
                    "网易云下不动「%s / %s」，溯音163 补到原唱",
                    top.get("title"),
                    top.get("artist"),
                )
                return cand, data
        if "kuwo" in self.sources:
            song, data = await self._kuwo_match(top)
            if data:
                return song, data
        for song in songs[1: max(1, tries)]:
            data = await self.fetch(song)
            if data:
                return song, data
        return None, None

    async def _kuwo_match(
        self, want: Dict[str, Any]
    ) -> Tuple[Optional[Dict[str, Any]], Optional[bytes]]:
        """去酷我找「同一首」：标题一致、歌手对得上。"""
        title = str(want.get("title") or "").strip()
        if not title:
            return None, None
        artist = str(want.get("artist") or "").replace("、", ",").split(",")[0].strip()
        candidates = await self._kuwo_search(f"{title} {artist}".strip(), limit=10)
        want_t = _clean_title(title)
        want_a = _norm(artist)
        want_d = int(want.get("duration") or 0)
        for cand in candidates:
            raw_t = str(cand.get("title") or "")
            if any(w in raw_t.lower() for w in _BAD_VERSION_WORDS):
                continue
            # 标题必须「同一首」：去括号后完全一致。「稻香老友」这种加料的、
            # 「小跳蛙+拔萝卜+…（片段）」这种串烧，都不算同一首。
            if _clean_title(raw_t) != want_t:
                continue
            cand_a = _norm(cand.get("artist"))
            if want_a and want_a not in cand_a:
                continue
            cand_d = int(cand.get("duration") or 0)
            if want_d >= 90 and cand_d and cand_d < want_d * 0.7:
                continue
            data = await self.fetch(cand)
            if data:
                log.info("网易云下不动「%s/%s」，改用酷我同名版", title, artist)
                return cand, data
        return None, None

    async def _pick_kuwo(
        self, keyword: str, tries: int = 6
    ) -> Tuple[Optional[Dict[str, Any]], Optional[bytes]]:
        """酷我：按它自己的相关度顺序试，多试几首（热门付费歌会只给试听，得往下翻）。

        不套 _rank 重排：酷我的相关度本身挺好（「勇气大爆发」第一条是贝乐虎儿歌、
        「听我说谢谢你」第一条是李昕融），套上「标题完全一致 +1000」反而会把
        腾格尔/洛天依这种翻唱顶到孩子的儿歌版前面。
        """
        songs = await self._kuwo_search(keyword, limit=10)
        if not songs:
            return None, None
        for song in songs[: max(1, tries)]:
            data = await self.fetch(song)
            if data:
                return song, data
        return None, None


class MusicPlayer:
    """绑在一台设备上的播放器：解码一次，按 20ms 帧节奏推流。"""

    def __init__(self, session: Any, source: MusicSource, bitrate: int = 64000) -> None:
        self.sess = session
        self.source = source
        self.bitrate = int(bitrate)
        self.current: Optional[Dict[str, Any]] = None
        self.queue: List[Dict[str, Any]] = []
        self.index = 0
        self.duration = 0.0
        self._pos_frames = 0
        self._task: Optional[asyncio.Task] = None
        self._stop = asyncio.Event()
        self._paused = asyncio.Event()
        self._start_offset = 0.0
        self.started_at = 0.0
        # 代际号：切歌/重播时旧任务会退出，它不能在 finally 里发 tts stop
        # （一发设备就回聆听态，新任务的第一帧还没到就被掐了）
        self._gen = 0
        # 循环模式：默认列表循环 —— 说「放儿歌」会拿到一整页搜索结果，
        # 播完接着播比每首都得重新喊一次自然。改过的模式落盘，服务重启还记得。
        self._mode: str = self.source.load_mode()

    @property
    def playing(self) -> bool:
        return self._task is not None and not self._task.done()

    @property
    def paused(self) -> bool:
        return self._paused.is_set()

    @property
    def position(self) -> float:
        return self._pos_frames * (self.sess.down_frame_ms / 1000.0)

    def status(self) -> Dict[str, Any]:
        song = self.current or {}
        return {
            "playing": self.playing,
            "paused": bool(self.playing and self._paused.is_set()),
            "song": {
                "id": str(song.get("id") or ""),
                "title": str(song.get("title") or ""),
                "artist": str(song.get("artist") or ""),
                "duration": int(song.get("duration") or 0),
            },
            "position": round(self.position, 1) if self.playing else 0.0,
            "duration": round(self.duration, 1),
            "index": self.index,
            "mode": self._mode,
            "mode_text": MUSIC_MODE_TEXT.get(self._mode, ""),
            "queue": [
                {
                    "id": str(s.get("id") or ""),
                    "title": str(s.get("title") or ""),
                    "artist": str(s.get("artist") or ""),
                }
                for s in self.queue
            ],
        }

    async def prepare(self, keyword: str) -> Optional[Dict[str, Any]]:
        """只搜歌 + 拿音频，不开始播放（等 TTS 把话说完再放）。

        顺带把整页搜索结果作为队列带回去：孩子说「放儿歌」会搜到好几首，
        第一首放完按模式接着放，不用每首都重新喊一次。
        """
        song, data = await self.source.pick(keyword)
        if not song or not data:
            return None
        queue: List[Dict[str, Any]] = []
        try:
            prov = str(song.get("provider") or "netease")
            if prov in ("kuwo", "suyinkw"):
                queue = list(await self.source._kuwo_search(keyword, limit=8))
            else:
                queue = list(await self.source.search(keyword, limit=8))
        except Exception as exc:  # noqa: BLE001
            log.debug("取队列失败（忽略）: %s", exc)
        index = 0
        if queue:
            ids = [str(s.get("id") or "") for s in queue]
            sid = str(song.get("id") or "")
            if sid in ids:
                index = ids.index(sid)
            else:
                queue.insert(0, song)
        else:
            queue = [song]
        return {"song": song, "data": data, "queue": queue, "index": index}

    async def prepare_song(self, song: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """指定某一首歌（搜索结果里的一首）取音频。"""
        if not song or not song.get("id"):
            return None
        data = await self.source.fetch(song)
        if not data:
            return None
        return {"song": song, "data": data}

    @property
    def mode(self) -> str:
        return self._mode

    def set_mode(self, mode: str) -> bool:
        """切换播放模式，值非法返回 False。"""
        mode = str(mode or "").strip().lower()
        if mode not in MUSIC_MODES:
            return False
        self._mode = mode
        self.source.save_mode(mode)
        log.info("播放模式切换：%s", MUSIC_MODE_TEXT.get(mode, mode))
        return True

    def _advance_index(self, step: int) -> Optional[int]:
        """按播放模式算下一个位置；没有下一首返回 None。"""
        if not self.queue:
            return None
        total = len(self.queue)
        if step > 0:
            if self._mode == "shuffle" and total > 1:
                others = [i for i in range(total) if i != self.index]
                return random.choice(others)
            nxt = self.index + 1
            if nxt >= total:
                # 列表/单曲循环回到第一首；只有顺序播放到这停
                return 0 if self._mode in ("all", "one") else None
            return nxt
        prev = self.index - 1
        if prev < 0:
            return total - 1 if self._mode in ("all", "shuffle", "one") else None
        return prev

    async def next_track(self) -> Optional[Dict[str, Any]]:
        """手动下一首（按模式取位置，one 模式也真的往后走）。"""
        idx = self._advance_index(1)
        if idx is None:
            return None
        prepared = await self.prepare_song(self.queue[idx])
        if prepared is None:
            return None
        self.index = idx
        return prepared

    async def prev_track(self) -> Optional[Dict[str, Any]]:
        idx = self._advance_index(-1)
        if idx is None:
            return None
        prepared = await self.prepare_song(self.queue[idx])
        if prepared is None:
            return None
        self.index = idx
        return prepared

    def _halted(self, gen: int, stop: asyncio.Event) -> bool:
        """这一路推流该不该停了。

        不能只看 self._stop：start() 换歌时会 new 一个 Event 赋回 self._stop，
        旧任务如果读 self._stop 就会看到「没被置位」而继续推流，于是两路 opus
        交错下发，设备放出来是杂音（实测切歌后 1 秒推了 100 帧，单路只该 50）。
        所以每个任务必须抱紧自己那一代的 stop 事件 + 代际号。
        """
        return (
            stop.is_set()
            or gen != self._gen
            or self.sess._closed
            or getattr(self.sess, "_abort", False)
        )

    def start(
        self,
        prepared: Dict[str, Any],
        queue: Optional[List[Dict[str, Any]]] = None,
        index: int = 0,
        offset_seconds: float = 0.0,
    ) -> None:
        self.stop_nowait()
        if getattr(self.sess, "_closed", False):
            return
        stop = asyncio.Event()
        self._stop = stop
        self._paused.clear()
        self.current = prepared.get("song")
        self.queue = list(queue or [prepared.get("song") or {}])
        self.index = max(0, min(int(index), len(self.queue) - 1)) if self.queue else 0
        self.duration = 0.0
        self._pos_frames = 0
        # 续播用：从第几秒开始放（只对队列里的第一首生效）
        self._start_offset = max(0.0, float(offset_seconds or 0.0))
        self.started_at = time.time()
        self._gen += 1
        self._task = asyncio.create_task(
            self._run(prepared, self._gen, stop), name="music-play"
        )

    def pause(self) -> bool:
        if not self.playing:
            return False
        self._paused.set()
        return True

    def resume(self) -> bool:
        if not self.playing:
            return False
        self._paused.clear()
        return True

    def stop_nowait(self) -> None:
        self._stop.set()
        self._paused.clear()
        self.current = None
        task, self._task = self._task, None
        if task is None or task.done():
            return
        # 不立刻 cancel：让 _stream 自己看到 _stop 退出，tts stop 才发得出去。
        # 万一它卡在 send 上，2 秒后强杀兜底。
        try:
            asyncio.get_running_loop().create_task(self._reap(task), name="music-reap")
        except RuntimeError:
            task.cancel()

    async def _reap(self, task: asyncio.Task) -> None:
        try:
            await asyncio.wait({task}, timeout=2.0)
        except Exception:  # noqa: BLE001
            pass
        if not task.done():
            task.cancel()

    async def stop(self) -> None:
        self.stop_nowait()

    async def _next_item(self, gen: int, stop: asyncio.Event) -> Optional[Dict[str, Any]]:
        """队列里的下一首；没有或者取不到音频就返回 None。"""
        if self._halted(gen, stop):
            return None
        idx = self._advance_index(1)
        if idx is None:
            return None
        self.index = idx
        song = self.queue[idx]
        try:
            data = await self.source.fetch(song)
        except Exception as exc:  # noqa: BLE001
            log.warning("下一首取音频失败: %s", exc)
            return None
        if not data:
            log.info("下一首「%s」拿不到音频，停在这里", song.get("title"))
            return None
        return {"song": song, "data": data}

    async def _run(
        self, prepared: Dict[str, Any], gen: int = 0, stop: Optional[asyncio.Event] = None
    ) -> None:
        sess = self.sess
        if stop is None:
            stop = self._stop
        announced = False
        try:
            item: Optional[Dict[str, Any]] = prepared
            while item is not None:
                if self._halted(gen, stop):
                    break
                song = item.get("song") or {}
                self.current = song
                pcm = await asyncio.to_thread(decode_bytes, item["data"], sess.down_rate)
                if pcm.size == 0:
                    log.warning("「%s」解码出来是空的", song.get("title"))
                    item = await self._next_item(gen, stop)
                    continue
                pcm = polish(pcm, sess.down_rate)
                self.duration = pcm.size / float(sess.down_rate)
                offset = self._start_offset
                self._start_offset = 0.0
                self._pos_frames = 0
                log.info(
                    "开始播放：%s - %s（%.1fs%s）",
                    song.get("title"),
                    song.get("artist"),
                    self.duration,
                    f"，从 {offset:.0f}s 续播" if offset > 0 else "",
                )
                # 界面上先出歌名和总时长，进度从 offset 起算
                try:
                    await sess.show_music(
                        song.get("title") or "",
                        song.get("artist") or "",
                        int(round(self.duration)),
                        int(round(offset)),
                        False,
                        self._mode,
                    )
                except Exception as exc:  # noqa: BLE001
                    log.debug("音乐界面下发失败: %s", exc)
                ok = await self._stream(pcm, gen, stop, announce=not announced,
                                        start_offset=offset)
                announced = True
                if not ok:
                    log.info("播放中止：%s", song.get("title"))
                    break
                log.info("播放结束：%s", song.get("title"))
                if self._mode == "one":
                    # 单曲循环：手头这份解码结果直接重放，不用重新下载/解码
                    item = {"song": song, "data": item["data"]}
                    continue
                item = await self._next_item(gen, stop)
        except asyncio.CancelledError:
            log.info("播放被打断：%s", (self.current or {}).get("title"))
            raise
        except Exception as exc:  # noqa: BLE001
            log.warning("播放失败: %s", exc)
        finally:
            # 代际不匹配 = 自己已经被新的一次播放顶掉了：别动共享状态、别发 tts stop，
            # 否则设备会回聆听态，新任务的第一帧还没到就被掐了。
            if gen == self._gen:
                if announced:
                    # 整个队列放完（或中途断掉）才发一次 tts stop。单曲之间绝不能发：
                    # 一发设备就回聆听态、开始上传麦克风，VAD 会把自己的声音当人声把音乐掐掉。
                    try:
                        await asyncio.shield(sess.send_json({"type": "tts", "state": "stop"}))
                    except BaseException:  # noqa: BLE001
                        pass
                sess._speaking = False
                if getattr(sess.stats, "state", "") == "speaking":
                    sess.stats.state = "idle"
                self.current = None
                self._paused.clear()
                # 队列放完 / 被停掉：收屏幕，背光由固件恢复
                try:
                    await asyncio.shield(sess.hide_music())
                except BaseException:  # noqa: BLE001
                    pass

    async def _stream(
        self,
        pcm: np.ndarray,
        gen: int = 0,
        stop: Optional[asyncio.Event] = None,
        announce: bool = True,
        start_offset: float = 0.0,
    ) -> bool:
        sess = self.sess
        if stop is None:
            stop = self._stop
        enc = OpusDownlinkEncoder(sess.down_rate, sess.down_frame_ms, self.bitrate)
        interval = sess.down_frame_ms / 1000.0
        step = enc.frame_samples
        next_at = time.monotonic()
        total = pcm.size
        pos = 0
        if start_offset > 0:
            # 续播：按整帧对齐地跳过前面那段，进度也一起对到断点
            skip = int(start_offset * sess.down_rate) // step * step
            if skip < total:
                pos = skip
                self._pos_frames = pos // step
        stopped = False
        sess._speaking = True
        sess.stats.state = "speaking"
        if announce:
            # 必须先发 tts start：固件收到它才切到 speaking 状态、才开始播下行音频。
            # 只推 opus 帧的话设备会当没收到（实测推了 43 秒一声不出）。
            await sess.send_json({"type": "tts", "state": "start"})
        while pos < total and not stopped:
            if self._halted(gen, stop):
                stopped = True
                break
            if self._paused.is_set():
                # 暂停：不推进 pos、不推帧，等恢复或停止。恢复后重新对齐节拍，
                # 否则暂停多久就欠多少帧，恢复瞬间会一口气猛推。
                try:
                    await sess.update_music(int(self.position), int(round(self.duration)), True)
                except Exception:  # noqa: BLE001
                    pass
                paused_at = time.monotonic()
                while self._paused.is_set() and not self._halted(gen, stop):
                    await asyncio.sleep(0.05)
                    if time.monotonic() - paused_at >= PAUSE_STOP_SECONDS:
                        # 暂停太久：按停止处理。设备一直停在 speaking 就不省电，
                        # 而且孩子多半已经走开了；进度记下来，说「继续播放」能接上。
                        log.info(
                            "暂停超过 %.0f 秒，按停止处理（进度 %.0fs）",
                            PAUSE_STOP_SECONDS,
                            self.position,
                        )
                        try:
                            sess.record_music_resume(self.position)
                        except Exception as exc:  # noqa: BLE001
                            log.debug("记录暂停进度失败: %s", exc)
                        self._paused.clear()
                        stopped = True
                        break
                if stopped:
                    break
                try:
                    await sess.update_music(int(self.position), int(round(self.duration)), False)
                except Exception:  # noqa: BLE001
                    pass
                next_at = time.monotonic()
                continue
            for payload in enc.encode(pcm[pos : pos + step]):
                if self._halted(gen, stop):
                    stopped = True
                    break
                await sess.send_audio(payload)
                self._pos_frames += 1
                next_at += interval
                delay = next_at - time.monotonic()
                if delay > 0:
                    await asyncio.sleep(delay)
                else:
                    # 落后太多就重新对齐，别越积越多
                    next_at = time.monotonic()
            pos += step
        if not stopped:
            for payload in enc.flush():
                if self._halted(gen, stop):
                    stopped = True
                    break
                await sess.send_audio(payload)
        return not stopped
