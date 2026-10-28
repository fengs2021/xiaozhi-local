"""B 站音频源：搜儿童故事合集 → 按分P取 DASH 音轨 → 整段下载。

实现参考 fengqiao57/abts（MIT，Dart 写的第三方 B 站有声小说播放器）的三段风控闭环：

1. WBI 签名 —— nav 接口给 img_url/sub_url，按 64 位 mixinKey 表拼出 key，参数加 wts
   排序后 w_rid = md5(参数串 + key)。不带 w_rid，搜索和 playurl 直接 412。
2. buvid 激活 —— 本地造 buvid3/buvid4 后上报 /x/internal/gaia-gateway/ExClimbWuzhi。
   不做这步，pagelist / view 一律 412（本机实测复现：410 之后激活立刻 code 0）。
3. 限速退避 —— 匿名请求最小间隔 min_gap（默认 600ms），命中 -412/-352 时
   1.2/2.6/5.0 秒退避，并丢掉 buvid 重新预热。

取流走 DASH：playurl 传 fnval=4048 拿 data.dash.audio[]，按带宽挑轨（30280=192K、
30232=132K、30216=64K、30250=Dolby）。下载时 **Referer 必须是 https://www.bilibili.com/
本身**——实测换成视频页 Referer 或者不带 Referer，CDN 一律 503 空响应 0 字节。
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import random
import re
import time
import urllib.parse
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger("bili")

WEB_HOME = "https://www.bilibili.com/"
API = "https://api.bilibili.com"
NAV = "/x/web-interface/nav"
SEARCH = "/x/web-interface/wbi/search/type"
PAGELIST = "/x/player/pagelist"
PLAYURL = "/x/player/wbi/playurl"
ACTIVATE = "/x/internal/gaia-gateway/ExClimbWuzhi"
SEARCH_HOME = "https://search.bilibili.com"

_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    " (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)

# WBI 混淆表（abts lib/core/network/wbi_signer.dart 里的 64 位魔改表）
_MIXIN_TAB = [
    46, 47, 18, 2, 53, 8, 23, 32, 15, 50, 10, 31, 58, 3, 45, 35,
    27, 43, 5, 49, 33, 9, 42, 19, 29, 28, 14, 39, 12, 38, 41, 13,
    37, 48, 7, 16, 24, 55, 40, 61, 26, 17, 0, 1, 60, 51, 30, 4,
    22, 25, 54, 21, 56, 59, 6, 63, 57, 62, 11, 36, 20, 34, 44, 52,
]
_SIGN_STRIP = "!'()*"

# 风控重试用
_RISK_CODES = (-412, -352)
_BACKOFF = (1.2, 2.6, 5.0)

MAX_AUDIO_BYTES = 24 * 1024 * 1024
MIN_AUDIO_BYTES = 60 * 1024  # 单集语音一般几百 KB，再小基本是空响应

# 故事类内容的标题特征词（命中即可，够短再说时长）
_STORY_WORDS = (
    "故事", "童话", "成语", "寓言", "神话", "睡前", "绘本", "儿童", "宝宝", "亲子",
    "幼儿园", "西游记", "三国", "水浒", "安徒生", "格林", "一千零一夜", "伊索",
    "国学", "唐诗", "弟子规", "三字经", "科普", "恐龙", "十万个为什么", "小学",
)
# 明显不是「整段讲故事」的内容
_NOISE_WORDS = (
    "解说", "鬼畜", "沙雕", "混剪", "速看", "reaction", "预告", "花絮", "带货",
    "直播回放", "课程介绍", "片头曲", "主题曲", "mv", "翻唱", "舞蹈",
)
_NORM_RE = re.compile(r"[\s\-_—·,，。.、!！?？:：;；\"'“”‘’()（）【】\[\]《》<>/|~]+")


class BiliError(Exception):
    """B 站取数失败（风控、空音轨、下载异常都归这里，调用方回退兜底源）。"""


def _norm(text: Any) -> str:
    return _NORM_RE.sub("", str(text or "")).lower()


def _clean(text: Any) -> str:
    """搜索结果是 HTML，带 <em class="keyword"> 高亮标签和实体。"""
    s = str(text or "")
    s = re.sub(r"<[^>]+>", "", s)
    for a, b in (("&amp;", "&"), ("&quot;", '"'), ("&lt;", "<"), ("&gt;", ">"), ("&#39;", "'")):
        s = s.replace(a, b)
    return re.sub(r"\s+", " ", s).strip()


def _duration(value: Any) -> int:
    """搜索结果的 duration 是 "12:34" / "1:02:03" 这种字符串。"""
    if isinstance(value, (int, float)):
        return int(value)
    parts = [p for p in str(value or "").split(":") if p.strip().isdigit()]
    secs = 0
    for p in parts:
        secs = secs * 60 + int(p)
    return secs


def _fake_buvid() -> str:
    """abts 的做法：UUID 大写 + 随机补零 + infoc 后缀，看着像真设备。"""
    raw = "".join(random.choice("0123456789ABCDEF") for _ in range(32))
    tail = "".join(random.choice("0123456789") for _ in range(5))
    return f"{raw[:8]}-{raw[8:12]}-{raw[12:16]}-{raw[16:20]}-{raw[20:]}{tail}infoc"


class BiliClient:
    """最小的匿名 B 站客户端：只做搜索、分P列表、取流、下载。"""

    def __init__(
        self,
        timeout: float = 12.0,
        min_gap_ms: int = 600,
        sessdata: str = "",
        keys_path: Any = None,
    ) -> None:
        self.timeout = float(timeout or 12.0)
        self.min_gap = max(0.0, float(min_gap_ms or 0) / 1000.0)
        self.sessdata = str(sessdata or "").strip()
        self.keys_path = Path(keys_path) if keys_path else None
        self._session: Any = None
        self._cookies: Dict[str, str] = {}
        self._key = ""
        self._ready = False
        self._last_req = 0.0
        self._warm_lock = asyncio.Lock()
        self._url_cache: Dict[str, tuple] = {}  # "bvid_cid" -> (urls, 时间)

    # ---------- 基础 ----------
    async def _http(self) -> Any:
        if self._session is None or getattr(self._session, "closed", True):
            import aiohttp

            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            )
        return self._session

    async def close(self) -> None:
        if self._session is not None and not getattr(self._session, "closed", True):
            try:
                await self._session.close()
            except Exception:  # noqa: BLE001
                pass
        self._session = None

    def _headers(self, referer: str = WEB_HOME, origin: str = "") -> Dict[str, str]:
        headers = {
            "User-Agent": _UA,
            "Referer": referer,
            "Accept": "*/*",
            "Accept-Language": "zh-CN,zh;q=0.9",
        }
        if origin:
            headers["Origin"] = origin
        if self._cookies:
            headers["Cookie"] = "; ".join(f"{k}={v}" for k, v in sorted(self._cookies.items()))
        return headers

    def _merge(self, resp: Any) -> None:
        try:
            for key, morsel in resp.cookies.items():
                value = getattr(morsel, "value", morsel)
                if key and value is not None:
                    self._cookies[str(key)] = str(value)
        except Exception:  # noqa: BLE001
            pass

    async def _throttle(self) -> None:
        if self.min_gap:
            gap = self.min_gap - (time.monotonic() - self._last_req)
            if gap > 0:
                await asyncio.sleep(gap)
        self._last_req = time.monotonic()

    # ---------- WBI 签名 ----------
    def _load_key(self) -> str:
        if not self.keys_path:
            return ""
        try:
            data = json.loads(self.keys_path.read_text("utf-8"))
        except Exception:  # noqa: BLE001
            return ""
        if not isinstance(data, dict):
            return ""
        # 每天换一次，跨天就重取（B 站并不强制，但缓存太久没必要）
        if str(data.get("date") or "") != time.strftime("%Y%m%d"):
            return ""
        return str(data.get("key") or "")

    def _save_key(self, key: str) -> None:
        if not self.keys_path:
            return
        try:
            self.keys_path.parent.mkdir(parents=True, exist_ok=True)
            self.keys_path.write_text(
                json.dumps({"key": key, "date": time.strftime("%Y%m%d")}), "utf-8"
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("写入 wbi key 缓存失败: %s", exc)

    @staticmethod
    def _mixin_key(data: Dict[str, Any]) -> str:
        img = ((data or {}).get("data") or {}).get("wbi_img") or {}

        def stem(url: Any) -> str:
            return str(url or "").rsplit("/", 1)[-1].split(".")[0]

        raw = stem(img.get("img_url")) + stem(img.get("sub_url"))
        if len(raw) < 64:
            return ""
        return "".join(raw[i] for i in _MIXIN_TAB)[:32]

    def _sign(self, params: Dict[str, Any]) -> Dict[str, Any]:
        if not self._key:
            raise BiliError("wbi key 未就绪")
        signed = {k: v for k, v in params.items() if v is not None}
        signed["wts"] = int(time.time())
        parts = []
        for key in sorted(signed):
            value = "".join(c for c in str(signed[key]) if c not in _SIGN_STRIP)
            parts.append(
                urllib.parse.quote(str(key), safe="") + "=" + urllib.parse.quote(value, safe="")
            )
        query = "&".join(parts)
        signed["w_rid"] = hashlib.md5((query + self._key).encode()).hexdigest()
        return signed

    # ---------- 预热 ----------
    async def _ensure(self) -> None:
        if self._key and self._ready:
            return
        async with self._warm_lock:
            if self._key and self._ready:
                return
            await self._warmup()

    async def warmup(self) -> None:
        """预热一次（app 启动时后台调，省掉第一次点歌的 1 秒多）。"""
        try:
            await self._ensure()
        except BiliError as exc:
            log.warning("B 站预热失败: %s", exc)

    async def _warmup(self) -> None:
        session = await self._http()
        # 1) 首页拿 buvid3 / b_nut
        try:
            async with session.get(WEB_HOME, headers=self._headers(), allow_redirects=True) as resp:
                self._merge(resp)
                await resp.read()
        except Exception as exc:  # noqa: BLE001
            log.warning("B 站首页预热失败: %s", exc)
        # 2) 补齐 buvid3/buvid4/b_nut（首页只给 buvid3 和 b_nut）
        if not self._cookies.get("buvid3"):
            self._cookies["buvid3"] = _fake_buvid()
        if not self._cookies.get("buvid4"):
            self._cookies["buvid4"] = _fake_buvid()
        if not self._cookies.get("b_nut"):
            self._cookies["b_nut"] = str(int(time.time()))
        if self.sessdata:
            self._cookies.setdefault("SESSDATA", self.sessdata)
        # 3) 激活 buvid：不做这步 pagelist/view 一律 412
        payload = {
            "3064": 1,
            "39c8": "333.1387.fp.risk",
            "3c43": {"adca": "Linux", "bfe9": _fake_buvid()[:50]},
        }
        try:
            async with session.post(
                API + ACTIVATE,
                data={"payload": json.dumps(payload, separators=(",", ":"))},
                headers=self._headers(origin=WEB_HOME),
            ) as resp:
                self._merge(resp)
                await resp.read()
        except Exception as exc:  # noqa: BLE001
            log.warning("B 站 buvid 激活失败: %s", exc)
        # 4) nav 取 wbi key（未登录返回 code -101，但 wbi_img 照样给）
        key = self._load_key()
        if not key:
            try:
                async with session.get(API + NAV, headers=self._headers()) as resp:
                    self._merge(resp)
                    text = await resp.text()
                key = self._mixin_key(json.loads(text))
                if key:
                    self._save_key(key)
            except Exception as exc:  # noqa: BLE001
                log.warning("B 站 nav 取 wbi key 失败: %s", exc)
        if not key:
            self._ready = False
            raise BiliError("拿不到 wbi key（B 站可能改风控了）")
        self._key = key
        self._ready = True
        log.info("B 站客户端就绪（wbi key %s…）", key[:8])

    async def _reset_identity(self) -> None:
        """命中风控时换一身游客身份再来。"""
        self._cookies.pop("buvid3", None)
        self._cookies.pop("buvid4", None)
        self._ready = False
        self._key = ""
        self._url_cache.clear()

    # ---------- 通用请求 ----------
    async def _get_json(
        self,
        path: str,
        params: Dict[str, Any],
        *,
        use_wbi: bool = False,
        referer: str = WEB_HOME,
        origin: str = "",
        tag: str = "",
    ) -> Dict[str, Any]:
        await self._ensure()
        url = path if path.startswith("http") else API + path
        last = ""
        for attempt in range(3):
            await self._throttle()
            query = self._sign(dict(params)) if use_wbi else dict(params)
            status = 0
            text = ""
            try:
                session = await self._http()
                async with session.get(
                    url, params=query, headers=self._headers(referer, origin)
                ) as resp:
                    status = resp.status
                    self._merge(resp)
                    text = await resp.text()
            except Exception as exc:  # noqa: BLE001
                last = f"网络错误 {exc}"
                log.warning("B 站 %s 请求失败: %s", tag or path, exc)
                await asyncio.sleep(_BACKOFF[min(attempt, len(_BACKOFF) - 1)])
                continue
            if status == 200 and text.lstrip().startswith("{"):
                try:
                    data = json.loads(text)
                except Exception:  # noqa: BLE001
                    data = {}
                code = int(data.get("code") or 0)
                if code in _RISK_CODES:
                    last = f"风控 code={code}"
                    log.warning("B 站 %s 命中风控（code=%s），换身份重试", tag or path, code)
                    await self._reset_identity()
                    await asyncio.sleep(_BACKOFF[min(attempt, len(_BACKOFF) - 1)])
                    continue
                if code != 0:
                    last = f"code={code} {data.get('message') or ''}".strip()
                    # -404 之类的业务错没必要重试
                    raise BiliError(f"{tag or path} 失败: {last}")
                return data
            last = f"HTTP {status}（{len(text)} 字节）"
            log.warning("B 站 %s 返回异常：%s", tag or path, last)
            if status in (412, 403):
                await self._reset_identity()
            await asyncio.sleep(_BACKOFF[min(attempt, len(_BACKOFF) - 1)])
        raise BiliError(f"{tag or path} 失败: {last or '未知错误'}")

    # ---------- 业务接口 ----------
    async def search(self, keyword: str, limit: int = 8) -> List[Dict[str, Any]]:
        """搜视频（合集），只留像「讲故事」的。"""
        keyword = str(keyword or "").strip()
        if not keyword:
            return []
        data = await self._get_json(
            SEARCH,
            {
                "search_type": "video",
                "keyword": keyword,
                "page": 1,
                "page_size": max(10, limit * 2),
                "order": "totalrank",
                "platform": "pc",
                "web_location": 1430654,
            },
            use_wbi=True,
            referer=f"{SEARCH_HOME}/video?keyword={urllib.parse.quote(keyword)}",
            origin=SEARCH_HOME,
            tag="搜索",
        )
        items = ((data.get("data") or {}).get("result") or [])
        want = _norm(keyword)
        out: List[Dict[str, Any]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            bvid = str(item.get("bvid") or "")
            if not bvid:
                continue
            title = _clean(item.get("title"))
            low = title.lower()
            if any(w in low for w in _NOISE_WORDS):
                continue
            secs = _duration(item.get("duration"))
            if any(w in title for w in _STORY_WORDS):
                if secs < 60:
                    continue
            elif secs < 600:  # 标题看不出是故事，就要求足够长（合集通常几十分钟）
                continue
            score = 0.0
            if want and want in _norm(title):
                score += 100.0
            score += min(secs / 60.0, 120.0) * 0.1
            out.append(
                {
                    "bvid": bvid,
                    "title": title,
                    "author": _clean(item.get("author")),
                    "duration": secs,
                    "score": score,
                }
            )
        out.sort(key=lambda x: x["score"], reverse=True)
        return out[:limit]

    async def pages(self, bvid: str) -> List[Dict[str, Any]]:
        """分P列表：分P天然就是单集，正好一篇故事一集。"""
        data = await self._get_json(PAGELIST, {"bvid": bvid}, tag="分P列表")
        items = data.get("data") or []
        out: List[Dict[str, Any]] = []
        for idx, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            cid = item.get("cid")
            if not cid:
                continue
            out.append(
                {
                    "index": idx,
                    "cid": int(cid),
                    "title": _clean(item.get("part")),
                    "duration": int(item.get("duration") or 0),
                }
            )
        return out

    async def audio_urls(self, bvid: str, cid: int) -> List[str]:
        """取 DASH 音频直链（按带宽降序，备用链接跟在后面）。"""
        cache_key = f"{bvid}_{cid}"
        cached = self._url_cache.get(cache_key)
        if cached and time.time() - cached[1] < 1800:
            return list(cached[0])
        data = await self._get_json(
            PLAYURL,
            {
                "bvid": bvid,
                "cid": cid,
                "qn": 80,
                "fnval": 4048,
                "fnver": 0,
                "fourk": 1,
                "try_look": 1,
                "gaia_source": "prefer-ua",
            },
            use_wbi=True,
            tag="取流",
        )
        body = data.get("data") or {}
        dash = body.get("dash") or {}
        tracks = [t for t in (dash.get("audio") or []) if isinstance(t, dict)]
        tracks.sort(key=lambda t: int(t.get("bandwidth") or 0), reverse=True)
        urls: List[str] = []
        for track in tracks:
            for url in [track.get("baseUrl") or track.get("base_url")] + list(
                track.get("backupUrl") or track.get("backup_url") or []
            ):
                if url:
                    urls.append(str(url))
        if not urls:  # 老视频没有 dash，退回 durl（带视频轨，PyAV 只取音轨）
            for seg in body.get("durl") or []:
                for url in [seg.get("url")] + list(seg.get("backup_url") or []):
                    if url:
                        urls.append(str(url))
        if not urls:
            raise BiliError("响应里没有可用音轨")
        self._url_cache[cache_key] = (urls, time.time())
        return list(urls)

    async def _download_one(self, url: str) -> bytes:
        session = await self._http()
        # Referer 必须是 B 站首页本身，否则 CDN 回 503 空响应
        headers = {"User-Agent": _UA, "Referer": WEB_HOME, "Accept": "*/*"}
        async with session.get(url, headers=headers) as resp:
            if resp.status not in (200, 206):
                raise BiliError(f"音频 HTTP {resp.status}")
            ctype = (resp.headers.get("Content-Type") or "").lower()
            if ctype and not any(
                w in ctype for w in ("audio", "video", "octet-stream", "mpegurl")
            ):
                raise BiliError(f"音频类型异常 {ctype}")
            chunks: List[bytes] = []
            total = 0
            async for chunk in resp.content.iter_chunked(256 * 1024):
                total += len(chunk)
                if total > MAX_AUDIO_BYTES:
                    raise BiliError("音频超过大小上限")
                chunks.append(chunk)
        if total < MIN_AUDIO_BYTES:
            raise BiliError(f"音频过小（{total} 字节）")
        return b"".join(chunks)

    async def download(self, bvid: str, cid: int) -> bytes:
        """整段下载一集音频（单集一般 1~4MB，0.3~1 秒）。"""
        last = ""
        for attempt in range(2):
            try:
                urls = await self.audio_urls(bvid, cid)
            except BiliError as exc:
                last = str(exc)
                await asyncio.sleep(_BACKOFF[min(attempt, len(_BACKOFF) - 1)])
                continue
            for url in urls[:4]:
                try:
                    return await self._download_one(url)
                except Exception as exc:  # noqa: BLE001
                    last = f"{type(exc).__name__}: {exc}"
                    log.warning("B 站音频下载失败: %s", last)
            # 链接可能已过期，下一轮重新签名
            self._url_cache.pop(f"{bvid}_{cid}", None)
        raise BiliError(f"音频下载失败: {last or '未知错误'}")


_CLIENT: Optional[BiliClient] = None


def client(**kwargs: Any) -> BiliClient:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = BiliClient(**kwargs)
    return _CLIENT


def reset() -> None:
    global _CLIENT
    _CLIENT = None

