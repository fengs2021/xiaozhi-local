"""在线讲故事音频：B 站为主源，喜马拉雅播客 RSS 兜底。

story_teller 原来的三级是「bedbook 文本库 → 现编」。这里插进来一层音频：
库里有就照旧念文本，库里没有就去网上找一集真人讲的音频直接播。

B 站这条链见 server/bili.py，比 RSS 强在音质（132K）和分P天然就是单集，
代价是爬的是 UGC 内容、风控随时会收紧，所以：

- 下载下来的整集缓存在 data/story_audio/cache/，同一集重播不再请求 B 站；
- B 站整条链失败才退到喜马拉雅官方播客 RSS（音质 48K，只在 Apple 登记过的专辑有）；
- 都失败返回 None，让 tools.story_teller 退回「现编一个」。

一集正文动辄 5~12 分钟，所以播完默认自动接下一集（max_episodes 控制上限），
孩子说「接下来」也能续上。
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
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("story_audio")

ITUNES = "https://itunes.apple.com/search"
ITUNES_NS = "{http://www.itunes.com/dtds/podcast-1.0.dtd}duration"
_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    " (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)
MIN_AUDIO_BYTES = 120 * 1024
MAX_AUDIO_BYTES = 40 * 1024 * 1024
# 单集时长偏好：太短是片花，超过 15 分钟孩子听不完也占时间
MIN_EPISODE_SECONDS = 60
MAX_EPISODE_SECONDS = 900
_NORM_RE = re.compile(r"[\s\-_—·,，。.、!！?？:：;；\"'“”‘’()（）【】\[\]《》<>/|~]+")


def _norm(text: Any) -> str:
    return _NORM_RE.sub("", str(text or "")).lower()


def _clean(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


# 分P 标题里这些是「感谢观看」「下集预告」之类的垃圾页，播出来孩子只会听到一句废话
_BAD_PAGE_WORDS = (
    "感谢观看", "感谢收看", "关注不迷路", "点赞", "投币", "收藏转发", "下集预告",
    "预告", "片头", "片尾", "主题曲", "花絮", "彩蛋", "广告", "直播", "唠嗑",
    "闲聊", "版权", "合集介绍", "写在最后", "未完待续",
)


def _page_title_ok(title: Any) -> bool:
    """分P 标题能不能当篇名用（有的是 1638937199128.mp4 这种文件名）。"""
    text = str(title or "").strip()
    if not text:
        return False
    if any(w in text for w in _BAD_PAGE_WORDS):
        return False
    if re.fullmatch(r"[\d\W_]+", text):  # 纯数字或符号
        return False
    if re.search(r"\.(mp4|flv|wmv|avi|mkv|mov)$", text, re.I):
        return False
    return True


def _secs(value: Any) -> int:
    """"04:59" / "299" / "1:02:03" 都吃。"""
    s = str(value or "").strip()
    if not s:
        return 0
    if s.isdigit():
        return int(s)
    total = 0
    for part in s.split(":"):
        part = part.strip()
        if part.isdigit():
            total = total * 60 + int(part)
    return total


class StoryAudio:
    """按关键词找一集故事音频，下载并缓存。"""

    def __init__(self, root: Any, cfg: Dict[str, Any]) -> None:
        self.root = Path(root)
        self.enabled = (cfg.get("enabled", True) is not False)
        providers = str(cfg.get("providers") or "bili,ximalaya")
        self.providers = [p.strip() for p in providers.split(",") if p.strip()]
        self.max_episodes = max(1, int(cfg.get("max_episodes") or 2))
        # 讲故事时是否优先上网找真人音频（true = 先找网上，找不到才用本地故事库文本）
        self.prefer_online = cfg.get("prefer_online", True) is not False
        self.timeout = float(cfg.get("timeout") or 12.0)
        self.min_gap_ms = int(cfg.get("bili_min_gap_ms") or 600)
        self.sessdata = str(cfg.get("sessdata") or "")
        self.cache_dir = self.root / "data" / "story_audio" / "cache"
        self.keys_path = self.root / "data" / "bili_keys.json"
        self.cache_max_bytes = int(float(cfg.get("cache_mb") or 400) * 1024 * 1024)
        self._bili: Any = None
        self._session: Any = None
        self._pages: Dict[str, Tuple[List[Dict[str, Any]], float]] = {}
        self._feeds: Dict[str, Tuple[List[Dict[str, Any]], float]] = {}
        self._last: Optional[Dict[str, Any]] = None

    # ---------- 基础 ----------
    def bili(self) -> Any:
        if self._bili is None:
            from .bili import BiliClient

            self._bili = BiliClient(
                timeout=self.timeout,
                min_gap_ms=self.min_gap_ms,
                sessdata=self.sessdata,
                keys_path=self.keys_path,
            )
        return self._bili

    async def _http(self) -> Any:
        if self._session is None or getattr(self._session, "closed", True):
            import aiohttp

            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout),
                headers={"User-Agent": _UA},
            )
        return self._session

    async def close(self) -> None:
        if self._bili is not None:
            try:
                await self._bili.close()
            except Exception:  # noqa: BLE001
                pass
            self._bili = None
        if self._session is not None and not getattr(self._session, "closed", True):
            try:
                await self._session.close()
            except Exception:  # noqa: BLE001
                pass
        self._session = None

    async def warmup(self) -> None:
        if not self.enabled or "bili" not in self.providers:
            return
        try:
            await self.bili().warmup()
        except Exception as exc:  # noqa: BLE001
            log.warning("故事音频预热失败: %s", exc)

    # ---------- 找集 ----------
    async def find(self, keyword: str) -> Optional[Dict[str, Any]]:
        """按关键词找一集：B 站优先，失败退喜马拉雅 RSS。"""
        keyword = _clean(keyword)
        if not keyword or not self.enabled:
            return None
        if "bili" in self.providers:
            plan = await self._find_bili(keyword)
            if plan:
                self._last = plan
                return plan
        if "ximalaya" in self.providers:
            plan = await self._find_ximalaya(keyword)
            if plan:
                self._last = plan
                return plan
        return None

    async def continue_last(self) -> Optional[Dict[str, Any]]:
        """「接下来」「再讲一个」：接着上次那套合集播下一集。"""
        plan = self._last
        if not plan:
            return None
        return await self.next_episode(plan)

    async def _bili_pages(self, bvid: str) -> List[Dict[str, Any]]:
        cached = self._pages.get(bvid)
        if cached and time.time() - cached[1] < 3600:
            return cached[0]
        pages = await self.bili().pages(bvid)
        self._pages[bvid] = (pages, time.time())
        return pages

    def _pick_page(
        self, pages: List[Dict[str, Any]], want: str
    ) -> Optional[Dict[str, Any]]:
        in_range = [
            p
            for p in pages
            if MIN_EPISODE_SECONDS <= int(p.get("duration") or 0) <= MAX_EPISODE_SECONDS
        ]
        good = [p for p in in_range if _page_title_ok(p.get("title"))]
        if not good:
            # 分P标题全是文件名/废话页时，退回只看时长（篇名后面用合集名兜底）
            good = in_range
        if not good:
            return pages[0] if pages else None
        for page in good:  # 分P标题里直接点了名（「第3集 三只小猪」）
            if want and want in _norm(page.get("title")):
                return page
        # 3~8 分钟最适合睡前听；否则在前几集里随机挑一个，连听不会总是第一集
        sweet = [p for p in good if 120 <= int(p.get("duration") or 0) <= 480]
        return random.choice((sweet or good)[:8])

    async def _find_bili(self, keyword: str) -> Optional[Dict[str, Any]]:
        from .bili import BiliError

        try:
            hits = await self.bili().search(keyword, limit=6)
        except BiliError as exc:
            log.warning("B 站搜索失败: %s", exc)
            return None
        if not hits:
            log.info("B 站没搜到「%s」", keyword)
            return None
        want = _norm(keyword)
        for hit in hits:
            try:
                pages = await self._bili_pages(hit["bvid"])
            except BiliError as exc:
                log.warning("B 站分P列表失败: %s", exc)
                continue
            page = self._pick_page(pages, want)
            if page is None:
                continue
            total = len(pages)
            episode = _clean(page.get("title"))
            if not _page_title_ok(episode):
                episode = "%s 第 %d 集" % (hit["title"], int(page["index"]) + 1)
            return {
                "source": "bili",
                "bvid": hit["bvid"],
                "cid": page["cid"],
                "collection": hit["title"],
                "episode": episode,
                "index": int(page["index"]),
                "total": total,
                "duration": int(page.get("duration") or 0),
                "author": hit.get("author") or "",
            }
        return None

    async def _find_ximalaya(self, keyword: str) -> Optional[Dict[str, Any]]:
        try:
            albums = await self._itunes(keyword)
        except Exception as exc:  # noqa: BLE001
            log.warning("iTunes 搜索失败: %s", exc)
            return None
        for album in albums:
            try:
                items = await self._feed(album["feed"])
            except Exception as exc:  # noqa: BLE001
                log.warning("播客 RSS 读取失败: %s", exc)
                continue
            if not items:
                continue
            want = _norm(keyword)
            good = [
                it
                for it in items
                if MIN_EPISODE_SECONDS <= it["duration"] <= MAX_EPISODE_SECONDS
            ] or items
            picked = good[0]
            for it in good:
                if want and want in _norm(it["title"]):
                    picked = it
                    break
            return {
                "source": "ximalaya",
                "feed": album["feed"],
                "collection": album["name"],
                "episode": picked["title"] or album["name"],
                "audio": picked["url"],
                "index": items.index(picked),
                "total": len(items),
                "duration": picked["duration"],
                "author": album.get("artist") or "",
            }
        return None

    async def _itunes(self, keyword: str) -> List[Dict[str, Any]]:
        session = await self._http()
        params = {
            "term": keyword,
            "country": "cn",
            "media": "podcast",
            "entity": "podcast",
            "limit": 8,
        }
        async with session.get(ITUNES, params=params) as resp:
            data = await resp.json(content_type=None)
        want = _norm(keyword)
        out: List[Dict[str, Any]] = []
        for item in (data or {}).get("results") or []:
            feed = str(item.get("feedUrl") or "")
            if not feed:
                continue
            name = _clean(item.get("collectionName"))
            score = 0.0
            if want and want in _norm(name):
                score += 100.0
            count = int(item.get("trackCount") or 0)
            if count <= 1:  # 只在 Apple 登记了 1 集（RSS 残缺）
                score -= 40.0
            score += min(count, 300) * 0.05
            out.append(
                {
                    "name": name,
                    "feed": feed,
                    "artist": _clean(item.get("artistName")),
                    "count": count,
                    "score": score,
                }
            )
        out.sort(key=lambda x: x["score"], reverse=True)
        return out

    async def _feed(self, url: str) -> List[Dict[str, Any]]:
        cached = self._feeds.get(url)
        if cached and time.time() - cached[1] < 3600:
            return cached[0]
        session = await self._http()
        async with session.get(url) as resp:
            text = await resp.text(errors="ignore")
        root = ET.fromstring(text)
        items: List[Dict[str, Any]] = []
        for node in root.iter("item"):
            enc = node.find("enclosure")
            audio = (enc.get("url") if enc is not None else "") or ""
            if not audio:
                continue
            items.append(
                {
                    "title": _clean(node.findtext("title")),
                    "duration": _secs(node.findtext(ITUNES_NS)),
                    "url": audio.strip(),
                }
            )
        self._feeds[url] = (items, time.time())
        return items

    async def next_episode(self, plan: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """同一套合集的下一集；没有就返回 None。"""
        if not plan:
            return None
        try:
            if plan.get("source") == "bili":
                pages = await self._bili_pages(plan["bvid"])
                idx = int(plan.get("index") or 0) + 1
                for page in pages[idx: idx + 8]:
                    if MIN_EPISODE_SECONDS <= int(page.get("duration") or 0) <= MAX_EPISODE_SECONDS:
                        nxt = dict(plan)
                        nxt.update(
                            {
                                "cid": page["cid"],
                                "episode": page["title"] or plan.get("collection") or "",
                                "index": int(page["index"]),
                                "duration": int(page.get("duration") or 0),
                            }
                        )
                        return nxt
                return None
            if plan.get("source") == "ximalaya":
                items = await self._feed(plan["feed"])
                idx = int(plan.get("index") or 0) + 1
                for offset, item in enumerate(items[idx: idx + 8]):
                    if MIN_EPISODE_SECONDS <= item["duration"] <= MAX_EPISODE_SECONDS:
                        nxt = dict(plan)
                        nxt.update(
                            {
                                "audio": item["url"],
                                "episode": item["title"] or plan.get("collection") or "",
                                "index": idx + offset,
                                "duration": item["duration"],
                            }
                        )
                        return nxt
        except Exception as exc:  # noqa: BLE001
            log.warning("取下一集失败: %s", exc)
        return None

    # ---------- 下载与缓存 ----------
    def _cache_key(self, plan: Dict[str, Any]) -> str:
        if plan.get("source") == "bili":
            return f"bili_{plan['bvid']}_{plan['cid']}"
        raw = str(plan.get("audio") or plan.get("feed") or plan.get("episode") or "")
        return "xmp_" + hashlib.md5(raw.encode()).hexdigest()[:16]

    def _cache_path(self, plan: Dict[str, Any]) -> Path:
        return self.cache_dir / f"{self._cache_key(plan)}.audio"

    async def fetch(self, plan: Optional[Dict[str, Any]]) -> bytes:
        """取一集的音频字节（命中缓存直接返回）。"""
        if not plan:
            return b""
        self._last = plan
        path = self._cache_path(plan)
        try:
            if path.exists():
                data = path.read_bytes()
                if len(data) >= MIN_AUDIO_BYTES:
                    return data
        except Exception as exc:  # noqa: BLE001
            log.warning("读缓存失败: %s", exc)
        data = await self._download(plan)
        if len(data) < MIN_AUDIO_BYTES:
            return b""
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            self._trim()
        except Exception as exc:  # noqa: BLE001
            log.warning("写缓存失败: %s", exc)
        return data

    async def _download(self, plan: Dict[str, Any]) -> bytes:
        source = plan.get("source")
        if source == "bili":
            from .bili import BiliError

            try:
                return await self.bili().download(plan["bvid"], plan["cid"])
            except BiliError as exc:
                log.warning("B 站音频下载失败: %s", exc)
                return b""
        if source == "ximalaya":
            try:
                session = await self._http()
                async with session.get(plan["audio"]) as resp:
                    if resp.status not in (200, 206):
                        log.warning("喜马拉雅音频 HTTP %s", resp.status)
                        return b""
                    data = await resp.read()
                if len(data) > MAX_AUDIO_BYTES:
                    log.warning("喜马拉雅音频过大，丢掉")
                    return b""
                return data
            except Exception as exc:  # noqa: BLE001
                log.warning("喜马拉雅音频下载失败: %s", exc)
                return b""
        return b""

    def _trim(self) -> None:
        """按 mtime 淘汰，别把磁盘塞满（一集 1~4MB）。"""
        try:
            files = [p for p in self.cache_dir.glob("*.audio") if p.is_file()]
        except Exception:  # noqa: BLE001
            return
        total = sum(p.stat().st_size for p in files)
        if total <= self.cache_max_bytes:
            return
        files.sort(key=lambda p: p.stat().st_mtime)
        for path in files:
            if total <= self.cache_max_bytes:
                break
            try:
                total -= path.stat().st_size
                path.unlink()
            except Exception:  # noqa: BLE001
                continue


_CLIENT: Optional[StoryAudio] = None


def plan_meta(plan: Dict[str, Any]) -> Dict[str, Any]:
    """把播放计划整理成「这一轮讲的是哪个故事」的展示信息。

    控制中心的历史对话要把来源和链接显出来，但这些字段不该让前端去猜
    （B 站的分P序号从 0 开始、喜马拉雅给的是 RSS 地址），统一在这里算好。
    """
    plan = dict(plan or {})
    source = str(plan.get("source") or "").strip()
    meta: Dict[str, Any] = {
        "source": source,
        "label": "",
        "collection": str(plan.get("collection") or "").strip(),
        "episode": str(plan.get("episode") or "").strip(),
        "index": int(plan.get("index") or 0) + 1,
        "total": int(plan.get("total") or 0),
        "author": str(plan.get("author") or "").strip(),
        "duration": int(plan.get("duration") or 0),
        "url": "",
    }
    bvid = str(plan.get("bvid") or "").strip()
    if source == "bili":
        meta["label"] = "B站"
        if bvid:
            meta["url"] = "https://www.bilibili.com/video/%s?p=%d" % (bvid, meta["index"])
    elif source == "ximalaya":
        meta["label"] = "喜马拉雅"
        found = re.search(r"/album/(\d+)", str(plan.get("feed") or ""))
        if found:
            meta["url"] = "https://www.ximalaya.com/album/%s" % found.group(1)
        if not meta["url"]:
            meta["url"] = str(plan.get("audio") or "")
    elif source:
        meta["label"] = source
    return meta


def _cfg() -> Dict[str, Any]:
    from .config import load_config

    try:
        cfg = load_config()
    except Exception as exc:  # noqa: BLE001
        log.warning("读取配置失败：%s", exc)
        return {}
    section = cfg.get("story_audio") or {}
    return section if isinstance(section, dict) else {}


def client(root: Any = None) -> StoryAudio:
    """进程内单例（B 站 session / wbi key / 缓存都只建一次）。"""
    global _CLIENT
    if _CLIENT is None:
        base = Path(root) if root else Path(__file__).resolve().parents[1]
        _CLIENT = StoryAudio(base, _cfg())
    return _CLIENT


def reset() -> None:
    global _CLIENT
    _CLIENT = None

