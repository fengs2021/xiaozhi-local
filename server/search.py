"""联网搜索：走 Bing 中国站的 HTML 接口，不需要任何 API key。

给孩子用，所以额外做两层过滤：
1. 结果域名白名单/黑名单（屏蔽成人类、小说站、广告站）；
2. 摘要里出现敏感词直接丢弃该条。
"""
from __future__ import annotations

import asyncio
import html
import logging
import re
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

log = logging.getLogger("search")

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)

# 命中这些域名的结果直接丢弃（面向儿童场景）
BLOCK_HOSTS = (
    "porn", "sex", "adult", "18av", "avbo", "yazhou", "qingse",
    "bet", "casino", "lottery", "caipiao",
    "novel", "booktxt", "biquge", "xbiqu", "69shu",
)

# 摘要含这些词也丢弃
BLOCK_WORDS = (
    "色情", "裸", "性交", "赌博", "博彩", "开奖", "情色", "成人网",
    "代孕", "贷款秒批", "办证", "发票",
)


def _clean(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text or "")
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def _host_of(url: str) -> str:
    try:
        return (urlparse(url).hostname or "").lower()
    except Exception:
        return ""


def _unwrap_bing(url: str) -> str:
    """Bing 有时把结果包成 /ck/a?...&u=a1aHR0cHM... 的跳转链接。"""
    if "bing.com/ck/a" not in url:
        return url
    try:
        qs = parse_qs(urlparse(url).query)
        u = (qs.get("u") or [""])[0]
        if u.startswith("a1"):
            import base64

            raw = u[2:]
            raw += "=" * (-len(raw) % 4)
            return base64.urlsafe_b64decode(raw).decode("utf-8", "ignore")
    except Exception:
        pass
    return url


def _blocked(url: str, snippet: str) -> bool:
    host = _host_of(url)
    if any(b in host for b in BLOCK_HOSTS):
        return True
    blob = (snippet or "").lower()
    return any(w in blob for w in BLOCK_WORDS)


async def bing_search(query: str, limit: int = 5) -> List[Dict[str, str]]:
    """抓 Bing 中国站的搜索结果，返回 [{title,url,snippet}]。"""
    import aiohttp

    query = (query or "").strip()
    if not query:
        return []
    headers = {
        "User-Agent": UA,
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "Accept": "text/html,application/xhtml+xml",
    }
    timeout = aiohttp.ClientTimeout(total=12)
    async with aiohttp.ClientSession(headers=headers, timeout=timeout) as sess:
        async with sess.get(
            "https://cn.bing.com/search",
            params={"q": query, "ensearch": "0", "setlang": "zh-CN"},
        ) as resp:
            if resp.status != 200:
                log.warning("Bing 返回 %s", resp.status)
                return []
            body = await resp.text(errors="ignore")

    blocks = re.findall(r'<li class="b_algo".*?</li>', body, re.S)
    out: List[Dict[str, str]] = []
    seen = set()
    for b in blocks:
        m = re.search(r'<h2[^>]*>\s*<a[^>]*href="([^"]+)"[^>]*>(.*?)</a>', b, re.S)
        if not m:
            continue
        url = _unwrap_bing(html.unescape(m.group(1)))
        title = _clean(m.group(2))
        snip = ""
        ms = re.search(r"<p[^>]*>(.*?)</p>", b, re.S)
        if ms:
            snip = _clean(ms.group(1))
        if not title or not url.startswith("http"):
            continue
        if url in seen or _blocked(url, snip):
            continue
        seen.add(url)
        out.append({"title": title, "url": url, "snippet": snip[:200]})
        if len(out) >= limit:
            break
    return out




# ---------------- AnySearch（免 key，英文效果好） ----------------

SOGOU_URL = "https://www.sogou.com/web"
SO360_URL = "https://www.so.com/s"


async def sogou_search(query: str, limit: int = 5) -> List[Dict[str, str]]:
    """抓搜狗网页结果。

    注意：搜狗对高频请求会返回 5452 字节的 antispider 验证页（含 antispider.min.js、
    verify.css），此时页面里没有 vr-title，解析结果为空。调用方必须容忍返回 []。
    """
    import aiohttp

    query = (query or "").strip()
    if not query:
        return []
    headers = {
        "User-Agent": UA,
        "Accept-Language": "zh-CN,zh;q=0.9",
        "Accept": "text/html,application/xhtml+xml",
        "Referer": "https://www.sogou.com/",
    }
    timeout = aiohttp.ClientTimeout(total=12)
    async with aiohttp.ClientSession(headers=headers, timeout=timeout) as sess:
        async with sess.get(SOGOU_URL, params={"query": query}) as resp:
            if resp.status != 200:
                log.warning("搜狗返回 %s", resp.status)
                return []
            body = await resp.text(errors="ignore")

    if "antispider" in body:
        log.info("搜狗触发反爬，跳过")
        return []

    out: List[Dict[str, str]] = []
    seen = set()
    for m in re.finditer(
        r'<h3[^>]*class="[^"]*vr-title[^"]*"[^>]*>\s*<a[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
        body,
        re.S,
    ):
        href, raw_title = m.group(1), m.group(2)
        title = _clean(raw_title)
        if not title:
            continue
        url = html.unescape(href)
        if url.startswith("/"):
            url = "https://www.sogou.com" + url
        tail = body[m.end(): m.end() + 2000]
        snip = ""
        for pat in (
            r'class="[^"]*space-txt[^"]*"[^>]*>(.*?)</div>',
            r'class="[^"]*text-layout[^"]*"[^>]*>(.*?)</div>',
        ):
            pm = re.search(pat, tail, re.S)
            if pm:
                cand = _clean(pm.group(1))
                if len(cand) >= 12:
                    snip = cand
                    break
        if _blocked(url, snip):
            continue
        if title in seen:
            continue
        seen.add(title)
        out.append({"title": title, "url": url, "snippet": snip[:200]})
        if len(out) >= limit:
            break
    return out


async def so360_search(query: str, limit: int = 5) -> List[Dict[str, str]]:
    """抓 360 搜索（www.so.com）。中文专有名词命中率明显好于 Bing。

    360 的结果链接是 https://www.so.com/link?m=... 跳转形式，这里保留跳转链接
    （模型只会读摘要，不需要真实 URL）。
    """
    import aiohttp

    query = (query or "").strip()
    if not query:
        return []
    headers = {
        "User-Agent": UA,
        "Accept-Language": "zh-CN,zh;q=0.9",
        "Accept": "text/html,application/xhtml+xml",
        "Referer": "https://www.so.com/",
    }
    timeout = aiohttp.ClientTimeout(total=12)
    async with aiohttp.ClientSession(headers=headers, timeout=timeout) as sess:
        async with sess.get(SO360_URL, params={"q": query}) as resp:
            if resp.status != 200:
                log.warning("360 搜索返回 %s", resp.status)
                return []
            body = await resp.text(errors="ignore")

    blocks = re.findall(r'<li class="res-list".*?</li>', body, re.S)
    out: List[Dict[str, str]] = []
    seen = set()
    for b in blocks:
        # 去掉内联 style/script，避免干扰文本提取
        b = re.sub(r"<style.*?</style>", "", b, flags=re.S)
        b = re.sub(r"<script.*?</script>", "", b, flags=re.S)
        m = re.search(r'<a[^>]*href="(https?://[^"]+)"[^>]*>(.*?)</a>', b, re.S)
        if not m:
            continue
        url, raw_title = html.unescape(m.group(1)), m.group(2)
        title = _clean(raw_title)
        if len(title) < 6 or not url.startswith("http"):
            continue
        # 摘要：优先 res-desc，其次 res-comm-con / 段落文本
        snip = ""
        for pat in (
            r'<p[^>]*class="[^"]*res-desc[^"]*"[^>]*>(.*?)</p>',
            r'<div[^>]*class="[^"]*res-rich[^"]*"[^>]*>(.*?)</div>',
            r'<span[^>]*class="[^"]*res-list-summary[^"]*"[^>]*>(.*?)</span>',
        ):
            pm = re.search(pat, b, re.S)
            if pm:
                cand = _clean(pm.group(1))
                if len(cand) >= 15:
                    snip = cand
                    break
        if _blocked(url, snip):
            continue
        if title in seen:
            continue
        seen.add(title)
        out.append({"title": title, "url": url, "snippet": snip[:200]})
        if len(out) >= limit:
            break
    return out

ANYSEARCH_URL = "https://api.anysearch.com/v1/search"


async def anysearch(query: str, limit: int = 5) -> List[Dict[str, str]]:
    """AnySearch 匿名免费档。中文查询质量差，只作英文后备。"""
    import aiohttp

    body = {"query": query, "max_results": max(1, min(limit, 8))}
    timeout = aiohttp.ClientTimeout(total=12)
    async with aiohttp.ClientSession(timeout=timeout) as sess:
        async with sess.post(
            ANYSEARCH_URL,
            json=body,
            headers={"content-type": "application/json"},
        ) as resp:
            if resp.status != 200:
                log.warning("AnySearch 返回 %s", resp.status)
                return []
            data = await resp.json(content_type=None)

    rows = ((data or {}).get("data") or {}).get("results") or []
    out: List[Dict[str, str]] = []
    for r in rows:
        url = str(r.get("url") or "").strip()
        title = str(r.get("title") or "").strip()
        snip = str(r.get("snippet") or r.get("content") or "").strip()
        if not url.startswith("http") or not title:
            continue
        if _blocked(url, snip):
            continue
        out.append({"title": title, "url": url, "snippet": snip[:200]})
        if len(out) >= limit:
            break
    return out


def _has_cjk(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in (text or ""))


async def search_web(query: str, limit: int = 5) -> Dict[str, Any]:
    """多引擎后备：Bing 主（中文好）→ AnySearch 备（英文好）。

    中文查询 Bing 若失败，AnySearch 中文质量差，仍会试一次但不作为首选。
    """
    errors: List[str] = []

    # 中文查询 Bing 效果最好；英文查询 Bing 中国站会返回中文词条，改走 AnySearch
    # 中文优先搜狗（Bing 对中文专有名词泛化严重），英文用 AnySearch，Bing 兜底
    if _has_cjk(query):
        order = ["so360", "bing", "sogou", "anysearch"]
    else:
        order = ["anysearch", "bing", "so360"]

    for engine in order:
        try:
            if engine == "so360":
                items = await so360_search(query, limit)
            elif engine == "sogou":
                items = await sogou_search(query, limit)
            elif engine == "bing":
                items = await bing_search(query, limit)
            else:
                items = await anysearch(query, limit)
            if items:
                return {"ok": True, "error": "", "items": items, "engine": engine}
        except asyncio.TimeoutError:
            errors.append(f"{engine} 超时")
        except Exception as exc:
            log.warning("%s 搜索失败: %s", engine, exc)
            errors.append(f"{engine}: {exc}")

    detail = "；".join(errors) if errors else "无结果"
    return {"ok": False, "error": f"搜索暂时不可用（{detail}）", "items": [], "engine": ""}
