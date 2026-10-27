"""中国天气数据源：中央气象台 nmc.cn 公开 REST 接口。

为什么不用 Open-Meteo：
- Open-Meteo 是欧洲模型，对中国县级市覆盖差，中文描述还得靠 WMO 码翻译；
- nmc.cn 是中央气象台官方数据，中文原生，覆盖到县级市（2522 个站点）。

接口（免 key、免代理、直连）：
- GET /rest/province                    省级列表
- GET /rest/province/{code}             该省全部市/县站点
- GET /rest/weather?stationid={code}    实况 + 7 天预报
- GET /rest/findAlarm?pageNo=1&pageSize=N  全国气象预警（不能带中文 province 参数，会 417）

坑：9999 / "-" 是"无数据"哨兵值，必须过滤。
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("weather_cn")

BASE = "http://www.nmc.cn/rest"
TIMEOUT = 8
# 索引缓存有效期（秒）：站点表变化很慢，一天刷一次足够
INDEX_TTL = 86400

# 无数据哨兵
_SENTINELS = {"9999", "9999.0", "-", "", "None", "null"}


def _clean(value: Any) -> Optional[Any]:
    """过滤 nmc 的哨兵值。"""
    if value is None:
        return None
    if isinstance(value, str):
        v = value.strip()
        if v in _SENTINELS:
            return None
        return v
    if isinstance(value, float) and value == 9999.0:
        return None
    if isinstance(value, int) and value == 9999:
        return None
    return value


def _num(value: Any) -> Optional[float]:
    v = _clean(value)
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


class StationIndex:
    """城市名 -> 站点编号 的索引，落盘缓存避免每次启动都抓 34 个省。"""

    def __init__(self, cache_path: Optional[Path] = None) -> None:
        self.cache_path = cache_path
        # 城市名 -> (站点code, 省名)
        self.by_city: Dict[str, Tuple[str, str]] = {}
        self.loaded_at: float = 0.0
        self._lock = asyncio.Lock()

    # ---------- 缓存 ----------
    def _load_cache(self) -> bool:
        if not self.cache_path or not self.cache_path.exists():
            return False
        try:
            raw = json.loads(self.cache_path.read_text(encoding="utf-8"))
            by_city = raw.get("by_city") or {}
            loaded_at = float(raw.get("loaded_at") or 0)
            if not by_city or time.time() - loaded_at > INDEX_TTL:
                return False
            self.by_city = {k: (v[0], v[1]) for k, v in by_city.items()}
            self.loaded_at = loaded_at
            return True
        except Exception as exc:
            log.warning("读取站点缓存失败: %s", exc)
            return False

    def _save_cache(self) -> None:
        if not self.cache_path:
            return
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(
                json.dumps(
                    {
                        "loaded_at": self.loaded_at,
                        "by_city": {k: list(v) for k, v in self.by_city.items()},
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
        except Exception as exc:
            log.warning("保存站点缓存失败: %s", exc)

    # ---------- 构建 ----------
    async def ensure(self) -> None:
        """确保索引可用；并发调用只抓一次。"""
        if self.by_city and time.time() - self.loaded_at < INDEX_TTL:
            return
        async with self._lock:
            if self.by_city and time.time() - self.loaded_at < INDEX_TTL:
                return
            if self._load_cache():
                log.info("站点索引来自缓存：%d 个城市", len(self.by_city))
                return
            await self._build()

    async def _build(self) -> None:
        import aiohttp

        t0 = time.time()
        timeout = aiohttp.ClientTimeout(total=TIMEOUT)
        async with aiohttp.ClientSession(timeout=timeout) as sess:
            async with sess.get(f"{BASE}/province") as resp:
                if resp.status != 200:
                    raise RuntimeError(f"省份列表返回 HTTP {resp.status}")
                provs = await resp.json(content_type=None)

            sem = asyncio.Semaphore(8)

            async def load(prov: Dict[str, Any]) -> List[Tuple[str, str, str]]:
                async with sem:
                    try:
                        async with sess.get(f"{BASE}/province/{prov['code']}") as r:
                            if r.status != 200:
                                return []
                            rows = await r.json(content_type=None)
                        return [
                            (str(x.get("city") or "").strip(), str(x.get("code") or ""), prov["name"])
                            for x in rows
                            if x.get("city") and x.get("code")
                        ]
                    except Exception as exc:
                        log.warning("抓取 %s 站点失败: %s", prov.get("name"), exc)
                        return []

            groups = await asyncio.gather(*[load(p) for p in provs])

        by_city: Dict[str, Tuple[str, str]] = {}
        for rows in groups:
            for city, code, prov_name in rows:
                # 同名城市保留第一个（省会是主要城市，接口按省顺序返回）
                by_city.setdefault(city, (code, prov_name))

        self.by_city = by_city
        self.loaded_at = time.time()
        self._save_cache()
        log.info("站点索引已构建：%d 个城市，耗时 %.2fs", len(by_city), time.time() - t0)

    # ---------- 查询 ----------
    def resolve(self, city: str) -> Optional[Tuple[str, str, str]]:
        """城市名 -> (站点code, 站名, 省名)。找不到返回 None。"""
        name = (city or "").strip()
        if not name:
            return None
        # 去掉行政区后缀逐级尝试
        cands = [name]
        for suf in ("特别行政区", "自治区", "自治州", "地区", "新区", "市辖区", "市", "县", "区", "镇", "街道"):
            if name.endswith(suf) and len(name) > len(suf):
                cands.append(name[: -len(suf)])
        for c in cands:
            hit = self.by_city.get(c)
            if hit:
                return (hit[0], c, hit[1])
        # 包含式：用户可能说「浦东新区」「杭州西湖」
        for c in cands:
            if len(c) < 2:
                continue
            best = ""
            for key in self.by_city:
                if key in c or c in key:
                    if len(key) > len(best):
                        best = key
            if best:
                code, prov = self.by_city[best]
                return (code, best, prov)
        return None


# 全局索引（由 app 启动时注入缓存路径）
INDEX = StationIndex()


async def fetch_weather(station_code: str) -> Optional[Dict[str, Any]]:
    """拉实况 + 7 天预报原始 JSON。"""
    import aiohttp

    timeout = aiohttp.ClientTimeout(total=TIMEOUT)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as sess:
            async with sess.get(f"{BASE}/weather?stationid={station_code}") as resp:
                if resp.status != 200:
                    log.warning("nmc 天气返回 HTTP %s", resp.status)
                    return None
                return await resp.json(content_type=None)
    except Exception as exc:
        log.warning("nmc 天气请求失败: %s", exc)
        return None


def _detail_days(raw: Dict[str, Any]) -> List[Dict[str, Any]]:
    return (((raw or {}).get("data") or {}).get("predict") or {}).get("detail") or []


def _day_brief(row: Dict[str, Any], period: str = "day") -> str:
    """把一天的预报压成一句适合朗读的话。"""
    seg = row.get(period) or {}
    wea = seg.get("weather") or {}
    wind = seg.get("wind") or {}
    info = _clean(wea.get("info"))
    temp = _num(wea.get("temperature"))
    parts: List[str] = []
    if info:
        parts.append(str(info))
    if temp is not None:
        parts.append(f"{temp:.0f} 度")
    direct = _clean(wind.get("direct"))
    power = _clean(wind.get("power"))
    if direct and power:
        parts.append(f"{direct}{power}")
    elif direct:
        parts.append(str(direct))
    return "，".join(parts)


def describe_now(raw: Dict[str, Any], city: str) -> str:
    """实况描述。"""
    real = ((raw or {}).get("data") or {}).get("real") or {}
    wea = real.get("weather") or {}
    wind = real.get("wind") or {}

    temp = _num(wea.get("temperature"))
    feel = _num(wea.get("feelst"))
    humidity = _num(wea.get("humidity"))
    info = _clean(wea.get("info"))

    if temp is None and info is None:
        return f"暂时拿不到{city}的实况数据，请稍后再试。"

    parts = [f"{city}现在"]
    if info:
        parts[0] += str(info)
    if temp is not None:
        parts.append(f"气温 {temp:.1f} 度")
    if feel is not None and abs(feel - (temp or feel)) >= 2:
        parts.append(f"体感 {feel:.1f} 度")
    if humidity is not None:
        parts.append(f"湿度 {humidity:.0f}%")
    direct = _clean(wind.get("direct"))
    power = _clean(wind.get("power"))
    if direct:
        parts.append(f"{direct}{power or ''}".strip())

    text = "，".join(parts) + "。"
    warn = describe_warn(real, city)
    if warn:
        text += warn
    return text


def describe_warn(real: Dict[str, Any], city: str) -> str:
    """当前站点是否有生效预警。"""
    warn = (real or {}).get("warn") or {}
    alert = _clean(warn.get("alert"))
    if not alert:
        return ""
    signaltype = _clean(warn.get("signaltype"))
    signallevel = _clean(warn.get("signallevel"))
    if signaltype and signallevel:
        return f"另外{city}正在发布{signaltype}{signallevel}预警信号，请注意安全。"
    return f"另外{city}有气象预警生效，请注意安全。"


def describe_forecast(raw: Dict[str, Any], city: str, index: int, label: str) -> str:
    """第 index 天的预报（today=0 / tomorrow=1 / day_after=2）。"""
    days = _detail_days(raw)
    if index >= len(days):
        return f"{city}的{label}预报暂时没有数据，请稍后再试。"
    row = days[index]
    date = str(row.get("date") or "")[5:]
    day = _day_brief(row, "day")
    night = _day_brief(row, "night")
    text = f"{city}{label}（{date}）"
    if day:
        text += f"白天{day}"
    if night:
        text += f"，夜间{night}"
    rain = _num(row.get("precipitation"))
    if rain is not None and rain > 0:
        text += f"，降水量约 {rain:.1f} 毫米"
    return text + "。"


def describe_week(raw: Dict[str, Any], city: str, limit: int = 5) -> str:
    """未来几天逐日一句话，控制长度避免语音太长。"""
    days = _detail_days(raw)
    if not days:
        return f"{city}的未来预报暂时没有数据，请稍后再试。"
    out: List[str] = []
    for row in days[:limit]:
        date = str(row.get("date") or "")[5:]
        day = _day_brief(row, "day")
        night_wea = (row.get("night") or {}).get("weather") or {}
        night_temp = _num(night_wea.get("temperature"))
        one = f"{date} {day}" if day else f"{date}"
        if night_temp is not None:
            one += f"，夜间 {night_temp:.0f} 度"
        out.append(one)
    return f"{city}未来几天天气：" + "；".join(out) + "。"


async def find_alarms(keyword: str = "", limit: int = 5) -> List[Dict[str, str]]:
    """查全国气象预警，可按城市/省名过滤。

    注意：province 参数传中文会返回 417，所以拉全国列表后本地过滤。
    """
    import aiohttp

    timeout = aiohttp.ClientTimeout(total=TIMEOUT)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as sess:
            async with sess.get(
                f"{BASE}/findAlarm?pageNo=1&pageSize=100"
            ) as resp:
                if resp.status != 200:
                    return []
                data = await resp.json(content_type=None)
    except Exception as exc:
        log.warning("nmc 预警请求失败: %s", exc)
        return []

    rows = ((data or {}).get("data") or {}).get("page") or {}
    items = rows.get("list") or []
    out: List[Dict[str, str]] = []
    key = (keyword or "").strip()
    for x in items:
        title = str(x.get("title") or "").strip()
        if not title:
            continue
        if key and key not in title:
            continue
        out.append(
            {
                "title": title,
                "time": str(x.get("issuetime") or "").strip(),
                "content": str(x.get("issuecontent") or "").strip(),
            }
        )
        if len(out) >= limit:
            break
    return out
