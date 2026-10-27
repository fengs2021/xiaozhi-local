"""内置 MCP 工具：不依赖任何外部 key，设备端可直接调用。

所有工具都返回 {"text": "...", "is_error": bool} 的中间结构，
由 session 层再包成 MCP 规范的 content 数组。
"""
from __future__ import annotations

import ast
import contextvars
import logging
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

log = logging.getLogger("tools")

# 当前工具调用上下文（由 llm.set_tool_context 注入 / call_tool 从 args["_ctx"] 载入），
# 用于取设备号、定时任务服务等会话信息。
# 用 ContextVar 而不是模块级全局变量：多台设备并发调工具时，各自的上下文互不覆盖。
_TOOL_CTX: contextvars.ContextVar = contextvars.ContextVar("tool_ctx", default={})


def _tool_ctx() -> Dict[str, Any]:
    """取当前任务绑定的工具上下文；没有就是空字典。"""
    return _TOOL_CTX.get() or {}


def set_tool_context(ctx: Dict[str, Any]) -> None:
    """由 LLM 客户端在每次对话前注入会话上下文。"""
    _TOOL_CTX.set(ctx or {})

# 中国标准时间，固定 +08:00（不依赖系统 tzdata）
CST = timezone(timedelta(hours=8))

# 定时任务：复用 schedule.py 里的时间描述，避免两边说法不一致
from .schedule import describe_repeat, fmt_time  # noqa: E402

# 常用城市的经纬度表（天气工具用），默认上海
CITY_COORDS: Dict[str, Dict[str, Any]] = {
    "北京": {"lat": 39.9042, "lon": 116.4074},
    "上海": {"lat": 31.2304, "lon": 121.4737},
    "广州": {"lat": 23.1291, "lon": 113.2644},
    "深圳": {"lat": 22.5431, "lon": 114.0579},
    "杭州": {"lat": 30.2741, "lon": 120.1551},
    "成都": {"lat": 30.5728, "lon": 104.0668},
    "武汉": {"lat": 30.5928, "lon": 114.3055},
    "西安": {"lat": 34.3416, "lon": 108.9398},
    "南京": {"lat": 32.0603, "lon": 118.7969},
    "重庆": {"lat": 29.5630, "lon": 106.5516},
    "合肥": {"lat": 31.8206, "lon": 117.2272},
    "天津": {"lat": 39.3434, "lon": 117.3616},
    "苏州": {"lat": 31.2989, "lon": 120.5853},
    "无锡": {"lat": 31.4912, "lon": 120.3119},
    "宁波": {"lat": 29.8683, "lon": 121.5440},
    "青岛": {"lat": 36.0671, "lon": 120.3826},
    "济南": {"lat": 36.6512, "lon": 117.1201},
    "郑州": {"lat": 34.7466, "lon": 113.6254},
    "长沙": {"lat": 28.2282, "lon": 112.9388},
    "福州": {"lat": 26.0745, "lon": 119.2965},
    "厦门": {"lat": 24.4798, "lon": 118.0894},
    "昆明": {"lat": 25.0389, "lon": 102.7183},
    "贵阳": {"lat": 26.6470, "lon": 106.6302},
    "南宁": {"lat": 22.8170, "lon": 108.3665},
    "南昌": {"lat": 28.6820, "lon": 115.8579},
    "太原": {"lat": 37.8706, "lon": 112.5489},
    "石家庄": {"lat": 38.0428, "lon": 114.5149},
    "沈阳": {"lat": 41.8057, "lon": 123.4315},
    "大连": {"lat": 38.9140, "lon": 121.6147},
    "长春": {"lat": 43.8171, "lon": 125.3235},
    "哈尔滨": {"lat": 45.8038, "lon": 126.5349},
    "兰州": {"lat": 36.0611, "lon": 103.8343},
    "银川": {"lat": 38.4872, "lon": 106.2309},
    "西宁": {"lat": 36.6171, "lon": 101.7782},
    "乌鲁木齐": {"lat": 43.8256, "lon": 87.6168},
    "拉萨": {"lat": 29.6520, "lon": 91.1721},
    "呼和浩特": {"lat": 40.8424, "lon": 111.7490},
    "海口": {"lat": 20.0444, "lon": 110.1999},
}

DEFAULT_CITY = "上海"

# WMO weather_code -> 中文描述
WEATHER_CODES: Dict[int, str] = {
    0: "晴",
    1: "大部晴朗",
    2: "多云",
    3: "阴",
    45: "有雾",
    48: "雾凇",
    51: "小毛毛雨",
    53: "毛毛雨",
    55: "大毛毛雨",
    56: "冻毛毛雨",
    57: "强冻毛毛雨",
    61: "小雨",
    63: "中雨",
    65: "大雨",
    66: "冻雨",
    67: "强冻雨",
    71: "小雪",
    73: "中雪",
    75: "大雪",
    77: "雪粒",
    80: "小阵雨",
    81: "阵雨",
    82: "强阵雨",
    85: "小阵雪",
    86: "大阵雪",
    95: "雷阵雨",
    96: "雷阵雨伴小冰雹",
    99: "雷阵雨伴大冰雹",
}

# 星期中文名（datetime.weekday(): 0=周一）
_WEEKDAYS = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]

TOOLS: List[Dict[str, Any]] = [
    {
        "name": "get_time",
        "description": "查询当前本地日期和时间（中国标准时间）。用户问现在几点、今天几号、今天星期几时使用。",
        "inputSchema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_weather",
        "description": (
            "查询天气。默认查当前天气；"
            "用户问明天/后天/未来几天/周末时，"
            "必须传 when 参数取预报，不要拿当前天气当预报。"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "城市名，例如：北京、上海、广州、深圳、杭州、成都、武汉、西安、南京、重庆",
                },
                "when": {
                    "type": "string",
                    "description": (
                        "查询时间。today=今天，tomorrow=明天，day_after=后天，"
                        "week=未来一周。不填或 now 表示当前天气。"
                    ),
                },
            },
            "required": [],
        },
    },
    {
        "name": "get_weather_alarm",
        "description": (
            "查询气象预警（暴雨、台风、大风、高温、寒潮等）。"
            "用户问有没有预警、台风来了吗、要不要注意天气时使用。"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "城市或省份名，例如：上海、浙江、广东。不填则查全国最新预警",
                }
            },
            "required": [],
        },
    },
    {
        "name": "calculator",
        "description": "计算四则运算表达式，支持加减乘除和括号。例如 (12+8)*3/4。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "要计算的算式，只能包含数字和 + - * / ( ) . 空格"}
            },
            "required": ["expression"],
        },
    },
    {
        "name": "get_device_status",
        "description": "查询当前连接的小智设备状态，包括设备号、连接时长、在线状态和收发数据包数量。",
        "inputSchema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "idiom_chain",
        "description": "成语接龙。用户说出一个成语时，返回以该成语最后一个字开头的成语；不给成语时随机给一个开头。适合陪小学生练习成语。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "idiom": {
                    "type": "string",
                    "description": "上一个成语，例如 一心一意。留空则随机开始",
                }
            },
            "required": [],
        },
    },
    {
        "name": "web_search",
        "description": "联网搜索。当孩子问的知识超出你的记忆、需要最新信息（新闻、天气以外的实时数据、某个人物/事件的近期情况）时使用。返回标题和摘要，你要用自己的话把结果讲给孩子听，不要念网址。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索关键词，用简洁的中文短语，例如 恐龙为什么灭绝、嫦娥六号最新消息",
                },
                "limit": {
                    "type": "integer",
                    "description": "返回几条结果，默认 4，最多 8",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "pinyin_lookup",
        "description": "查汉字的拼音和组词。孩子问某个字怎么读、有哪些词语时使用。支持一次查多个字。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "要查的字或词，例如 长、美丽、春天",
                }
            },
            "required": ["text"],
        },
    },
    {
        "name": "math_practice",
        "description": "口算练习。出题、判对错、看答案都用它。action 取值：new 出新题，check 检查答案，hint 给提示。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "new 出新题 / check 检查上一题的答案 / hint 提示上一题怎么算",
                    "enum": ["new", "check", "hint"],
                },
                "level": {
                    "type": "integer",
                    "description": "难度：1=20以内加减，2=100以内加减，3=乘法口诀，4=两位数乘一位数或除法。默认 1",
                },
                "answer": {
                    "type": "string",
                    "description": "action=check 时孩子给的答案，例如 21",
                },
            },
            "required": ["action"],
        },
    },
    {
        "name": "english_word",
        "description": "小学英语单词卡。查单词的中文意思和音标，或者随机来一张单词卡练习。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "英文单词或中文意思，例如 apple、苹果。留空则随机给一个",
                }
            },
            "required": [],
        },
    },
    {
        "name": "story_teller",
        "description": "讲一个故事（寓言、成语故事、民间故事），带一句道理。孩子说想听故事、讲个故事时使用。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "想听的故事名，例如 龟兔赛跑。留空则随机讲一个",
                }
            },
            "required": [],
        },
    },
    {
        "name": "recite_poem",
        "description": "背诵古诗词。用户说诗名、作者或诗句片段时，返回对应诗词的完整原文与简单释义。适合小学生背诗。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "诗名、作者或诗句片段，例如 静夜思、李白、床前明月光",
                }
            },
            "required": ["query"],
        },
    },
    {
        "name": "set_reminder",
        "description": (
            "设置提醒或闹钟，到点小智会自己开口说话。"
            "「5 分钟后提醒我喝水」用 in_seconds=300；"
            "「半小时后叫我」用 in_seconds=1800；"
            "用户只说「倒计时 5 分钟」没说提醒内容时，也要调本工具：in_seconds=300、text 写「时间到啦」；"
            "秒数可以带小数（90 秒、1 分半都换算成秒）。"
            "「每天早晨 7 点喊我起床」用 at_clock=07:00 加 repeat=daily；"
            "「明天早上 7 点叫我」用 at_clock=07:00 加 day=tomorrow。"
            "text 要写成到点要说出口的那句话，例如 该喝水啦、起床啦该上学了。"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "到点要说的话，例如 该喝水啦、起床啦",
                },
                "in_seconds": {
                    "type": "integer",
                    "description": "多少秒之后提醒。用于「5 分钟后」「半小时后」这类相对时间",
                },
                "at_clock": {
                    "type": "string",
                    "description": "几点几分，24 小时制 HH:MM，例如 07:00、15:30。用于「7 点」「下午 3 点半」这类绝对时间",
                },
                "day": {
                    "type": "string",
                    "description": "配合 at_clock：today=今天（默认，这个点已过就算明天）、tomorrow=明天",
                },
                "repeat": {
                    "type": "string",
                    "description": "once=只提醒一次（默认）；daily=每天；weekdays=周一到周五；weekly=每周同一天",
                },
            },
            "required": ["text"],
        },
    },
    {
        "name": "list_reminders",
        "description": "查看已经设好的提醒和闹钟。用户问「我设了哪些提醒」「闹钟还在不在」时使用。",
        "inputSchema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "cancel_reminder",
        "description": (
            "取消提醒或闹钟。用户说「取消提醒」「别叫我起床了」时使用。"
            "不传 id 和 keyword 时取消最近设置的那一条。"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "id": {
                    "type": "string",
                    "description": "任务 id，从 list_reminders 得到。不填则按关键词或最近一条处理",
                },
                "keyword": {
                    "type": "string",
                    "description": "按内容关键词匹配，例如 喝水",
                },
            },
            "required": [],
        },
    },
    {
        "name": "play_music",
        "description": (
            "放一首歌。用户说「放首歌」「我想听稻香」「来点周杰伦」时使用。"
            "keyword 用歌名或歌手，例如 稻香、周杰伦 晴天、小星星。"
            "放起来后会一直播，直到用户说话或让停。"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "keyword": {
                    "type": "string",
                    "description": "歌名或歌手，例如 稻香、周杰伦、孤勇者",
                }
            },
            "required": ["keyword"],
        },
    },
    {
        "name": "stop_music",
        "description": "停止正在播放的音乐。用户说「别放了」「停」「关掉音乐」时使用。",
        "inputSchema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "control_music",
        "description": (
            "控制正在播放的音乐：下一首 / 上一首 / 暂停 / 继续 / 停止 / 换播放模式。"
            "用户说「下一首」「换一首」「上一首」「先暂停」「接着放」「单曲循环」"
            "「随机播放」「列表循环」「放完就停」时使用。只说「停止」用 stop_music 也行。"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["next", "prev", "pause", "resume", "stop", "mode"],
                    "description": "要做的操作",
                },
                "mode": {
                    "type": "string",
                    "enum": ["off", "one", "all", "shuffle"],
                    "description": (
                        "action=mode 时必填。off=顺序放完就停，one=单曲循环，"
                        "all=列表循环（默认），shuffle=随机播放"
                    ),
                },
            },
            "required": ["action"],
        },
    },
]

TOOL_NAMES = [t["name"] for t in TOOLS]


def to_openai_tools(names: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """把内置工具转成 OpenAI function calling 需要的格式。

    names 为空则全部导出；MCP 的 inputSchema 直接就是 JSON Schema，
    只需换成 {"type":"function","function":{...}} 外壳。
    """
    wanted = set(names) if names else set(TOOL_NAMES)
    out: List[Dict[str, Any]] = []
    for t in TOOLS:
        if t["name"] not in wanted:
            continue
        out.append(
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t.get("inputSchema") or {
                        "type": "object",
                        "properties": {},
                        "required": [],
                    },
                },
            }
        )
    return out


def _ok(text: str, direct: bool = False) -> Dict[str, Any]:
    """构造成功结果。

    direct=True 表示这段文本本身就是给用户听的话（例如报时、算式结果），
    可以跳过「把工具结果再喂回模型润色」的那一轮，直接播报。对应上游
    xiaozhi-esp32-server 里 Action.RESPONSE 的语义。
    """
    item: Dict[str, Any] = {"text": text, "is_error": False}
    if direct:
        item["direct"] = True
    return item


def _err(text: str) -> Dict[str, Any]:
    return {"text": text, "is_error": True}


# ---------- get_time ----------
async def get_time(args: Dict[str, Any]) -> Dict[str, Any]:
    """返回中国标准时间的自然语言描述。"""
    now = datetime.now(CST)
    text = f"现在是 {now.year}年{now.month}月{now.day}日 {_WEEKDAYS[now.weekday()]} {now.hour:02d}:{now.minute:02d}"
    return _ok(text, direct=True)


# ---------- get_weather ----------
def _weather_desc(code: Any) -> str:
    try:
        return WEATHER_CODES.get(int(code), "未知天气")
    except (TypeError, ValueError):
        return "未知天气"


_WHEN_ALIASES = {
    "now": "now", "": "now", "现在": "now", "当前": "now", "此刻": "now",
    "today": "today", "今天": "today", "今日": "today",
    "tomorrow": "tomorrow", "明天": "tomorrow", "明日": "tomorrow",
    "day_after": "day_after", "后天": "day_after", "后日": "day_after",
    "week": "week", "一周": "week", "未来一周": "week", "这几天": "week",
    "未来": "week", "周末": "week",
}


def _norm_when(raw: Any) -> str:
    """把各种写法归一成 now/today/tomorrow/day_after/week。"""
    key = str(raw or "").strip().lower()
    if not key:
        return "now"
    if key in _WHEN_ALIASES:
        return _WHEN_ALIASES[key]
    # 包含式匹配：「明天上午」之类也能中
    for word, norm in (
        ("后天", "day_after"), ("明天", "tomorrow"), ("明日", "tomorrow"),
        ("今天", "today"), ("今日", "today"), ("一周", "week"),
        ("周末", "week"), ("未来", "week"),
    ):
        if word in key:
            return norm
    if "tomorrow" in key:
        return "tomorrow"
    if "today" in key:
        return "today"
    return "now"


async def get_weather(args: Dict[str, Any]) -> Dict[str, Any]:
    """查天气：中国城市走中央气象台（覆盖 2522 个站点），海外回落 Open-Meteo。"""
    city = str((args or {}).get("city") or "").strip() or DEFAULT_CITY
    when = _norm_when((args or {}).get("when"))

    # 优先中国气象台数据源：中文原生、覆盖到县级市
    cn = await _weather_cn(city, when)
    if cn is not None:
        return cn

    return await _weather_openmeteo(city, when)


async def _weather_cn(city: str, when: str) -> Optional[Dict[str, Any]]:
    """中央气象台通道。城市解析不到或请求失败时返回 None 交给上层回落。"""
    from . import weather_cn

    try:
        await weather_cn.INDEX.ensure()
    except Exception as exc:
        log.warning("气象台站点索引不可用: %s", exc)
        return None

    hit = weather_cn.INDEX.resolve(city)
    if not hit:
        return None
    code, name, prov = hit

    raw = await weather_cn.fetch_weather(code)
    if not raw:
        return None
    if not weather_cn._detail_days(raw) and not ((raw.get("data") or {}).get("real")):
        return None

    if when == "now":
        return _ok(weather_cn.describe_now(raw, name))
    if when == "week":
        return _ok(weather_cn.describe_week(raw, name, 5))
    label = {"today": "今天", "tomorrow": "明天", "day_after": "后天"}.get(when, "今天")
    idx = {"today": 0, "tomorrow": 1, "day_after": 2}.get(when, 0)
    return _ok(weather_cn.describe_forecast(raw, name, idx, label))


async def _weather_openmeteo(city: str, when: str) -> Dict[str, Any]:
    """Open-Meteo 通道：海外城市或气象台不可用时的兜底。"""
    import aiohttp

    # 允许传「上海市」「上海 浦东」这类写法，取前两个字做模糊匹配
    matched = ""
    for name in CITY_COORDS:
        if name in city:
            matched = name
            break
    if not matched:
        # 没收录的城市回退到默认城市，比直接报错更有用
        matched = DEFAULT_CITY if DEFAULT_CITY in CITY_COORDS else next(iter(CITY_COORDS))
        city = f"{city}（暂未收录，先看{matched}）"

    coord = CITY_COORDS[matched]
    if when == "now":
        url = (
            "https://api.open-meteo.com/v1/forecast"
            f"?latitude={coord['lat']}&longitude={coord['lon']}"
            "&current=temperature_2m,relative_humidity_2m,weather_code"
            "&timezone=Asia%2FShanghai"
        )
    else:
        url = (
            "https://api.open-meteo.com/v1/forecast"
            f"?latitude={coord['lat']}&longitude={coord['lon']}"
            "&daily=weather_code,temperature_2m_max,temperature_2m_min,"
            "precipitation_probability_max,wind_speed_10m_max"
            "&forecast_days=7&timezone=Asia%2FShanghai"
        )
    try:
        timeout = aiohttp.ClientTimeout(total=5)
        async with aiohttp.ClientSession(timeout=timeout) as sess:
            async with sess.get(url) as resp:
                if resp.status != 200:
                    return _err(f"天气服务返回异常（HTTP {resp.status}），请稍后再试。")
                data = await resp.json()
    except Exception as exc:
        log.warning("查询天气失败: %s", exc)
        return _err("天气服务暂时连不上，请稍后再试。")

    if when == "now":
        cur = (data or {}).get("current") or {}
        temp = cur.get("temperature_2m")
        humidity = cur.get("relative_humidity_2m")
        desc = _weather_desc(cur.get("weather_code"))
        if temp is None:
            return _err("天气服务没有返回有效数据，请稍后再试。")
        parts = [f"{matched}现在{desc}，气温 {temp} 摄氏度"]
        if humidity is not None:
            parts.append(f"湿度 {humidity}%")
        return _ok("，".join(parts) + "。")

    daily = (data or {}).get("daily") or {}
    dates = daily.get("time") or []
    codes = daily.get("weather_code") or []
    tmax = daily.get("temperature_2m_max") or []
    tmin = daily.get("temperature_2m_min") or []
    rains = daily.get("precipitation_probability_max") or []
    winds = daily.get("wind_speed_10m_max") or []
    if not dates:
        return _err("天气服务没有返回预报数据，请稍后再试。")

    labels = {"today": 0, "tomorrow": 1, "day_after": 2}
    if when in labels:
        i = labels[when]
        if i >= len(dates):
            return _err("预报数据不足，请稍后再试。")
        desc = _weather_desc(codes[i] if i < len(codes) else None)
        hi = tmax[i] if i < len(tmax) else None
        lo = tmin[i] if i < len(tmin) else None
        rain = rains[i] if i < len(rains) else None
        wind = winds[i] if i < len(winds) else None
        name = {"today": "今天", "tomorrow": "明天", "day_after": "后天"}[when]
        seg = [f"{matched}{name}（{dates[i]}）{desc}"]
        if hi is not None and lo is not None:
            seg.append(f"{lo} 到 {hi} 摄氏度")
        if rain is not None:
            seg.append(f"降水概率 {rain}%")
        if wind is not None:
            seg.append(f"最大风速 {wind} 公里每小时")
        return _ok("，".join(seg) + "。")

    # week：未来几天逐日列出，最多 5 天，避免语音太长
    out = []
    for i in range(min(5, len(dates))):
        desc = _weather_desc(codes[i] if i < len(codes) else None)
        hi = tmax[i] if i < len(tmax) else None
        lo = tmin[i] if i < len(tmin) else None
        rain = rains[i] if i < len(rains) else None
        day = f"{dates[i][5:]}"
        one = f"{day} {desc}"
        if hi is not None and lo is not None:
            one += f" {lo}~{hi} 度"
        if rain is not None:
            one += f" 降水 {rain}%"
        out.append(one)
    return _ok(f"{matched}未来几天天气：" + "；".join(out) + "。")


# ---------- calculator ----------
_ALLOWED_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Constant,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.USub,
    ast.UAdd,
    ast.Load,
)


def safe_eval(expression: str) -> float:
    """只允许数字与四则运算的表达式求值，禁止任何函数调用/名字访问。"""
    expr = (expression or "").strip()
    if not expr:
        raise ValueError("算式不能为空")
    if len(expr) > 200:
        raise ValueError("算式太长了")
    allowed_chars = set("0123456789+-*/(). %")
    bad = [c for c in expr if c not in allowed_chars]
    if bad:
        raise ValueError("算式里有不支持的字符")
    node = ast.parse(expr, mode="eval")
    for sub in ast.walk(node):
        if not isinstance(sub, _ALLOWED_NODES):
            raise ValueError("算式里有不支持的运算")
        # 常量只能是数字，排除 True/False/字符串
        if isinstance(sub, ast.Constant) and not isinstance(sub.value, (int, float)):
            raise ValueError("算式里只能出现数字")
    value = eval(compile(node, "<calc>", "eval"), {"__builtins__": {}}, {})  # noqa: S307
    if isinstance(value, complex):
        raise ValueError("结果不是实数")
    return float(value)


async def calculator(args: Dict[str, Any]) -> Dict[str, Any]:
    """安全四则运算。"""
    expr = str((args or {}).get("expression") or "").strip()
    try:
        value = safe_eval(expr)
    except ZeroDivisionError:
        return _err("除数不能为零。")
    except Exception as exc:
        return _err(f"算不出来：{exc}。")
    # 整数结果不要带 .0
    if abs(value - round(value)) < 1e-9:
        text = str(int(round(value)))
    else:
        text = f"{value:.6f}".rstrip("0").rstrip(".")
    return _ok(f"{expr} = {text}", direct=True)


# ---------- get_device_status ----------
def _fmt_duration(seconds: float) -> str:
    """把秒数说成人话。"""
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds} 秒"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes} 分 {sec} 秒"
    hours, minutes = divmod(minutes, 60)
    return f"{hours} 小时 {minutes} 分"


async def get_device_status(args: Dict[str, Any]) -> Dict[str, Any]:
    """从 Hub 取当前设备信息（hub/session 由 call_tool 注入到 args 的 _ctx）。"""
    ctx = (args or {}).get("_ctx") or {}
    hub = ctx.get("hub")
    session = ctx.get("session")
    if hub is None:
        return _err("设备状态暂时读不到。")

    sessions = list(getattr(hub, "sessions", {}).values())
    if not sessions:
        return _ok("现在没有设备连着服务器。")

    target = session or sessions[0]
    info = target.info() if hasattr(target, "info") else {}
    device_id = info.get("device_id") or "未知设备"
    state_map = {"idle": "空闲", "listening": "正在听", "speaking": "正在说话"}
    state = state_map.get(info.get("state", ""), info.get("state") or "未知")
    text = (
        f"设备 {device_id} 已连接 {_fmt_duration(info.get('connected_seconds', 0))}，"
        f"当前状态{state}，"
        f"共收 {info.get('uplink_packets', 0)} 个音频包、发 {info.get('downlink_packets', 0)} 个，"
        f"对话 {info.get('utterances', 0)} 轮。"
    )
    return _ok(text)


# ---------- idiom_chain ----------
async def idiom_chain(args: Dict[str, Any]) -> Dict[str, Any]:
    """成语接龙：给出以上一个成语尾字开头的成语。"""
    from . import kid_content as kc

    idiom = str((args or {}).get("idiom") or "").strip()
    if not idiom:
        start = kc.random_idiom()
        return _ok(
            f"那我们从「{start}」开始吧！请你接一个以「{start[-1]}」字开头的成语。"
        )

    nxt = kc.find_next_idioms(idiom, limit=3)
    if not nxt:
        return _ok(
            f"「{idiom}」这个词库里暂时接不下去啦，换一个成语试试？"
            f"比如「{kc.random_idiom()}」。"
        )
    seg = f"「{idiom}」的最后一个字是「{idiom[-1]}」，可以接：" + "、".join(nxt) + "。"
    meaning = kc.idiom_meaning(nxt[0])
    if meaning:
        seg += f" 比如「{nxt[0]}」的意思是：{meaning}"
    seg += " 你选一个试试？"
    return _ok(seg, direct=True)


# ---------- recite_poem ----------
async def recite_poem(args: Dict[str, Any]) -> Dict[str, Any]:
    """背诵古诗：按诗名/作者/诗句片段返回原文与释义。"""
    from . import kid_content as kc

    query = str((args or {}).get("query") or "").strip()
    if not query:
        return _err("你想听哪首诗呀？可以说诗名、作者，或者记得的一句诗。")

    hits = kc.search_poems(query, limit=2)
    if not hits:
        sample = "、".join(p["title"] for p in kc.POEMS[:6])
        return _ok(
            f"没有找到和「{query}」有关的诗。我现在会背的有：{sample} 等等，"
            "你可以说诗名或者作者名字。"
        )
    return _ok(" ".join(kc.poem_text(p) for p in hits), direct=True)



# ---------- web_search ----------
async def web_search(args: Dict[str, Any]) -> Dict[str, Any]:
    """联网搜索，返回可读的结果摘要。"""
    from .search import search_web

    a = args or {}
    query = str(a.get("query") or "").strip()
    if not query:
        return _err("你想搜什么呢？说个关键词就行。")
    try:
        limit = int(a.get("limit") or 4)
    except (TypeError, ValueError):
        limit = 4
    limit = max(1, min(limit, 8))

    res = await search_web(query, limit)
    if not res.get("ok"):
        return _err(res.get("error") or "搜索失败了。")
    items = res.get("items") or []
    if not items:
        return _ok(f"没有搜到关于「{query}」的内容，换个说法试试？")

    lines = [f"关于「{query}」搜到 {len(items)} 条："]
    for i, it in enumerate(items, 1):
        snip = it.get("snippet") or ""
        lines.append(f"{i}. {it['title']}" + (f"：{snip}" if snip else ""))
    lines.append("（请用自己的话讲给孩子听，不要说网址。）")
    return _ok("\n".join(lines))


# ---------- pinyin_lookup ----------
async def pinyin_lookup(args: Dict[str, Any]) -> Dict[str, Any]:
    """查字的拼音、组词与多音字提示。"""
    from . import kid_tools as kt

    text = str((args or {}).get("text") or "").strip()
    if not text:
        return _err("你想查哪个字呀？说一个汉字就行。")
    text = text[:8]
    parts: List[str] = []
    known = 0
    for ch, py in kt.pinyin_of(text):
        if not py:
            continue
        info = kt.lookup_char(ch)
        known += 1
        seg = f"「{ch}」读 {py}"
        if len(info["readings"]) > 1:
            seg += "（也读 " + "、".join(r for r in info["readings"] if r != py) + "）"
        if info["words"]:
            seg += "，可以组词：" + "、".join(info["words"][:4])
        parts.append(seg)
        if info["hint"]:
            parts.append(f"  小提示：{info['hint']}")
    if not parts:
        return _ok(f"「{text}」里没找到汉字，换个字试试？")
    return _ok("\n".join(parts))


# ---------- math_practice ----------
def _math_key(args: Dict[str, Any]) -> str:
    """用设备号/会话号做练习状态 key，缺省用 default。"""
    ctx = _tool_ctx()
    sess = ctx.get("session")
    dev = ""
    try:
        dev = getattr(getattr(sess, "stats", None), "device_id", "") or ""
    except Exception:
        dev = ""
    return dev or "default"


async def math_practice(args: Dict[str, Any]) -> Dict[str, Any]:
    """口算练习：出题 / 判对错 / 提示。"""
    from . import kid_tools as kt

    a = args or {}
    action = str(a.get("action") or "new").strip().lower()
    key = _math_key(a)

    if action == "new":
        try:
            level = int(a.get("level") or 1)
        except (TypeError, ValueError):
            level = 1
        q = kt.new_math_question(key, level)
        names = {1: "20以内加减", 2: "100以内加减", 3: "乘法口诀", 4: "两位数乘一位数"}
        return _ok(f"来一道{names.get(q['level'], '口算')}题：{q['question']} = ?")

    if action == "check":
        res = kt.check_math_answer(key, str(a.get("answer") or ""))
        if not res.get("has_question"):
            return _ok("现在还没有题目哦，先说「出一道题」吧。")
        if res.get("correct") is None:
            return _ok(f"你的答案是「{a.get('answer')}」，我看不太懂，直接说一个数字就行。")
        if res["correct"]:
            return _ok(f"答对啦！{res['question']} = {res['expected']}，真棒！要不要再来一道？")
        return _ok(
            f"再想想哦，{res['question']} 的答案不是 {a.get('answer')}。"
            "要不要我给你一点提示？"
        )

    if action == "hint":
        st = kt._math_state.get(key)
        if not st:
            return _ok("现在还没有题目哦，先说「出一道题」吧。")
        q = st["q"]
        if "×" in q:
            x, y = q.split("×")
            return _ok(
                f"提示：{x.strip()} × {y.strip()} 就是 {y.strip()} 个 {x.strip()} 相加，"
                "想想乘法口诀表。"
            )
        if "÷" in q:
            x, y = q.split("÷")
            return _ok(f"提示：想一想 {y.strip()} 乘几等于 {x.strip()}，那个数就是答案。")
        if "+" in q:
            x, y = q.split("+")
            return _ok(f"提示：先凑整十，{x.strip()} 加上 {y.strip()}，可以拆成整十加零头。")
        x, y = q.split("-")
        return _ok(f"提示：从 {x.strip()} 里去掉 {y.strip()}，可以先用整十去减，再加回来。")

    return _err(f"不认识的动作：{action}，只能是 new / check / hint。")


# ---------- english_word ----------
async def english_word(args: Dict[str, Any]) -> Dict[str, Any]:
    """小学英语单词卡。"""
    from . import kid_tools as kt

    query = str((args or {}).get("query") or "").strip()
    if not query:
        w = kt.random_word()
        return _ok(
            f"单词卡：{w['word']} {w['phonetic']}，意思是「{w['cn']}」，{w['grade']} 词汇。"
            "跟我读一遍试试？"
        )
    w = kt.find_word(query)
    if not w:
        sample = "、".join(x["word"] for x in kt.ENGLISH_WORDS[:8])
        return _ok(
            f"词库里没有「{query}」。我这儿有 {len(kt.ENGLISH_WORDS)} 个小学单词，"
            f"比如 {sample} 等等，你可以说英文或中文。"
        )
    return _ok(
        f"{w['word']} {w['phonetic']} 意思是「{w['cn']}」，{w['grade']} 词汇。"
        f"比如可以说：I like {w['word']}."
    )


# ---------- story_teller ----------
async def story_teller(args: Dict[str, Any]) -> Dict[str, Any]:
    """讲一个故事。"""
    from . import kid_tools as kt

    title = str((args or {}).get("title") or "").strip()
    if title:
        s = kt.find_story(title)
        if not s:
            sample = "、".join(x["title"] for x in kt.STORIES[:6])
            return _ok(f"没有找到《{title}》。我会讲的有：{sample} 等等，要不要随便来一个？")
    else:
        s = kt.random_story()

    text = (s.get("text") or "").strip()
    # 格林童话有的一万多字，语音场景必须截断
    if len(text) > 700:
        cut = max(text.rfind("。", 0, 700), text.rfind("！", 0, 700), text.rfind("？", 0, 700))
        text = text[: cut + 1] if cut > 300 else text[:700] + "……"
    seg = f"《{s['title']}》\n{text}"
    moral = (s.get("moral") or "").strip()
    if moral:
        seg += f"\n这个故事告诉我们：{moral}"
    return _ok(seg)

async def get_weather_alarm(args: Dict[str, Any]) -> Dict[str, Any]:
    """查气象预警。city 为空则给全国最新几条。"""
    from . import weather_cn

    city = str((args or {}).get("city") or "").strip()
    try:
        alarms = await weather_cn.find_alarms(city, 5)
    except Exception as exc:
        log.warning("预警查询失败: %s", exc)
        return _err("预警服务暂时不可用，稍后再试。")

    if not alarms:
        if city:
            return _ok(f"{city}目前没有生效的气象预警。")
        return _ok("全国目前没有生效的气象预警。")

    lines: List[str] = []
    if city:
        lines.append(f"{city}相关的气象预警有 {len(alarms)} 条：")
    else:
        lines.append(f"全国最新的气象预警有 {len(alarms)} 条：")
    for a in alarms:
        title = a.get("title") or ""
        when = (a.get("time") or "")[:16]
        content = (a.get("content") or "").strip()
        # 语音场景下正文不能太长，截到 60 字
        if len(content) > 60:
            content = content[:60] + "……"
        lines.append(f"{when}，{title}。{content}")
    return _ok("\n".join(lines))


# ---------- 定时任务（提醒 / 闹钟） ----------
def _schedule_service():
    """取定时任务服务；没装配时返回 None。"""
    return _tool_ctx().get("schedule")


def _parse_clock(raw: str) -> tuple:
    """解析时间说法，返回 (时, 分)。

    支持 07:00 / 7:00 / 7点 / 7点半 / 7点30 / 下午3点 / 晚上8点 / 早上7点。
    """
    text = str(raw or "").strip().replace("：", ":").replace(" ", "")
    if not text:
        raise ValueError("没听清是几点")
    lowered = text
    is_pm = any(k in lowered for k in ("下午", "晚上", "傍晚", "夜里"))
    is_midnight = "凌晨" in lowered
    for k in ("上午", "下午", "早上", "早晨", "晚上", "凌晨", "傍晚", "中午", "夜里"):
        text = text.replace(k, "")
    text = text.replace("点半", "点30").replace("时半", "时30")
    m = re.match(r"^(\d{1,2})(?:[:点时](\d{1,2})?分?)?$", text)
    if not m:
        raise ValueError(f"没听清是几点：{raw}")
    hour = int(m.group(1))
    minute = int(m.group(2)) if m.group(2) else 0
    if is_pm and hour < 12:
        hour += 12
    # 「晚上 12 点」「凌晨 12 点」都是午夜 0 点，不是中午
    if hour == 12 and (is_midnight or "晚上" in lowered or "夜里" in lowered):
        hour = 0
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ValueError(f"这个时间不合法：{raw}")
    return hour, minute


def _resolve_fire_at(args: Dict[str, Any]) -> tuple:
    """把工具参数换算成 (fire_at 秒, clock "HH:MM", repeat)。"""
    now = time.time()
    repeat = str(args.get("repeat") or "once").strip().lower()
    if repeat not in ("once", "daily", "weekdays", "weekly"):
        repeat = "once"
    raw_in = args.get("in_seconds")
    if raw_in not in (None, "", 0, "0"):
        try:
            secs = float(raw_in)
        except (TypeError, ValueError):
            raise ValueError("多少秒之后要说个数字")
        if secs <= 0:
            raise ValueError("时间得是将来的")
        if repeat != "once":
            raise ValueError("每天、每周的提醒要用几点几分来定")
        return now + secs, "", repeat
    clock_raw = str(args.get("at_clock") or "").strip()
    if not clock_raw:
        raise ValueError("要么说多少分钟之后，要么说几点")
    hour, minute = _parse_clock(clock_raw)
    clock = f"{hour:02d}:{minute:02d}"
    base = datetime.now(CST)
    if repeat == "weekly":
        try:
            human_wd = int(args.get("weekday") or 0)
        except (TypeError, ValueError):
            human_wd = 0
        if not (1 <= human_wd <= 7):
            raise ValueError("每周的提醒要说是星期几")
        py_wd = human_wd - 1  # Python 里周一=0
        delta = (py_wd - base.weekday()) % 7
        cand = (base + timedelta(days=delta)).replace(
            hour=hour, minute=minute, second=0, microsecond=0
        )
        if cand.timestamp() <= now + 1:
            cand = cand + timedelta(days=7)
        return cand.timestamp(), clock, repeat
    target = base.replace(hour=hour, minute=minute, second=0, microsecond=0)
    tomorrow = str(args.get("day") or "").strip().lower() == "tomorrow"
    if tomorrow or target.timestamp() <= now + 1:
        target = target + timedelta(days=1)
    return target.timestamp(), clock, repeat


def _human_when(fire_at: float, repeat: str, clock: str) -> str:
    """把触发时刻说成人话，用于确认语。"""
    secs = fire_at - time.time()
    if repeat == "once" and 0 < secs <= 3600:
        mins = int(round(secs / 60.0))
        return "1 分钟后" if mins <= 1 else f"{mins} 分钟后"
    try:
        when = fmt_time(fire_at)
    except Exception:  # noqa: BLE001
        when = ""
    if repeat == "daily":
        return f"每天 {clock}（下一次 {when}）"
    if repeat == "weekdays":
        return f"每个工作日 {clock}（下一次 {when}）"
    if repeat == "weekly":
        try:
            wd = "一二三四五六日"[datetime.fromtimestamp(fire_at, CST).weekday()]
        except Exception:  # noqa: BLE001
            wd = ""
        return f"每周{wd} {clock}（下一次 {when}）"
    return when or "到点"


async def set_reminder(args: Dict[str, Any]) -> Dict[str, Any]:
    svc = _schedule_service()
    if svc is None:
        return _err("定时任务还没准备好，等会儿再试。")
    text = str(args.get("text") or "").strip()
    # 纯「倒计时 5 分钟」通常不会带内容：给一句默认话术，别反问打断。
    _secs_hint = args.get("in_seconds")
    if not text and isinstance(_secs_hint, (int, float)) and _secs_hint > 0:
        text = "时间到啦"
    if not text:
        return _err("要提醒你什么事呢？")
    try:
        fire_at, clock, repeat = _resolve_fire_at(args)
    except ValueError as exc:
        return _err(str(exc))
    weekday = None
    if repeat == "weekly":
        weekday = datetime.fromtimestamp(fire_at, CST).weekday()
    device_id = ""
    sess = _tool_ctx().get("session")
    stats = getattr(sess, "stats", None) if sess is not None else None
    if stats is not None:
        device_id = str(getattr(stats, "device_id", "") or "")
    try:
        item = svc.store.create(
            text=text,
            fire_at=fire_at,
            repeat=repeat,
            clock=clock,
            weekday=weekday,
            device_id=device_id,
            kind="alarm" if repeat != "once" else "remind",
            created_by="voice",
        )
    except ValueError as exc:
        return _err(str(exc))
    when = _human_when(fire_at, repeat, clock)
    # 相对时间（「倒计时 5 分钟」）才切屏幕倒数界面；绝对时间（「明早 7 点」）
    # 可能差十几个小时，显示倒数没意义。界面是纯叠加层，唤醒词和对话都不受影响。
    secs = args.get("in_seconds")
    if sess is not None and repeat == "once" and isinstance(secs, (int, float)) and secs > 0:
        try:
            await sess.show_countdown(int(secs), text, str((item or {}).get("id") or ""))
        except Exception as exc:  # noqa: BLE001
            log.debug("下发倒计时界面失败: %s", exc)
    return _ok(f"好，{when}我会提醒你：{text}", direct=True)


async def list_reminders(args: Dict[str, Any]) -> Dict[str, Any]:
    svc = _schedule_service()
    if svc is None:
        return _err("定时任务还没准备好。")
    items = [it for it in svc.store.list_items() if it.get("enabled")]
    if not items:
        return _ok("现在没有设任何提醒。", direct=True)
    items.sort(key=lambda it: float(it.get("fire_at") or 0))
    parts = []
    for it in items[:8]:
        if (it.get("repeat") or "once") == "once":
            when = fmt_time(float(it.get("fire_at") or 0))
        else:
            when = describe_repeat(it)
        parts.append(f"{when}：{it.get('text')}")
    more = f"，另外还有 {len(items) - 8} 条" if len(items) > 8 else ""
    return _ok("你设了这些提醒——" + "；".join(parts) + more, direct=True)


async def cancel_reminder(args: Dict[str, Any]) -> Dict[str, Any]:
    svc = _schedule_service()
    if svc is None:
        return _err("定时任务还没准备好。")
    items = svc.store.list_items()
    tid = str(args.get("id") or "").strip()
    keyword = str(args.get("keyword") or "").strip()
    target = None
    if tid:
        target = svc.store.get(tid)
    elif keyword:
        for it in reversed(items):
            if keyword in str(it.get("text") or ""):
                target = it
                break
    else:
        enabled = [it for it in items if it.get("enabled")]
        if enabled:
            target = max(enabled, key=lambda it: float(it.get("created_at") or 0))
    if target is None:
        return _err("没找到要取消的提醒。")
    said = str(target.get("text") or "")
    try:
        svc.store.delete(str(target.get("id") or ""))
    except KeyError:
        return _err("没找到要取消的提醒。")
    sess = _tool_ctx().get("session")
    if sess is not None:
        try:
            # 只收正在倒数的那一条；取消别的提醒不影响到时界面。
            await sess.hide_countdown(str(target.get("id") or ""))
        except Exception as exc:  # noqa: BLE001
            log.debug("收起倒计时界面失败: %s", exc)
    return _ok(f"好，已取消「{said}」。", direct=True)


async def play_music(args: Dict[str, Any]) -> Dict[str, Any]:
    session = _tool_ctx().get("session")
    player = getattr(session, "music", None)
    if player is None:
        return _err("音乐播放还没准备好。")
    keyword = str(args.get("keyword") or "").strip()
    if not keyword:
        return _err("要放什么歌呢？说个歌名或歌手吧。")
    try:
        prepared = await player.prepare(keyword)
    except Exception as exc:  # noqa: BLE001
        log.warning("搜歌失败: %s", exc)
        return _err("音乐服务好像不太通，等会儿再试试。")
    if not prepared:
        return _ok(f"没找到「{keyword}」，换个说法试试？", direct=True)
    song = prepared.get("song") or {}
    title = str(song.get("title") or keyword)
    artist = str(song.get("artist") or "").strip()
    # 先挂起来：这句「好，给你放……」还要走 TTS 下发，等它播完再真正开始，
    # 否则两路 opus 同时推设备会互相盖掉（见 session._start_pending_music）。
    session.pending_music = prepared
    said = f"好，给你放{title}" + (f"，{artist}的" if artist else "") + "。"
    return _ok(said, direct=True)


async def stop_music(args: Dict[str, Any]) -> Dict[str, Any]:
    session = _tool_ctx().get("session")
    player = getattr(session, "music", None)
    if player is None:
        return _err("音乐播放还没准备好。")
    pending = getattr(session, "pending_music", None)
    playing = bool(getattr(player, "playing", False) or getattr(player, "current", None))
    if not playing and not pending:
        # 用户一开口说话，listen start 就已经把音乐停了（不然盖住人声），
        # 所以这里要认「刚刚才停过」这件事，别答成「现在没在放歌」。
        recent = getattr(session, "_music_recent", None) or {}
        song = recent.get("song") or {}
        # 阈值放宽到 2 分钟：从「开始聆听」到模型真的调工具，中间要等 VAD 收句 + 识别，
        # 孩子说一句话前面往往还先愣几秒，5 秒根本不够。
        fresh = recent.get("at") and (time.time() - float(recent.get("at") or 0)) < 120.0
        if fresh and song.get("title"):
            session._music_recent = None
            return _ok("好，不放了。", direct=True)
        return _ok("现在没在放歌。", direct=True)
    if session is not None:
        session._stop_music("用户停止")
    return _ok("好，不放了。", direct=True)


_MUSIC_MODE_SAY = {
    "off": "顺序播放",
    "one": "单曲循环",
    "all": "列表循环",
    "shuffle": "随机播放",
}


async def control_music(args: Dict[str, Any]) -> Dict[str, Any]:
    session = _tool_ctx().get("session")
    player = getattr(session, "music", None)
    if player is None:
        return _err("音乐播放还没准备好。")
    action = str(args.get("action") or "").strip()
    if action == "stop":
        return await stop_music({})
    if action == "mode":
        mode = str(args.get("mode") or "").strip()
        if not player.set_mode(mode):
            return _err("这个播放模式我不认识。")
        phrase = _MUSIC_MODE_SAY.get(mode) or ""
        return _ok(f"好，{phrase}。", direct=True)
    if action in ("next", "prev"):
        if not player.queue:
            return _err("现在没有播放列表。")
        prepared = (
            await player.next_track() if action == "next" else await player.prev_track()
        )
        if prepared is None:
            said = "这已经是最后一首啦。" if action == "next" else "已经是最前面一首啦。"
            return _ok(said, direct=True)
        # 和点歌同一条路：先说完「好，换一首」，等 TTS 收尾再开播
        session.pending_music = prepared
        return _ok("好，换一首。", direct=True)
    if action == "pause":
        if not player.pause():
            return _ok("现在没在放歌。", direct=True)
        return _ok("好，先暂停。", direct=True)
    if action == "resume":
        if not player.resume():
            return _ok("现在没在放歌。", direct=True)
        return _ok("好，接着放。", direct=True)
    return _err("没听懂要做什么操作。")


_DISPATCH = {
    "get_time": get_time,
    "get_weather": get_weather,
    "get_weather_alarm": get_weather_alarm,
    "calculator": calculator,
    "get_device_status": get_device_status,
    "idiom_chain": idiom_chain,
    "recite_poem": recite_poem,
    "pinyin_lookup": pinyin_lookup,
    "math_practice": math_practice,
    "english_word": english_word,
    "story_teller": story_teller,
    "web_search": web_search,
    "set_reminder": set_reminder,
    "list_reminders": list_reminders,
    "cancel_reminder": cancel_reminder,
    "play_music": play_music,
    "stop_music": stop_music,
    "control_music": control_music,
}


async def call_tool(name: str, args: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """按名字调用工具；未知工具返回 is_error，不抛异常。

    llm 层会把会话上下文放进 args["_ctx"] 带下来（hub / session / schedule），
    这里载入当前任务的 ContextVar，工具内部的 _schedule_service()、_math_key()
    才能拿到设备号和定时任务服务。执行完恢复原值，不污染调用方。
    """
    args = args or {}
    token = None
    ctx = args.get("_ctx")
    if isinstance(ctx, dict):
        token = _TOOL_CTX.set(ctx)
    try:
        func = _DISPATCH.get((name or "").strip())
        if func is None:
            return _err(f"没有名为 {name} 的工具。")
        try:
            return await func(args)
        except Exception as exc:
            log.exception("工具 %s 执行失败", name)
            return _err(f"工具执行出错：{exc}")
    finally:
        if token is not None:
            _TOOL_CTX.reset(token)
