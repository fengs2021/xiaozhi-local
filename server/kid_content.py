"""给小朋友用的本地内容库：成语接龙 + 小学必背古诗。

数据来自 data/ 目录（由 _build_data.py 从公开语料生成）：
  45776 条四字成语（带拼音与释义）
  319 首古诗（title / author / type / lines）
缺文件时自动退回内置的精简常用词表，保证功能不中断。
"""
from __future__ import annotations

import random
from typing import Any, Dict, List

from . import kid_data

# ---------------- 成语 ----------------

# 小学生友好的常用成语（随机出题、示例提示时优先展示，避免冷僻词）
COMMON_IDIOMS: List[str] = [
    "一心一意", "一鸣惊人", "一箭双雕", "一举两得", "一目了然", "一诺千金",
    "一丝不苟", "一帆风顺", "一往无前", "一落千丈", "一衣带水", "一鼓作气",
    "三心二意", "三头六臂", "三思而行", "四面八方", "四通八达", "五光十色",
    "五颜六色", "六神无主", "七上八下", "七嘴八舌", "八面威风", "九牛一毛",
    "十全十美", "百发百中", "百折不挠", "百花齐放", "千军万马", "千方百计",
    "万紫千红", "万众一心", "万里无云", "气象万千", "画龙点睛", "画蛇添足",
    "守株待兔", "亡羊补牢", "掩耳盗铃", "刻舟求剑", "井底之蛙", "对牛弹琴",
    "狐假虎威", "自相矛盾", "滥竽充数", "拔苗助长", "叶公好龙", "杯弓蛇影",
    "惊弓之鸟", "塞翁失马", "愚公移山", "精卫填海", "夸父逐日", "女娲补天",
    "闻鸡起舞", "悬梁刺股", "凿壁偷光", "囊萤映雪", "程门立雪", "卧薪尝胆",
    "日积月累", "水滴石穿", "持之以恒", "坚持不懈", "锲而不舍", "专心致志",
    "全神贯注", "聚精会神", "废寝忘食", "夜以继日", "争分夺秒", "惜时如金",
    "兴高采烈", "眉开眼笑", "喜出望外", "心花怒放", "欢天喜地", "载歌载舞",
    "龙飞凤舞", "生龙活虎", "虎头虎脑", "鸟语花香", "山清水秀", "春暖花开",
    "秋高气爽", "冰天雪地", "烈日炎炎", "风和日丽", "云开雾散", "雨过天晴",
    "五湖四海", "天南地北", "东张西望", "左顾右盼", "南辕北辙", "北雁南飞",
    "名副其实", "实事求是", "脚踏实地", "勤学好问", "不耻下问", "温故知新",
    "举一反三", "触类旁通", "融会贯通", "学以致用", "循序渐进", "熟能生巧",
    "雪中送炭", "锦上添花", "助人为乐", "拾金不昧", "见义勇为", "尊老爱幼",
    "和睦相处", "同心协力", "齐心协力", "团结一致", "众志成城", "同舟共济",
    "光明正大", "表里如一", "言而有信", "诚实守信", "谦虚谨慎",
    "不骄不躁", "知错就改", "勇于认错", "有始有终", "善始善终",
]

_ROWS = kid_data.load_idioms()  # [(word, pinyin, explanation)]

IDIOMS: List[str] = [w for w, _py, _ex in _ROWS] or list(COMMON_IDIOMS)
_IDIOM_MEANING: Dict[str, str] = {w: ex for w, _py, ex in _ROWS if ex}
_IDIOM_PINYIN: Dict[str, str] = {w: py for w, py, _ex in _ROWS if py}
_COMMON_SET = set(COMMON_IDIOMS)

# 首字 -> 成语列表（接龙查表用）
_IDIOM_BY_FIRST: Dict[str, List[str]] = {}
for _it in IDIOMS:
    _IDIOM_BY_FIRST.setdefault(_it[0], []).append(_it)

# 常用词优先排序：常用成语排前面，避免冷僻词
for _ch in _IDIOM_BY_FIRST:
    _IDIOM_BY_FIRST[_ch].sort(key=lambda w: (w not in _COMMON_SET, w))


def find_next_idioms(idiom: str, limit: int = 3) -> List[str]:
    """找出以 idiom 最后一个字开头的成语，常用词优先。"""
    idiom = (idiom or "").strip()
    if len(idiom) < 2:
        return []
    cand = [x for x in _IDIOM_BY_FIRST.get(idiom[-1], []) if x != idiom]
    return cand[:limit]


def has_next(idiom: str) -> bool:
    """判断这个成语在词库里还有没有下家。"""
    idiom = (idiom or "").strip()
    if len(idiom) < 2:
        return False
    return bool([x for x in _IDIOM_BY_FIRST.get(idiom[-1], []) if x != idiom])


def random_idiom() -> str:
    """随机给一个常用成语（保证小学生认识）。"""
    return random.choice(COMMON_IDIOMS)


def random_idioms(n: int = 3) -> List[str]:
    """随机取 n 个不重复的常用成语。"""
    return random.sample(COMMON_IDIOMS, min(n, len(COMMON_IDIOMS)))


def all_start_chars() -> List[str]:
    """词库里所有成语的首字（用于提示可接哪些）。"""
    return sorted(_IDIOM_BY_FIRST.keys())


def idiom_meaning(idiom: str) -> str:
    """查成语释义，没有则返回空串。"""
    return _IDIOM_MEANING.get((idiom or "").strip(), "")


def idiom_pinyin(idiom: str) -> str:
    return _IDIOM_PINYIN.get((idiom or "").strip(), "")


# ---------------- 古诗 ----------------

POEMS: List[Dict[str, Any]] = kid_data.load_poems()

# 每首诗拼成一个大字符串用于检索
_POEM_HAY: List[str] = [
    (p.get("title", "") + p.get("author", "") + p.get("type", "") + "".join(p.get("lines") or []))
    for p in POEMS
]


def search_poems(query: str, limit: int = 3) -> List[Dict[str, Any]]:
    """按标题 / 作者 / 诗句片段模糊匹配古诗。

    优先级：标题完全相同 > 标题包含 > 标题被包含 > 作者 > 诗句片段。
    分档是为了避免「静夜思」被《夜思》这种子串先命中。
    """
    q = (query or "").strip().strip("《》<>\"' ")
    if not q:
        return []

    same: List[Dict[str, Any]] = []
    has: List[Dict[str, Any]] = []
    within: List[Dict[str, Any]] = []
    other: List[Dict[str, Any]] = []

    for p, hay in zip(POEMS, _POEM_HAY):
        title = p.get("title", "")
        author = p.get("author", "")
        if title == q:
            same.append(p)
        elif q in title:
            has.append(p)
        elif len(title) >= 2 and title in q:
            within.append(p)
        elif q in author or q in hay:
            other.append(p)
        elif len(q) >= 2 and any(q[i : i + 2] in hay for i in range(len(q) - 1)):
            other.append(p)
        if len(same) + len(has) >= limit:
            break

    # 标题精确命中就不再带出「被包含」的近似项（如「静夜思」不该再带出《夜思》）
    if same:
        return same[:limit]
    return (same + has + within + other)[:limit]


def poem_text(p: Dict[str, Any]) -> str:
    """把一首诗拼成适合朗读的文本。"""
    head = f"《{p.get('title', '')}》"
    if p.get("author"):
        head += f"{p.get('author')}"
    grade = p.get("grade") or 0
    if grade:
        head += f"，{grade} 年级课本"
    body = "".join(p.get("lines") or [])
    return f"{head}。{body}"


def random_poem() -> Dict[str, Any]:
    return random.choice(POEMS) if POEMS else {}
