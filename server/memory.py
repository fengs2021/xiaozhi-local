"""长期记忆 v2：结构化画像 + 可检索的记忆条目，让助手在长期对话里越来越了解用户。

设计要点：
- 两层结构：profile（画像槽位，覆盖式，最该长期记住的稳定信息）+ items（事实/偏好/经历/约定）
- 检索式注入：每轮按用户当前这句话做相关度打分（关键词 2-gram + 重要度 + 时间衰减 + 命中次数），
  只把相关的少数几条塞进 system message，而不是把所有记忆一股脑灌进去
- 画像永远全量注入（它短且最关键），条目按相关度挑选
- 兼容 v1 的纯列表格式，首次载入自动迁移
"""
from __future__ import annotations

import json
import logging
import math
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

log = logging.getLogger("memory")

# 记忆分类
KINDS = ("kid", "prefer", "fact", "event", "plan")
KIND_LABEL = {
    "kid": "用户情况",
    "prefer": "偏好习惯",
    "fact": "记住的事实",
    "event": "经历见闻",
    "plan": "约定安排",
}

# 画像槽位：长期最该了解用户的字段（key, 展示名）
PROFILE_FIELDS: Tuple[Tuple[str, str], ...] = (
    ("name", "名字/称呼"),
    ("age", "年龄"),
    ("grade", "年级/学段"),
    ("city", "所在城市"),
    ("school", "学校"),
    ("family", "家人"),
    ("likes", "喜欢的事"),
    ("dislikes", "不喜欢的事"),
    ("habits", "生活习惯"),
    ("study", "学习情况"),
    ("goal", "目标心愿"),
    ("personality", "性格特点"),
    ("schedule", "作息安排"),
    ("other", "其他"),
)
PROFILE_LABEL = dict(PROFILE_FIELDS)
PROFILE_ORDER = [k for k, _ in PROFILE_FIELDS]
# 拼成一句话时的顺序（名字/年龄/年级/城市先说，读起来最自然）
PROFILE_INLINE_ORDER = (
    "name", "age", "grade", "city", "school", "family",
    "likes", "dislikes", "habits", "study", "goal", "personality", "schedule", "other",
)

MAX_ITEMS = 400            # 最多保留多少条（超出按重要度+时间淘汰）
PROMPT_MAX_CHARS = 700     # 注入 system message 的字数上限
PROMPT_MAX_ITEMS = 10      # 除画像外最多挑几条

# 默认设置（可在网页上改）
DEFAULT_SETTINGS = {
    "auto": True,          # 是否开启后台自动记忆
    "reflect_every": 10,   # 每多少轮对话做一次「反思整理」，0 表示关闭
    "inject_max_chars": PROMPT_MAX_CHARS,
}

DEFAULT_STATS = {
    "turns": 0,            # 累计处理过多少轮对话
    "extracted": 0,        # 累计抽取落库多少条
    "llm_calls": 0,
    "llm_failures": 0,
    "last_extract_at": 0.0,
    "last_reflect_at": 0.0,
    "last_turn_at": 0.0,
}


def _now() -> float:
    return time.time()


def _norm(text: str) -> str:
    """归一化用于查重：去空白、去标点、转小写。"""
    s = re.sub("[\\s，。！？、；：\u201c\u201d\u2018\u2019（）()\\[\\]【】,.!?;:~·\\-]+", "", str(text or ""))
    return s.lower()


_STOP_GRAMS = set(
    """的 了 是 我 你 他 她 它 在 有 和 就 都 也 很 还 不 吗 呢 吧 啊 呀 哦 嗯 这 那 什么 怎么
    一个 我们 你们 他们 今天 明天 昨天 现在 然后 所以 但是 因为 如果 可以 知道 觉得 就是 这个 那个""".split()
)


def _grams(text: str, n: int = 2) -> set:
    """粗粒度中文分词：按 2-gram 切，够用且不依赖分词库。"""
    s = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9]+", "", str(text or "")).lower()
    if not s:
        return set()
    if len(s) <= n:
        return {s}
    return {s[i : i + n] for i in range(len(s) - n + 1)} - _STOP_GRAMS


class MemoryStore:
    """长期记忆存储：单文件 JSON + 线程锁。"""

    def __init__(self, root: str | Path):
        self.path = Path(root) / "data" / "memory.json"
        self._lock = threading.RLock()
        self.items: List[Dict[str, Any]] = []
        self.profile: Dict[str, Dict[str, Any]] = {}
        self.settings: Dict[str, Any] = dict(DEFAULT_SETTINGS)
        self.stats: Dict[str, Any] = dict(DEFAULT_STATS)
        self.user_hint = ""
        self._load()

    # ---------- 持久化 ----------
    def _load(self) -> None:
        raw: Any = None
        try:
            if self.path.exists():
                raw = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception as exc:
            log.warning("读取长期记忆失败: %s", exc)
            raw = None
        if isinstance(raw, list):
            # v1：纯列表
            self.items = [x for x in raw if isinstance(x, dict) and x.get("text")]
            log.info("已载入 v1 长期记忆 %d 条（已自动升级）", len(self.items))
        elif isinstance(raw, dict):
            self.items = [x for x in (raw.get("items") or []) if isinstance(x, dict) and x.get("text")]
            prof = raw.get("profile")
            if isinstance(prof, dict):
                self.profile = {k: v for k, v in prof.items() if isinstance(v, dict) and v.get("value")}
            st = raw.get("settings")
            if isinstance(st, dict):
                self.settings.update({k: st[k] for k in DEFAULT_SETTINGS if k in st})
            stats = raw.get("stats")
            if isinstance(stats, dict):
                self.stats.update({k: stats[k] for k in DEFAULT_STATS if k in stats})
            log.info("已载入长期记忆 %d 条、画像 %d 项", len(self.items), len(self.profile))
        # 补齐字段默认值
        for it in self.items:
            it.setdefault("kind", "fact")
            it.setdefault("importance", 0.5)
            it.setdefault("hits", 1)
            it.setdefault("slot", "")
            it.setdefault("tags", [])
            it.setdefault("created_at", _now())
            it.setdefault("updated_at", it.get("created_at", _now()))
            it.setdefault("last_used_at", 0.0)
            it.setdefault("source", "auto")

    def _save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "version": 2,
                "updated_at": _now(),
                "settings": self.settings,
                "stats": self.stats,
                "profile": self.profile,
                "items": self.items,
            }
            tmp = self.path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(self.path)
        except Exception as exc:
            log.warning("保存长期记忆失败: %s", exc)

    def save(self) -> None:
        with self._lock:
            self._save()

    # ---------- 画像 ----------
    def set_profile_field(self, field: str, value: str, source: str = "auto") -> bool:
        """写入一个画像槽位；空值表示删除该槽位。"""
        field = str(field or "").strip()
        body = str(value or "").strip()
        if field not in PROFILE_LABEL:
            field = "other"
        with self._lock:
            if not body:
                if field in self.profile:
                    self.profile.pop(field, None)
                    self._save()
                    return True
                return False
            cur = self.profile.get(field) or {}
            if _norm(cur.get("value", "")) == _norm(body):
                cur["updated_at"] = _now()
                self.profile[field] = cur
                self._save()
                return False
            self.profile[field] = {
                "value": body,
                "updated_at": _now(),
                "source": str(source or "auto"),
            }
            self._save()
            log.info("画像[%s] = %s", PROFILE_LABEL.get(field, field), body)
            return True

    def set_profile(self, mapping: Dict[str, Any], source: str = "auto") -> int:
        """批量写画像，返回实际变化条数。"""
        changed = 0
        for k, v in (mapping or {}).items():
            if k not in PROFILE_LABEL:
                continue
            if self.set_profile_field(k, v, source):
                changed += 1
        return changed

    def profile_items(self) -> List[Dict[str, Any]]:
        """给网页用的画像列表（按固定顺序）。"""
        out = []
        with self._lock:
            for key in PROFILE_ORDER:
                cell = self.profile.get(key)
                if not cell:
                    continue
                out.append(
                    {
                        "field": key,
                        "label": PROFILE_LABEL.get(key, key),
                        "value": cell.get("value", ""),
                        "updated_at": cell.get("updated_at", 0),
                        "source": cell.get("source", "auto"),
                    }
                )
        return out

    def profile_line(self) -> str:
        """把画像压成一句话，用于注入。"""
        parts: List[str] = []
        with self._lock:
            for key in PROFILE_INLINE_ORDER:
                cell = self.profile.get(key)
                if not cell:
                    continue
                val = str(cell.get("value") or "").strip()
                if val:
                    parts.append(val)
        return "；".join(parts)

    # ---------- 增删改查 ----------
    def add(
        self,
        text: str,
        kind: str = "fact",
        importance: float = 0.5,
        source: str = "",
        slot: str = "",
        tags: Optional[Iterable[str]] = None,
        confidence: float = 0.7,
        event_day: str = "",
    ) -> Optional[Dict[str, Any]]:
        """新增一条记忆。

        slot 非空表示这是「某个字段」，同 slot 只保留最新一条（覆盖式）。
        内容完全重复则提升重要度、累加命中次数，返回 None。
        """
        body = str(text or "").strip()
        if not body:
            return None
        kind = kind if kind in KINDS else "fact"
        key = _norm(body)
        slot = str(slot or "").strip()
        tag_list = [str(t).strip() for t in (tags or []) if str(t).strip()][:8]
        with self._lock:
            if slot:
                for it in self.items:
                    if it.get("slot") == slot:
                        if _norm(it.get("text", "")) == key:
                            it["updated_at"] = _now()
                            it["hits"] = int(it.get("hits", 0)) + 1
                            self._save()
                            return None
                        it["text"] = body
                        it["kind"] = kind
                        it["importance"] = max(0.0, min(1.0, float(importance)))
                        it["updated_at"] = _now()
                        it["hits"] = int(it.get("hits", 0)) + 1
                        it["tags"] = tag_list or it.get("tags", [])
                        self._save()
                        log.info("更新记忆[%s]：%s", slot, body)
                        return it
            for it in self.items:
                if _norm(it.get("text", "")) == key:
                    it["importance"] = min(1.0, max(float(it.get("importance", 0.5)), importance) + 0.1)
                    it["updated_at"] = _now()
                    it["hits"] = int(it.get("hits", 0)) + 1
                    self._save()
                    return None
            item = {
                "id": f"m{int(_now() * 1000)}",
                "text": body,
                "slot": slot,
                "kind": kind,
                "tags": tag_list,
                "importance": max(0.0, min(1.0, float(importance))),
                "confidence": max(0.0, min(1.0, float(confidence))),
                "event_day": str(event_day or ""),
                "created_at": _now(),
                "updated_at": _now(),
                "last_used_at": 0.0,
                "hits": 1,
                "source": str(source or ""),
            }
            self.items.append(item)
            self._trim_locked()
            self._save()
            log.info("记住：%s", body)
            return item

    def add_many(self, records: Iterable[Dict[str, Any]], source: str = "llm") -> int:
        """批量落库（抽取结果用），返回新增条数。"""
        n = 0
        for rec in records or []:
            if not isinstance(rec, dict):
                continue
            try:
                made = self.add(
                    str(rec.get("text") or ""),
                    kind=str(rec.get("kind") or "fact"),
                    importance=float(rec.get("importance", 0.6)),
                    source=str(rec.get("source") or source),
                    slot=str(rec.get("slot") or ""),
                    tags=rec.get("tags") or [],
                    confidence=float(rec.get("confidence", 0.7)),
                    event_day=str(rec.get("event_day") or ""),
                )
            except Exception:
                continue
            if made:
                n += 1
        return n

    def _trim_locked(self) -> None:
        if len(self.items) <= MAX_ITEMS:
            return
        self.items.sort(
            key=lambda x: (float(x.get("importance", 0.5)), float(x.get("updated_at", 0)))
        )
        drop = len(self.items) - MAX_ITEMS
        self.items = self.items[drop:]
        log.info("记忆超过上限，淘汰最旧最不重要的 %d 条", drop)

    def remove(self, mid: str) -> bool:
        with self._lock:
            before = len(self.items)
            self.items = [x for x in self.items if x.get("id") != mid]
            if len(self.items) != before:
                self._save()
                return True
            return False

    def update(self, mid: str, **fields: Any) -> Optional[Dict[str, Any]]:
        with self._lock:
            for it in self.items:
                if it.get("id") == mid:
                    if "text" in fields:
                        t = str(fields["text"] or "").strip()
                        if t:
                            it["text"] = t
                    if "kind" in fields and fields["kind"] in KINDS:
                        it["kind"] = fields["kind"]
                    if "importance" in fields:
                        try:
                            it["importance"] = max(0.0, min(1.0, float(fields["importance"])))
                        except (TypeError, ValueError):
                            pass
                    it["updated_at"] = _now()
                    self._save()
                    return it
            return None

    def list(self, kind: str = "", keyword: str = "") -> List[Dict[str, Any]]:
        out = list(self.items)
        if kind and kind in KINDS:
            out = [x for x in out if x.get("kind") == kind]
        kw = (keyword or "").strip()
        if kw:
            out = [x for x in out if kw in str(x.get("text", ""))]
        out.sort(
            key=lambda x: (float(x.get("importance", 0.5)), float(x.get("updated_at", 0))),
            reverse=True,
        )
        return out

    def clear_slot(self, slot: str) -> bool:
        slot = str(slot or "").strip()
        if not slot:
            return False
        with self._lock:
            before = len(self.items)
            self.items = [x for x in self.items if x.get("slot") != slot]
            if len(self.items) != before:
                self._save()
                return True
            return False

    def clear(self, keep_profile: bool = False) -> int:
        with self._lock:
            n = len(self.items)
            self.items = []
            if not keep_profile:
                self.profile = {}
            self._save()
            return n

    # ---------- 检索 ----------
    def _score(self, item: Dict[str, Any], qgrams: set, now: float) -> float:
        kind = str(item.get("kind") or "fact")
        base = float(item.get("importance", 0.5))
        kw = 0.0
        if qgrams:
            body = str(item.get("text", "")) + " " + " ".join(item.get("tags") or [])
            ig = _grams(body)
            if ig:
                kw = len(qgrams & ig) / max(1.0, min(len(qgrams), 6.0))
        kw_w = 3.2 if kind in ("fact", "kid", "prefer", "plan") else 1.6
        ts = float(item.get("updated_at") or item.get("created_at") or now)
        days = max(0.0, (now - ts) / 86400.0)
        half = 21.0 if kind == "event" else 120.0
        rec = math.exp(-days / half)
        hits = int(item.get("hits", 1))
        return kw_w * kw + 1.2 * base + 0.8 * rec + 0.15 * math.log(1.0 + hits)

    def search(self, query: str, limit: int = PROMPT_MAX_ITEMS) -> List[Dict[str, Any]]:
        """按相关度挑记忆（query 为空则只按重要度+新鲜度）。"""
        qgrams = _grams(query)
        now = _now()
        with self._lock:
            pool = list(self.items)
        scored = sorted(pool, key=lambda it: self._score(it, qgrams, now), reverse=True)
        return scored[: max(0, limit)]

    def mark_used(self, ids: Iterable[str]) -> None:
        ids = {str(i) for i in (ids or []) if i}
        if not ids:
            return
        changed = False
        with self._lock:
            now = _now()
            for it in self.items:
                if it.get("id") in ids:
                    it["last_used_at"] = now
                    changed = True
            if changed:
                self._save()

    # ---------- 给 LLM 用 ----------
    def build_context(self, query: str = "", max_chars: int = 0, mark: bool = True) -> str:
        """生成注入 system message 的记忆段落。

        画像全量在前（一句话），然后是相关度最高的事实与最近的经历。
        """
        limit = int(max_chars or self.settings.get("inject_max_chars") or PROMPT_MAX_CHARS)
        line = self.profile_line()
        picked = self.search(query, PROMPT_MAX_ITEMS) if self.items else []
        facts: List[str] = []
        events: List[str] = []
        used = 0
        used_ids: List[str] = []
        for it in picked:
            body = str(it.get("text", "")).strip()
            if not body:
                continue
            if used + len(body) > limit:
                break
            used += len(body)
            used_ids.append(it.get("id", ""))
            if str(it.get("kind")) == "event":
                events.append(body)
            else:
                facts.append(body)
        if not line and not facts and not events:
            return ""
        lines: List[str] = ["【你记得的事】"]
        if line:
            lines.append(f"关于对方：{line}")
        if facts:
            lines.append("还记得：")
            lines.extend(f"- {x}" for x in facts)
        if events:
            lines.append("最近的经历：")
            lines.extend(f"- {x}" for x in events)
        lines.append("聊天时自然地用上这些信息，不要生硬复述，也不要说「根据记忆」「我记得你」这类话。")
        if used_ids and mark:
            self.mark_used(used_ids)
        return "\n".join(lines)

    def prompt_block(self, limit: int = 0, max_chars: int = 0) -> str:
        """兼容旧调用：不带当前输入时只按重要度+新鲜度挑。"""
        return self.build_context("", max_chars)

    # ---------- 自动提取（规则兜底，不依赖额外 LLM 调用） ----------
    def extract_from_turn(self, user_text: str, reply: str) -> List[Dict[str, Any]]:
        """从一轮对话里用规则抽取值得记住的信息。

        只抽高置信度的：自我介绍、年龄年级、喜好、名字。
        """
        found: List[Dict[str, Any]] = []
        text = str(user_text or "").strip()
        if not text or len(text) > 200:
            return found
        flat = re.sub(r"[，,。；;！!？?、\s]+", " ", text).strip()
        rules = [
            (r"(?:^| )我?(?:叫|是)([\u4e00-\u9fa5]{2,4})(?: |$)", "kid", 0.9, "名字", "name"),
            (r"(?:^| )我?(?:今年)?([一二三四五六七八九十\d]{1,2})岁", "kid", 0.9, "年龄", "age"),
            (r"(?:^| )我?(?:上|读|在)([\u4e00-\u9fa5]{1,6}(?:年级|班))", "kid", 0.85, "年级", "grade"),
            (r"(?:^| )我?(?:最)?(?:喜欢|爱)(?:吃|玩|看|听|读)?([\u4e00-\u9fa5]{1,10})", "prefer", 0.7, "喜好", ""),
            (r"(?:^| )我?(?:不|讨厌)(?:喜欢|吃|玩|看)([\u4e00-\u9fa5]{1,10})", "prefer", 0.7, "不喜欢", ""),
        ]
        for pattern, kind, imp, label, slot in rules:
            m = re.search(pattern, flat)
            if not m:
                continue
            val = m.group(1).strip()
            if not val:
                continue
            who = self._user_name_hint()
            if label == "名字":
                body = f"{who}叫{val}"
                self.set_profile_field("name", val, "auto")
            elif label == "年龄":
                body = f"{who}{val}岁"
                self.set_profile_field("age", f"{val} 岁", "auto")
            elif label == "年级":
                body = f"{who}上{val}"
                self.set_profile_field("grade", val, "auto")
            elif label == "喜好":
                body = f"{who}喜欢{val}"
                cur = (self.profile.get("likes") or {}).get("value", "")
                if val not in cur:
                    self.set_profile_field("likes", (cur + "、" + val).strip("、") if cur else val, "auto")
            else:
                body = f"{who}不喜欢{val}"
                cur = (self.profile.get("dislikes") or {}).get("value", "")
                if val not in cur:
                    self.set_profile_field("dislikes", (cur + "、" + val).strip("、") if cur else val, "auto")
            it = self.add(body, kind=kind, importance=imp, source="auto", slot=slot)
            if it:
                found.append(it)
        return found

    def _user_name_hint(self) -> str:
        """抽取出来的句子主语，默认「孩子」，可由外部设置成配置里的称呼。"""
        return getattr(self, "user_hint", "") or "孩子"

    # ---------- 统计与视图 ----------
    def bump(self, key: str, delta: int = 1) -> None:
        with self._lock:
            self.stats[key] = int(self.stats.get(key, 0)) + delta
            self._save()

    def touch(self, key: str, value: Optional[float] = None) -> None:
        with self._lock:
            self.stats[key] = _now() if value is None else value
            self._save()

    def stats_view(self) -> Dict[str, Any]:
        with self._lock:
            st = dict(self.stats)
            st["items"] = len(self.items)
            st["profile_fields"] = len(self.profile)
            by_kind: Dict[str, int] = {}
            for it in self.items:
                k = str(it.get("kind") or "fact")
                by_kind[k] = by_kind.get(k, 0) + 1
            st["by_kind"] = by_kind
            return st

    def set_setting(self, key: str, value: Any) -> bool:
        if key not in DEFAULT_SETTINGS:
            return False
        with self._lock:
            if key == "inject_max_chars":
                try:
                    value = max(200, min(3000, int(value)))
                except (TypeError, ValueError):
                    return False
            elif key == "reflect_every":
                try:
                    value = max(0, min(100, int(value)))
                except (TypeError, ValueError):
                    return False
            else:
                value = bool(value)
            self.settings[key] = value
            self._save()
            return True

    def snapshot(self) -> Dict[str, Any]:
        return {
            "profile": self.profile_items(),
            "items": self.list(),
            "stats": self.stats_view(),
            "settings": dict(self.settings),
            "kinds": [{"id": k, "label": KIND_LABEL.get(k, k)} for k in KINDS],
            "profile_fields": [{"id": k, "label": v} for k, v in PROFILE_FIELDS],
        }


def _default_path(root: str | Path) -> Path:
    return Path(root) / "data" / "memory.json"
