"""人设库：多套人设的存取、复制与切换。

每条记录是一整套 persona 字段；切换人设时整体写回 config.yaml 的 persona 段，
LLM 的 system prompt 和 TTS 的语速/音调都跟着走。
"""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import DEFAULT_PERSONA, sanitize_persona

ITEM_FIELDS = tuple(DEFAULT_PERSONA.keys())
DEFAULT_EMOJI = "🎭"
MAX_NAME = 24


def _now() -> float:
    return time.time()


def guess_name(persona: Dict[str, Any]) -> str:
    """从性格设定里猜人设名（找 `# Role: xxx` 那一行），猜不到就用默认名。"""
    text = str(persona.get("character") or "")
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        low = s.lower()
        if low.startswith("# role:") or low.startswith("role:"):
            name = s.split(":", 1)[1].strip(" #*：")
            if name:
                return name[:MAX_NAME]
    return "默认人设"


class PersonaStore:
    """data/personas.json 的读写。文件不存在时由 ensure_seed 用当前配置兜底。"""

    def __init__(self, root: str | Path):
        self.path = Path(root) / "data" / "personas.json"
        self._items: List[Dict[str, Any]] = []
        self._active: str = ""
        self.load()

    # ---------- 持久化 ----------
    def load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return
        if not isinstance(raw, dict):
            return
        items = raw.get("items")
        if isinstance(items, list):
            self._items = [it for it in items if isinstance(it, dict) and it.get("id")]
        self._active = str(raw.get("active") or "")

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"active": self._active, "items": self._items}
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    # ---------- 内部工具 ----------
    @property
    def active_id(self) -> str:
        return self._active

    def _unique_name(self, name: str, skip_id: str = "") -> str:
        base = (name or "").strip()[:MAX_NAME] or "未命名人设"
        taken = {it.get("name") for it in self._items if it.get("id") != skip_id}
        if base not in taken:
            return base
        n = 2
        while f"{base} ({n})" in taken:
            n += 1
        return f"{base} ({n})"

    def _make(self, name: str, data: Dict[str, Any], emoji: str = "") -> Dict[str, Any]:
        clean = sanitize_persona(data)
        now = round(_now(), 3)
        item: Dict[str, Any] = {
            "id": "p_" + uuid.uuid4().hex[:8],
            "name": self._unique_name(name),
            "emoji": (str(emoji or "").strip() or DEFAULT_EMOJI)[:4],
        }
        for key in ITEM_FIELDS:
            item[key] = clean[key]
        item["created_at"] = now
        item["updated_at"] = now
        return item

    # ---------- 查询 ----------
    def get(self, pid: str) -> Optional[Dict[str, Any]]:
        for it in self._items:
            if it.get("id") == pid:
                return it
        return None

    def list_items(self) -> List[Dict[str, Any]]:
        return [dict(it) for it in self._items]

    def active_item(self) -> Optional[Dict[str, Any]]:
        return self.get(self._active)

    # ---------- 变更 ----------
    def ensure_seed(self, persona: Dict[str, Any]) -> bool:
        """首次运行：把 config.yaml 里正在用的人设收成第一条记录。"""
        if self._items:
            if not self.get(self._active):
                self._active = self._items[0]["id"]
                self.save()
            return False
        item = self._make(guess_name(persona), persona or {})
        self._items.append(item)
        self._active = item["id"]
        self.save()
        return True

    def create(self, data: Dict[str, Any]) -> Dict[str, Any]:
        item = self._make(str(data.get("name") or ""), data, str(data.get("emoji") or ""))
        self._items.insert(0, item)
        if not self._active:
            self._active = item["id"]
        self.save()
        return item

    def update(self, pid: str, data: Dict[str, Any]) -> Dict[str, Any]:
        item = self.get(pid)
        if not item:
            raise KeyError(pid)
        if "name" in data:
            item["name"] = self._unique_name(str(data.get("name") or ""), skip_id=pid)
        if "emoji" in data:
            item["emoji"] = (str(data.get("emoji") or "").strip() or DEFAULT_EMOJI)[:4]
        merged = {k: data.get(k, item.get(k)) for k in ITEM_FIELDS}
        clean = sanitize_persona(merged)
        for key in ITEM_FIELDS:
            item[key] = clean[key]
        item["updated_at"] = round(_now(), 3)
        self.save()
        return item

    def duplicate(self, pid: str) -> Dict[str, Any]:
        src = self.get(pid)
        if not src:
            raise KeyError(pid)
        item = self._make(f"{src.get('name') or '人设'} 副本", src, str(src.get("emoji") or ""))
        self._items.insert(0, item)
        self.save()
        return item

    def delete(self, pid: str) -> Dict[str, Any]:
        if not self.get(pid):
            raise KeyError(pid)
        if len(self._items) <= 1:
            raise ValueError("至少要保留一套人设")
        self._items = [it for it in self._items if it.get("id") != pid]
        switched = None
        if self._active == pid:
            self._active = self._items[0]["id"]
            switched = self._items[0]
        self.save()
        return {"deleted": pid, "switched": switched}

    def set_active(self, pid: str) -> Dict[str, Any]:
        item = self.get(pid)
        if not item:
            raise KeyError(pid)
        self._active = pid
        self.save()
        return item

    def sync_active(self, persona: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """config.yaml 里的人设被直接改动时，把变化同步回当前使用的那条记录。"""
        item = self.active_item()
        if not item:
            return None
        merged = {k: item.get(k) for k in ITEM_FIELDS}
        for key in ITEM_FIELDS:
            if key in persona:
                merged[key] = persona[key]
        clean = sanitize_persona(merged)
        if all(item.get(k) == clean[k] for k in ITEM_FIELDS):
            return item
        for key in ITEM_FIELDS:
            item[key] = clean[key]
        item["updated_at"] = round(_now(), 3)
        self.save()
        return item
