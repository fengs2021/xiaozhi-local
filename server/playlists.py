"""歌单存储：控制中心里手管的几个歌单，落盘 data/playlists.json。

结构 {"items": [{"id","name","songs":[...],"created_at","updated_at"}]}
一首歌就是 {"id","title","artist","duration"}，和网易云搜索结果同形，方便直接丢给 MusicSource.fetch。
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger("playlists")

MAX_ITEMS = 100
MAX_SONGS = 500
MAX_NAME = 24


def _norm_song(song: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(song, dict):
        return None
    sid = str(song.get("id") or "").strip()
    if not sid:
        return None
    try:
        duration = int(song.get("duration") or 0)
    except (TypeError, ValueError):
        duration = 0
    return {
        "id": sid,
        "title": str(song.get("title") or "").strip()[:80],
        "artist": str(song.get("artist") or "").strip()[:80],
        "duration": max(0, duration),
    }


class PlaylistStore:
    def __init__(self, root: Path) -> None:
        self.path = Path(root) / "data" / "playlists.json"
        self.items: List[Dict[str, Any]] = []
        self.load()

    def load(self) -> None:
        self.items = []
        try:
            if not self.path.exists():
                return
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            log.warning("读歌单失败: %s", exc)
            return
        items = raw.get("items") if isinstance(raw, dict) else raw
        if not isinstance(items, list):
            return
        for it in items:
            if not isinstance(it, dict) or not it.get("id"):
                continue
            songs: List[Dict[str, Any]] = []
            for s in it.get("songs") or []:
                clean = _norm_song(s)
                if clean:
                    songs.append(clean)
            self.items.append(
                {
                    "id": str(it["id"]),
                    "name": str(it.get("name") or "未命名歌单")[:MAX_NAME],
                    "songs": songs[:MAX_SONGS],
                    "created_at": float(it.get("created_at") or time.time()),
                    "updated_at": float(it.get("updated_at") or time.time()),
                }
            )
        self.items.sort(key=lambda x: x["created_at"])

    def save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".json.tmp")
            tmp.write_text(
                json.dumps({"items": self.items}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            tmp.replace(self.path)
        except Exception as exc:  # noqa: BLE001
            log.warning("写歌单失败: %s", exc)

    # ---------- 查询 ----------

    def list_items(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": it["id"],
                "name": it["name"],
                "count": len(it["songs"]),
                "songs": list(it["songs"]),
                "created_at": it["created_at"],
                "updated_at": it["updated_at"],
            }
            for it in self.items
        ]

    def get(self, pid: str) -> Optional[Dict[str, Any]]:
        for it in self.items:
            if it["id"] == pid:
                return it
        return None

    def _unique_name(self, name: str, skip_id: str = "") -> str:
        base = (name or "").strip()[:MAX_NAME] or "未命名歌单"
        used = {it["name"] for it in self.items if it["id"] != skip_id}
        if base not in used:
            return base
        for n in range(2, 100):
            cand = f"{base} ({n})"[:MAX_NAME]
            if cand not in used:
                return cand
        return base

    # ---------- 写 ----------

    def create(self, name: str, songs: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        if len(self.items) >= MAX_ITEMS:
            raise ValueError(f"歌单最多 {MAX_ITEMS} 个")
        now = time.time()
        clean_songs: List[Dict[str, Any]] = []
        for s in songs or []:
            cs = _norm_song(s)
            if cs:
                clean_songs.append(cs)
        item = {
            "id": "pl_" + uuid.uuid4().hex[:8],
            "name": self._unique_name(name),
            "songs": clean_songs[:MAX_SONGS],
            "created_at": now,
            "updated_at": now,
        }
        self.items.append(item)
        self.save()
        return item

    def rename(self, pid: str, name: str) -> Dict[str, Any]:
        item = self.get(pid)
        if not item:
            raise KeyError(pid)
        item["name"] = self._unique_name(name, skip_id=pid)
        item["updated_at"] = time.time()
        self.save()
        return item

    def delete(self, pid: str) -> None:
        before = len(self.items)
        self.items = [it for it in self.items if it["id"] != pid]
        if len(self.items) == before:
            raise KeyError(pid)
        self.save()

    def add_song(self, pid: str, song: Dict[str, Any]) -> Dict[str, Any]:
        item = self.get(pid)
        if not item:
            raise KeyError(pid)
        clean = _norm_song(song)
        if not clean:
            raise ValueError("这首歌没有 id")
        if any(s["id"] == clean["id"] for s in item["songs"]):
            return item
        if len(item["songs"]) >= MAX_SONGS:
            raise ValueError(f"一个歌单最多 {MAX_SONGS} 首")
        item["songs"].append(clean)
        item["updated_at"] = time.time()
        self.save()
        return item

    def remove_song(self, pid: str, song_id: str) -> Dict[str, Any]:
        item = self.get(pid)
        if not item:
            raise KeyError(pid)
        item["songs"] = [s for s in item["songs"] if s["id"] != str(song_id)]
        item["updated_at"] = time.time()
        self.save()
        return item
