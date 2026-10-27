"""对话历史存储：按天追加写入 data/chats/YYYY-MM-DD.jsonl。

每行一条 JSON：{id, ts, session_id, device_id, user_text, reply_text,
asr_ms, llm_ms, tts_ms, first_audio_ms, play_ms, total_ms, audio, audio_secs}。
量很小，直接用同步 append，避免引入额外复杂度。

耗时字段口径（互相独立，不要相加代替彼此）：
  asr_ms         语音识别：收到完整语音 → 出文本
  llm_ms         模型生成：识别完成 → 最后一个字吐完
  tts_ms         语音合成累计：各句合成耗时之和（多句并行，可能大于墙钟时间）
  first_audio_ms 首包延迟：识别完成 → 第一块音频下发（用户实际感知的等待）
  play_ms        音频下发：第一块 → 最后一块，约等于音频本身时长
  total_ms       整轮：识别完成 → 播放结束，= 首包 + 播放（不含 ASR）
"""
from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger("history")


def _day_of(ts: float) -> str:
    """把时间戳转成本地日期字符串 YYYY-MM-DD。"""
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d")


class ChatStore:
    """对话记录的追加与查询。"""

    def __init__(self, root: str | Path, enabled: bool = True) -> None:
        self.root = Path(root)
        self.dir = self.root / "data" / "chats"
        self.enabled = bool(enabled)
        # 同步写文件，加锁保证多协程/线程下不会写串行
        self._lock = threading.Lock()
        if self.enabled:
            try:
                self.dir.mkdir(parents=True, exist_ok=True)
            except OSError as exc:  # 目录建不出来就降级为不记录
                log.warning("对话历史目录创建失败，已停用记录: %s", exc)
                self.enabled = False

    # ---------- 写入 ----------
    def append(
        self,
        session_id: str,
        device_id: str,
        user_text: str,
        reply_text: str,
        asr_ms: int = 0,
        llm_ms: int = 0,
        tts_ms: int = 0,
        first_audio_ms: int = 0,
        play_ms: int = 0,
        total_ms: int = 0,
        ts: Optional[float] = None,
        audio: Optional[bytes] = None,
        audio_secs: float = 0.0,
    ) -> Optional[Dict[str, Any]]:
        """追加一轮对话，返回写入的记录（未启用时返回 None）。

        audio 传入这一轮用户的原始录音（WAV 字节），会存到
        data/chats/audio/YYYY-MM-DD/<id>.wav，记录里只存相对路径。
        """
        if not self.enabled:
            return None
        now = time.time() if ts is None else float(ts)
        item_id = self._make_id(now)
        audio_rel = ""
        if audio:
            audio_rel = self._write_audio(item_id, now, audio)
        item: Dict[str, Any] = {
            "id": item_id,
            "ts": round(now, 3),
            "session_id": session_id or "",
            "device_id": device_id or "",
            "user_text": user_text or "",
            "reply_text": reply_text or "",
            "asr_ms": int(asr_ms or 0),
            "llm_ms": int(llm_ms or 0),
            "tts_ms": int(tts_ms or 0),
            "first_audio_ms": int(first_audio_ms or 0),
            "play_ms": int(play_ms or 0),
            "total_ms": int(total_ms or 0),
        }
        if audio_rel:
            item["audio"] = audio_rel
            item["audio_secs"] = round(float(audio_secs or 0), 2)
        path = self.dir / f"{_day_of(now)}.jsonl"
        line = json.dumps(item, ensure_ascii=False)
        try:
            with self._lock:
                with open(path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
        except OSError as exc:
            log.warning("写对话历史失败: %s", exc)
            return None
        return item

    def _make_id(self, ts: float) -> str:
        """id 用「毫秒时间戳-短 uuid」，既有序又不重复。"""
        return f"{int(ts * 1000)}-{uuid.uuid4().hex[:8]}"

    def _write_audio(self, item_id: str, ts: float, wav_bytes: bytes) -> str:
        """把这一轮用户的原始录音写成 wav，返回相对 data/chats 的路径。"""
        day = _day_of(ts)
        try:
            target = self.dir / "audio" / day
            target.mkdir(parents=True, exist_ok=True)
            with open(target / f"{item_id}.wav", "wb") as f:
                f.write(wav_bytes)
        except OSError as exc:
            log.warning("写用户录音失败: %s", exc)
            return ""
        return f"audio/{day}/{item_id}.wav"

    def audio_path(self, rel: str) -> Optional[Path]:
        """把记录里的相对路径解析成绝对路径，且必须落在 data/chats 内。"""
        text = (rel or "").strip().replace("\\", "/")
        if not text or ".." in text.split("/"):
            return None
        base = self.dir.resolve()
        try:
            path = (base / text).resolve()
            path.relative_to(base)
        except (OSError, ValueError):
            return None
        return path if path.is_file() else None

    # ---------- 读取 ----------
    def list_days(self) -> List[str]:
        """返回所有有记录的日期，新的在前。"""
        if not self.dir.exists():
            return []
        days = []
        for p in self.dir.glob("*.jsonl"):
            stem = p.stem
            # 只认 YYYY-MM-DD 形式的文件名
            if len(stem) == 10 and stem[4] == "-" and stem[7] == "-":
                days.append(stem)
        days.sort(reverse=True)
        return days

    def _read_day(self, day: str) -> List[Dict[str, Any]]:
        """读取某一天的全部记录，坏行直接跳过。"""
        path = self.dir / f"{day}.jsonl"
        if not path.exists():
            return []
        items: List[Dict[str, Any]] = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(obj, dict):
                        items.append(obj)
        except OSError as exc:
            log.warning("读取对话历史失败 %s: %s", day, exc)
            return []
        return items

    def list_items(
        self, date: Optional[str] = None, limit: int = 100, offset: int = 0
    ) -> Dict[str, Any]:
        """按日期分页取记录（新的在前），返回 items 与总数。"""
        days = self.list_days()
        day = (date or "").strip() or (days[0] if days else _day_of(time.time()))
        items = self._read_day(day)
        items.reverse()  # 文件里是追加顺序，倒过来才是新的在前
        total = len(items)
        try:
            off = max(0, int(offset))
        except (TypeError, ValueError):
            off = 0
        try:
            lim = max(1, int(limit))
        except (TypeError, ValueError):
            lim = 100
        return {
            "days": days,
            "date": day,
            "items": items[off : off + lim],
            "total": total,
        }

    def list_all(self, max_days: int = 60) -> List[Dict[str, Any]]:
        """按时间正序取最近若干天的全部记录（用于按会话聚合）。"""
        days = self.list_days()[: max(1, int(max_days))]
        items: List[Dict[str, Any]] = []
        for day in reversed(days):  # 旧的在前，拼出来才是时间正序
            items.extend(self._read_day(day))
        items.sort(key=lambda x: float(x.get("ts") or 0))
        return items

    def sessions(self, max_days: int = 60) -> List[Dict[str, Any]]:
        """把所有记录按 session_id 聚成会话，新的在前。"""
        buckets: Dict[str, List[Dict[str, Any]]] = {}
        order: List[str] = []
        for it in self.list_all(max_days):
            sid = str(it.get("session_id") or "").strip() or "unknown"
            if sid not in buckets:
                buckets[sid] = []
                order.append(sid)
            buckets[sid].append(it)

        out: List[Dict[str, Any]] = []
        for sid in order:
            msgs = buckets[sid]
            first, last = msgs[0], msgs[-1]
            title = ""
            for m in msgs:
                text = str(m.get("user_text") or "").strip()
                if text:
                    title = text
                    break
            if not title:
                title = "（无语音内容）"
            start_ts = float(first.get("ts") or 0)
            out.append(
                {
                    "session_id": sid,
                    "title": title[:40],
                    "device_id": str(first.get("device_id") or ""),
                    "date": _day_of(start_ts or time.time()),
                    "start_ts": start_ts,
                    "end_ts": float(last.get("ts") or start_ts),
                    "count": len(msgs),
                }
            )
        out.sort(key=lambda s: s["end_ts"], reverse=True)
        return out

    def session_messages(self, session_id: str, max_days: int = 60) -> List[Dict[str, Any]]:
        """取某个会话的全部消息（时间正序）。"""
        sid = (session_id or "").strip()
        if not sid:
            return []
        return [
            m
            for m in self.list_all(max_days)
            if str(m.get("session_id") or "").strip() == sid
        ]

    def get_item(self, item_id: str) -> Optional[Dict[str, Any]]:
        """按 id 查找一条记录；id 前缀是毫秒时间戳，可据此定位到具体文件。"""
        target = (item_id or "").strip()
        if not target:
            return None
        # id 形如 1792721464655-abcd1234，取前半段算日期，命中率高
        day = ""
        head = target.split("-", 1)[0]
        if head.isdigit():
            try:
                day = _day_of(int(head) / 1000.0)
            except (OverflowError, OSError, ValueError):
                day = ""
        days = [day] if day else []
        # 兜底：日期推算失败时全量扫（数据量小，可接受）
        for d in days + [x for x in self.list_days() if x != day]:
            for obj in self._read_day(d):
                if str(obj.get("id", "")) == target:
                    return obj
        return None
