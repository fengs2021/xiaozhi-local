"""每日输出字数统计：给儿童使用场景做「聊太久就劝退」的保护。

上游 xiaozhi-esp32-server 用 utils/output_counter.py 做同样的事，这里保持
接口更小：按「设备 + 自然日」累计模型回复的字数，跨天自动从零开始。
数据落盘到 data/output_count.json，重启不丢；只保留最近 7 天。
"""
from __future__ import annotations

import json
import logging
import threading
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict

log = logging.getLogger("limits")

KEEP_DAYS = 7


class OutputCounter:
    """按设备累计当日输出字数。"""

    def __init__(self, root: str):
        self.path = Path(root) / "data" / "output_count.json"
        self._lock = threading.Lock()
        self._data: Dict[str, Dict[str, int]] = {}
        self._load()

    # ---------- 落盘 ----------
    def _load(self) -> None:
        try:
            if self.path.exists():
                raw = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    for day, per_dev in raw.items():
                        if isinstance(per_dev, dict):
                            self._data[str(day)] = {
                                str(k): int(v) for k, v in per_dev.items() if isinstance(v, (int, float))
                            }
        except Exception as exc:  # noqa: BLE001
            log.warning("读取输出计数失败: %s", exc)

    def _save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # 只留最近 KEEP_DAYS 天，避免文件无限长大
            keep_from = (date.today() - timedelta(days=KEEP_DAYS)).isoformat()
            self._data = {k: v for k, v in self._data.items() if k >= keep_from}
            tmp = self.path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(self._data, ensure_ascii=False), encoding="utf-8")
            tmp.replace(self.path)
        except Exception as exc:  # noqa: BLE001
            log.warning("保存输出计数失败: %s", exc)

    # ---------- 查询 ----------
    @staticmethod
    def _today() -> str:
        return date.today().isoformat()

    def used(self, device_id: str) -> int:
        """今天这台设备已经说了多少字。"""
        with self._lock:
            return int(self._data.get(self._today(), {}).get(device_id or "-", 0))

    def add(self, device_id: str, chars: int) -> int:
        """累加并返回今天的累计字数。"""
        if chars <= 0:
            return self.used(device_id)
        key = device_id or "-"
        with self._lock:
            day = self._today()
            bucket = self._data.setdefault(day, {})
            bucket[key] = int(bucket.get(key, 0)) + int(chars)
            total = bucket[key]
            self._save()
            return total

    def snapshot(self) -> Dict[str, Any]:
        """给网页看的当日快照。"""
        with self._lock:
            day = self._today()
            return {"date": day, "devices": dict(self._data.get(day, {}))}

    def reset_today(self, device_id: str = "") -> None:
        """清空今天（或指定设备）的计数，方便调试。"""
        with self._lock:
            day = self._today()
            if device_id:
                self._data.get(day, {}).pop(device_id or "-", None)
            else:
                self._data.pop(day, None)
            self._save()
