"""定时任务：提醒与闹钟。

跟「被动应答」是两条完全独立的路径——这里由服务端时钟驱动，到点主动让设备说话。

三个必须处理好的点：

1. **空闲告别会杀死闹钟**。session 有 limits.idle_bye_seconds（默认 120 秒），
   没人说话就主动告别断开。夜里没人说话，闹钟根本等不到 7 点。
   所以「还有待触发任务的设备」必须豁免这条规则（has_pending_for）。

2. **重复任务不能用 fire_at += 86400 累加**。设备离线几天再上线，
   累加出来的时刻会漂移（离线 3 天就晚 3 天）。要按「时钟 + 日历」重新算下一次。

3. **离线到点不能直接丢弃**。记下 missed_at，设备重连后 30 分钟内补播；
   超过 30 分钟就不补了（晚了半小时的闹钟已经没有意义），重复任务直接跳到下一次。
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

log = logging.getLogger("schedule")

# 中国标准时间，固定 +08:00（跟 tools.py 保持一致，不依赖系统 tzdata）
CST = timezone(timedelta(hours=8))

REPEATS = ("once", "daily", "weekdays", "weekly")
KINDS = ("remind", "alarm")

# 设备离线错过之后，还能补播多久
MISSED_GRACE_SECONDS = 30 * 60

# 扫描间隔：5 秒足够准，又不会空转太凶
TICK_SECONDS = 5.0

# 单个任务最多攒多少条（防止误用把文件撑大）
MAX_ITEMS = 200

_WEEKDAY_NAMES = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]


def _clock_parts(clock: str) -> tuple:
    """把 "07:00" 解析成 (7, 0)；解析不出来就用 07:00。"""
    try:
        h, m = str(clock or "07:00").replace("：", ":").split(":")
        h, m = int(h), int(m)
        if 0 <= h <= 23 and 0 <= m <= 59:
            return h, m
    except Exception:  # noqa: BLE001
        pass
    return 7, 0


def compute_next(item: Dict[str, Any], now: Optional[float] = None) -> float:
    """按 repeat 算出下一次触发的绝对时刻（秒）。"""
    now = time.time() if now is None else now
    rep = item.get("repeat") or "once"
    if rep == "once":
        return float(item.get("fire_at") or now)
    h, m = _clock_parts(item.get("clock") or "07:00")
    base = datetime.fromtimestamp(now, CST)
    try:
        want_wd = int(item.get("weekday"))
    except (TypeError, ValueError):
        want_wd = 0
    for i in range(0, 15):
        day = base + timedelta(days=i)
        if rep == "weekdays" and day.weekday() >= 5:
            continue
        if rep == "weekly" and day.weekday() != want_wd:
            continue
        cand = day.replace(hour=h, minute=m, second=0, microsecond=0)
        # +1 秒容差：正好卡在当前这一秒时算「已经过了」，取下一次
        if cand.timestamp() > now + 1:
            return cand.timestamp()
    return now + 86400


def describe_repeat(item: Dict[str, Any]) -> str:
    """给控制台显示的中文描述。"""
    rep = item.get("repeat") or "once"
    clock = item.get("clock") or ""
    if rep == "once":
        return "一次"
    if rep == "daily":
        return f"每天 {clock}"
    if rep == "weekdays":
        return f"工作日 {clock}"
    if rep == "weekly":
        try:
            wd = int(item.get("weekday"))
        except (TypeError, ValueError):
            wd = 0
        return f"每{_WEEKDAY_NAMES[wd % 7]} {clock}"
    return rep


def fmt_time(ts: float) -> str:
    """时间戳 -> "10月26日 07:00"。"""
    if not ts:
        return "—"
    dt = datetime.fromtimestamp(float(ts), CST)
    return dt.strftime("%m月%d日 %H:%M")


class ScheduleStore:
    """定时任务的持久化（data/schedules.json）。"""

    def __init__(self, root: Path) -> None:
        self.path = Path(root) / "data" / "schedules.json"
        self._items: List[Dict[str, Any]] = []
        self.load()

    # ---------- 读写 ----------
    def load(self) -> None:
        try:
            if self.path.exists():
                data = json.loads(self.path.read_text(encoding="utf-8"))
                items = data.get("items") if isinstance(data, dict) else None
                if isinstance(items, list):
                    self._items = [it for it in items if isinstance(it, dict)]
                    log.info("已载入定时任务 %d 条", len(self._items))
        except Exception as exc:  # noqa: BLE001
            log.warning("读取定时任务失败: %s", exc)
            self._items = []

    def save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".json.tmp")
            tmp.write_text(
                json.dumps({"items": self._items}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            tmp.replace(self.path)
        except Exception as exc:  # noqa: BLE001
            log.warning("保存定时任务失败: %s", exc)

    # ---------- 查询 ----------
    def list_items(self) -> List[Dict[str, Any]]:
        """启用的排前面，同组按触发时间排序。"""
        return sorted(
            self._items,
            key=lambda it: (not it.get("enabled"), float(it.get("fire_at") or 0)),
        )

    def get(self, tid: str) -> Optional[Dict[str, Any]]:
        for it in self._items:
            if it.get("id") == tid:
                return it
        return None

    def has_pending_for(self, device_id: str = "") -> bool:
        """这台设备上还有没有待触发的任务（决定空闲时要不要告别）。"""
        for it in self._items:
            if not it.get("enabled"):
                continue
            dev = it.get("device_id") or ""
            if device_id and dev and dev != device_id:
                continue
            return True
        return False

    # ---------- 变更 ----------
    def create(
        self,
        *,
        text: str,
        fire_at: float,
        repeat: str = "once",
        clock: str = "",
        weekday: Optional[int] = None,
        device_id: str = "",
        kind: str = "remind",
        created_by: str = "voice",
    ) -> Dict[str, Any]:
        text = str(text or "").strip()[:200]
        if not text:
            raise ValueError("提醒内容不能为空")
        if len(self._items) >= MAX_ITEMS:
            raise ValueError(f"定时任务太多（上限 {MAX_ITEMS} 条），先删掉一些")
        item = {
            "id": "t_" + uuid.uuid4().hex[:8],
            "text": text,
            "kind": kind if kind in KINDS else "remind",
            "repeat": repeat if repeat in REPEATS else "once",
            "fire_at": float(fire_at),
            "clock": str(clock or ""),
            "weekday": weekday,
            "device_id": device_id or "",
            "enabled": True,
            "created_at": time.time(),
            "created_by": created_by,
            "last_fired_at": 0.0,
            "fired_count": 0,
            "missed_at": 0.0,
        }
        self._items.append(item)
        self.save()
        return item

    def delete(self, tid: str) -> bool:
        for i, it in enumerate(self._items):
            if it.get("id") == tid:
                del self._items[i]
                self.save()
                return True
        return False

    def set_enabled(self, tid: str, enabled: bool) -> Optional[Dict[str, Any]]:
        item = self.get(tid)
        if item is None:
            return None
        item["enabled"] = bool(enabled)
        if enabled and float(item.get("fire_at") or 0) <= time.time():
            # 重新打开时如果触发时刻已经过去，往后推，避免立刻炸一次
            rep = item.get("repeat") or "once"
            if rep == "once":
                item["fire_at"] = time.time() + 60
            else:
                item["fire_at"] = compute_next(item, time.time())
        if enabled:
            item["missed_at"] = 0.0
        self.save()
        return item


class ScheduleService:
    """后台扫描线程：到点就让设备说话。"""

    def __init__(
        self,
        store: ScheduleStore,
        hub: Any = None,
        sessions_getter: Optional[Callable[[], Dict[str, Any]]] = None,
    ) -> None:
        self.store = store
        self.hub = hub
        self._sessions = sessions_getter or (lambda: {})
        self._task: Optional[asyncio.Task] = None
        self._firing: set = set()

    # ---------- 生命周期 ----------
    def start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        self._task = asyncio.create_task(self._loop(), name="schedule-service")
        log.info("定时任务服务已启动，当前 %d 条", len(self.store.list_items()))

    async def stop(self) -> None:
        task, self._task = self._task, None
        if task is None:
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        log.info("定时任务服务已停止")

    def _next_delay(self) -> float:
        """算离最近一条任务还差几秒，用来精确睡到那一刻（原来固定 5 秒轮询，最坏晚 5 秒）。"""
        try:
            items = self.store.list_items()
        except Exception:  # noqa: BLE001
            return TICK_SECONDS
        now = time.time()
        nxt = 0.0
        for item in items:
            if not item.get("enabled"):
                continue
            fire_at = float(item.get("fire_at") or 0)
            if fire_at > now and (not nxt or fire_at < nxt):
                nxt = fire_at
        if not nxt:
            return TICK_SECONDS
        # 下限 0.05 秒防止空转，上限仍是 TICK_SECONDS 作为兜底扫描节奏
        return max(0.05, min(TICK_SECONDS, nxt - now))

    async def _loop(self) -> None:
        while True:
            try:
                await self.tick()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                log.warning("定时任务扫描出错: %s", exc)
            await asyncio.sleep(self._next_delay())

    # ---------- 核心 ----------
    def _online_session(self, device_id: str = "") -> Any:
        """找这台设备当前在线的会话；device_id 为空时随便找一个在线的。"""
        try:
            sessions = self._sessions() or {}
        except Exception:  # noqa: BLE001
            return None
        for sess in list(sessions.values()):
            if getattr(sess, "_closed", True):
                continue
            stats = getattr(sess, "stats", None)
            sid = getattr(stats, "device_id", "") if stats is not None else ""
            if not device_id or sid == device_id:
                return sess
        return None

    async def tick(self, now: Optional[float] = None) -> None:
        """扫一遍：到点的触发，之前错过的设备回来了就补播。"""
        now = time.time() if now is None else now
        for item in self.store.list_items():
            if not item.get("enabled"):
                continue
            tid = item.get("id")
            if tid in self._firing:
                continue
            fire_at = float(item.get("fire_at") or 0)
            missed = float(item.get("missed_at") or 0)
            # 到点，或错过且还在补播窗口内
            if fire_at and fire_at <= now:
                await self._fire(item, now, late=now - fire_at)
            elif missed and (now - missed) <= MISSED_GRACE_SECONDS:
                await self._fire(item, now, late=now - missed)

    async def _fire(self, item: Dict[str, Any], now: float, late: float = 0.0) -> None:
        tid = item.get("id")
        text = item.get("text") or ""
        self._firing.add(tid)
        try:
            sess = self._online_session(item.get("device_id") or "")
            if sess is None:
                await self._handle_offline(item, now)
                return

            kind = item.get("kind") or "remind"
            log.info("定时任务触发：%s（%s）", text, "迟到 %.1fs" % late if late > 1 else "准时")
            # 先把倒计时界面收掉再说话，否则播报被大字盖着。
            # 只收自己这条：屏幕可能正在给更晚的一条倒数，那条得接着走。
            hide = getattr(sess, "hide_countdown", None)
            if callable(hide):
                try:
                    await hide(str(tid or ""))
                except TypeError:  # 老版本/测试替身只收无参
                    try:
                        await hide()
                    except Exception as exc:  # noqa: BLE001
                        log.debug("收起倒计时界面失败: %s", exc)
                except Exception as exc:  # noqa: BLE001
                    log.debug("收起倒计时界面失败: %s", exc)
            ok = False
            speak = getattr(sess, "speak_proactive", None)
            if callable(speak):
                try:
                    ok = bool(await speak(text, tag=kind))
                except Exception as exc:  # noqa: BLE001
                    log.warning("定时任务播报失败: %s", exc)
            if not ok:
                # 播放失败也当作没播成，留到下一轮重试（同一 tick 内不会重复）
                log.warning("定时任务「%s」没有播成功，下一轮重试", text)
                return

            item["missed_at"] = 0.0
            item["last_fired_at"] = now
            item["fired_count"] = int(item.get("fired_count") or 0) + 1
            rep = item.get("repeat") or "once"
            if rep == "once":
                item["enabled"] = False
            else:
                item["fire_at"] = compute_next(item, now)
            self.store.save()
            if self.hub is not None:
                try:
                    suffix = f"（迟到 {int(late // 60)} 分钟）" if late >= 60 else ""
                    self.hub.add_log("schedule", f"已播报「{text}」{suffix}")
                except Exception:  # noqa: BLE001
                    pass
        finally:
            self._firing.discard(tid)

    async def _handle_offline(self, item: Dict[str, Any], now: float) -> None:
        """到点但设备不在线：记下来等重连，太久就放弃。"""
        text = item.get("text") or ""
        missed = float(item.get("missed_at") or 0)
        if not missed:
            item["missed_at"] = now
            self.store.save()
            log.info("定时任务「%s」到点但设备不在线，等待重连补播", text)
            if self.hub is not None:
                try:
                    self.hub.add_log("schedule", f"「{text}」到点，但设备不在线")
                except Exception:  # noqa: BLE001
                    pass
            return
        if (now - missed) <= MISSED_GRACE_SECONDS:
            return  # 还在补播窗口内，等设备回来
        # 错过太久：重复任务跳到下一次，一次性任务直接失效
        rep = item.get("repeat") or "once"
        item["missed_at"] = 0.0
        if rep == "once":
            item["enabled"] = False
            log.info("一次性任务「%s」错过太久，已失效", text)
        else:
            item["fire_at"] = compute_next(item, now)
            log.info("重复任务「%s」错过太久，跳到下一次 %s", text, fmt_time(item["fire_at"]))
        self.store.save()

    # ---------- 状态 ----------
    def status(self) -> Dict[str, Any]:
        items = self.store.list_items()
        enabled = [it for it in items if it.get("enabled")]
        upcoming = sorted(enabled, key=lambda it: float(it.get("fire_at") or 0))
        nxt = upcoming[0] if upcoming else None
        return {
            "total": len(items),
            "enabled": len(enabled),
            "next": (
                {
                    "text": nxt.get("text"),
                    "at": fmt_time(nxt.get("fire_at") or 0),
                    "in_seconds": max(0, int(round(float(nxt.get("fire_at") or 0) - time.time()))),
                }
                if nxt
                else None
            ),
        }


def build_item_view(item: Dict[str, Any]) -> Dict[str, Any]:
    """给前端/API 的展示结构。"""
    now = time.time()
    fire_at = float(item.get("fire_at") or 0)
    missed = float(item.get("missed_at") or 0)
    return {
        "id": item.get("id"),
        "text": item.get("text"),
        "kind": item.get("kind") or "remind",
        "repeat": item.get("repeat") or "once",
        "repeat_text": describe_repeat(item),
        "clock": item.get("clock") or "",
        "weekday": item.get("weekday"),
        "device_id": item.get("device_id") or "",
        "enabled": bool(item.get("enabled")),
        "fire_at": fire_at,
        "fire_text": fmt_time(fire_at),
        "in_seconds": max(0, int(round(fire_at - now))) if fire_at else 0,
        "missed": bool(missed),
        "fired_count": int(item.get("fired_count") or 0),
        "created_by": item.get("created_by") or "",
    }
