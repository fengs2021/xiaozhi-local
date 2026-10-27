"""后台记忆：用 LLM 从对话里抽取画像与事实，并定期把记忆库整理得更准确。

为什么需要它：
- 规则抽取（memory.MemoryStore.extract_from_turn）只能认出「我叫 X」「我 9 岁」这种固定句式，
  真实对话里大量信息（「我奶奶家养了只叫豆豆的猫」「我最近在学跳绳，还跳不到 20 个」）抓不到
- 所以这里挂一个后台队列：每轮对话结束后异步调一次小模型做抽取，绝不占用对话延迟
- 每 N 轮再跑一次「反思」：把散落在条目里的稳定信息归纳进画像、合并重复、淘汰过期

设计约束：
- 全部后台执行，失败只记日志，不影响对话
- 串行队列，一次只处理一个任务，避免把本地模型打满
- 抽取用 reasoning_effort=none 的轻配置，单次 1~2 秒
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

log = logging.getLogger("memory_agent")

EXTRACT_SYSTEM = """你是一个记忆整理助手。请阅读「用户」与「助手」的一轮对话，抽出值得长期记住的信息。

只输出 JSON，不要解释、不要 markdown 代码块：
{"profile": {"name": "", "age": "", "grade": "", "city": "", "school": "", "family": "", "likes": "", "dislikes": "", "habits": "", "study": "", "goal": "", "personality": "", "schedule": "", "other": ""},
 "facts": [{"text": "", "kind": "fact", "importance": 0.6, "slot": ""}],
 "events": [{"text": "", "importance": 0.5}]}

规则：
1. profile 只写本轮明确提到的字段，没提到的字段不要出现在 JSON 里（不要填空字符串占位）。
2. profile 的值要精炼：name 填孩子的昵称；age 填岁数；likes 填他喜欢的东西，多个用「、」连接。
3. facts 是稳定、以后还用得上的信息（爱好、家人、宠物、习惯、正在学的东西、怕的东西、身体情况等）。
   kind 取 fact / prefer / plan / kid 之一；一句话一条，不超过 30 字。
4. events 是某天发生的具体经历（今天去了动物园、考试考了 95 分），只在此刻确实发生了才填。
5. 寒暄、提问、助手自己说的话、临时情绪都不要记。
6. 没有值得记的就返回 {"profile":{},"facts":[],"events":[]}。
7. slot 只用于「同类只有一个值」的字段（name / age / grade / city / school），其它留空字符串。
8. 不要编造或推测对话里没有的内容。"""

REFLECT_SYSTEM = """你是一个记忆档案整理员。下面是一个助手关于某位用户的长期记忆库。
请把它整理得更准确、更像一份「人物档案」：

1. 把散落在条目里的稳定信息归纳进 profile（例如多条提到喜欢恐龙、喜欢画画，就合并成 likes「恐龙、画画」）。
2. 找出重复、已被更新内容取代、或明显过期的条目，给出它们的 id。
3. profile 要精炼、去重；同一字段有冲突时以时间更新的为准。
4. 不要编造条目里没有的信息。

只输出 JSON，不要解释、不要 markdown 代码块：
{"profile": {"name": "", "age": "", "grade": "", "city": "", "school": "", "family": "", "likes": "", "dislikes": "", "habits": "", "study": "", "goal": "", "personality": "", "schedule": "", "other": ""},
 "drop": ["m123"],
 "note": "一句话说明这次整理做了什么"}"""


def _strip_json(text: str) -> str:
    """从模型输出里抠出 JSON 主体（容忍代码块包裹与前后废话）。"""
    s = str(text or "").strip()
    tick = chr(96) * 3
    if tick in s:
        parts = [p for p in s.split(tick) if p.strip()]
        if parts:
            s = max(parts, key=len).strip()
        if s[:4].lower() == "json":
            s = s[4:].strip()
    i, j = s.find("{"), s.rfind("}")
    if i >= 0 and j > i:
        return s[i : j + 1]
    return s


def _as_dict(raw: str) -> Dict[str, Any]:
    try:
        data = json.loads(_strip_json(raw))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


class MemoryAgent:
    """记忆抽取 / 反思的后台执行器（串行队列）。"""

    def __init__(
        self,
        store: Any,
        make_llm: Callable[[], Any],
        user_name_getter: Optional[Callable[[], str]] = None,
    ):
        self.store = store
        self._make_llm = make_llm
        self._user_name = user_name_getter or (lambda: "")
        self._queue: "asyncio.Queue[Dict[str, Any]]" = asyncio.Queue(maxsize=64)
        self._worker: Optional[asyncio.Task] = None
        self._running = False
        self.busy = False
        self.last_error = ""

    # ---------- 生命周期 ----------
    def start(self) -> None:
        if self._worker is not None and not self._worker.done():
            return
        self._running = True
        self._worker = asyncio.create_task(self._loop(), name="memory-agent")
        log.info("后台记忆已启动")

    async def stop(self) -> None:
        self._running = False
        if self._worker is not None:
            self._worker.cancel()
            try:
                await self._worker
            except (asyncio.CancelledError, Exception):
                pass
            self._worker = None
        log.info("后台记忆已停止")

    def _enabled(self) -> bool:
        try:
            return bool(self.store.settings.get("auto", True))
        except Exception:
            return True

    def _submit(self, job: Dict[str, Any]) -> bool:
        if not self._enabled():
            return False
        if self._worker is None or self._worker.done():
            try:
                self.start()
            except RuntimeError:
                return False
        try:
            self._queue.put_nowait(job)
            return True
        except asyncio.QueueFull:
            log.warning("记忆队列已满，丢弃一个任务")
            return False

    # ---------- 对外接口 ----------
    def submit_turn(self, user_text: str, reply: str, session_id: str = "") -> bool:
        """对话结束后调用：入队一次抽取（非阻塞）。"""
        if not str(user_text or "").strip():
            return False
        return self._submit(
            {"type": "extract", "user": str(user_text), "reply": str(reply or ""), "session": session_id}
        )

    def submit_reflect(self) -> bool:
        """入队一次反思整理。"""
        return self._submit({"type": "reflect"})

    # ---------- 主循环 ----------
    async def _loop(self) -> None:
        while self._running:
            try:
                job = await self._queue.get()
            except asyncio.CancelledError:
                break
            self.busy = True
            try:
                if job.get("type") == "reflect":
                    await self._do_reflect()
                else:
                    await self._do_extract(job)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                self.last_error = str(exc)
                try:
                    self.store.bump("llm_failures")
                except Exception:
                    pass
                log.warning("记忆任务失败: %s", exc)
            finally:
                self.busy = False
                self._queue.task_done()
                try:
                    if job.get("type") == "extract":
                        await self._maybe_reflect()
                except Exception:
                    pass

    async def _maybe_reflect(self) -> None:
        try:
            every = int(self.store.settings.get("reflect_every", 10) or 0)
            turns = int(self.store.stats.get("turns", 0) or 0)
        except Exception:
            return
        if every <= 0 or turns <= 0 or turns % every != 0:
            return
        if self._queue.empty():
            self._submit({"type": "reflect"})

    # ---------- 抽取 ----------
    async def _do_extract(self, job: Dict[str, Any]) -> None:
        user_text = str(job.get("user") or "").strip()
        reply = str(job.get("reply") or "").strip()
        if not user_text:
            return
        if len(user_text) < 3 and len(reply) < 3:
            return
        try:
            self.store.bump("turns")
            self.store.touch("last_turn_at")
        except Exception:
            pass

        known_profile = {p["field"]: p["value"] for p in self.store.profile_items()}
        known_texts = [str(x.get("text", ""))[:40] for x in self.store.list()[:24]]
        prompt = (
            "用户称呼：" + (self._user_name() or "未知")
            + "\n已有画像：" + (json.dumps(known_profile, ensure_ascii=False) if known_profile else "（空）")
            + "\n已有记忆条目（避免重复抽取）："
            + ("\n".join("- " + t for t in known_texts) if known_texts else "（空）")
            + "\n\n本轮对话：\n用户：" + user_text[:400]
            + "\n助手：" + reply[:300]
        )

        llm = self._make_llm()
        if llm is None:
            return
        raw = await llm.complete(EXTRACT_SYSTEM, prompt, max_tokens=700, temperature=0.1, timeout=25.0)
        try:
            self.store.bump("llm_calls")
        except Exception:
            pass
        data = _as_dict(raw)
        if not data:
            log.info("记忆抽取返回无法解析：%s", str(raw)[:120])
            return

        prof = data.get("profile")
        if isinstance(prof, dict):
            changed = self.store.set_profile(prof, "llm")
            if changed:
                log.info("画像更新 %d 项", changed)

        records: List[Dict[str, Any]] = []
        for f in data.get("facts") or []:
            if isinstance(f, dict) and str(f.get("text") or "").strip():
                records.append(
                    {
                        "text": f.get("text"),
                        "kind": str(f.get("kind") or "fact"),
                        "importance": f.get("importance", 0.6),
                        "slot": str(f.get("slot") or ""),
                        "source": "llm",
                        "confidence": 0.75,
                    }
                )
        today = datetime.now().strftime("%Y-%m-%d")
        for ev in data.get("events") or []:
            if isinstance(ev, dict) and str(ev.get("text") or "").strip():
                records.append(
                    {
                        "text": ev.get("text"),
                        "kind": "event",
                        "importance": ev.get("importance", 0.5),
                        "event_day": today,
                        "source": "llm",
                        "confidence": 0.7,
                    }
                )
        n = self.store.add_many(records, source="llm")
        if n:
            try:
                self.store.bump("extracted", n)
            except Exception:
                pass
            log.info("本轮记住 %d 条：%s", n, "；".join(str(r["text"]) for r in records[:3]))
        try:
            self.store.touch("last_extract_at")
        except Exception:
            pass

    # ---------- 反思 ----------
    async def _do_reflect(self) -> None:
        items = self.store.list()
        if not items:
            return
        prof = {p["field"]: p["value"] for p in self.store.profile_items()}
        rows = []
        for it in items[:60]:
            rows.append(
                "- [%s] (%s, %.1f) %s"
                % (it.get("id"), it.get("kind"), float(it.get("importance", 0.5)), str(it.get("text"))[:60])
            )
        prompt = (
            "当前画像：" + (json.dumps(prof, ensure_ascii=False) if prof else "（空）")
            + "\n\n记忆条目（共 %d 条，列出 %d 条）：\n%s" % (len(items), len(rows), "\n".join(rows))
        )
        llm = self._make_llm()
        if llm is None:
            return
        raw = await llm.complete(REFLECT_SYSTEM, prompt, max_tokens=900, temperature=0.1, timeout=40.0)
        try:
            self.store.bump("llm_calls")
        except Exception:
            pass
        data = _as_dict(raw)
        if not data:
            log.info("反思返回无法解析：%s", str(raw)[:120])
            return
        prof2 = data.get("profile")
        if isinstance(prof2, dict):
            self.store.set_profile(prof2, "reflect")
        dropped = 0
        for mid in data.get("drop") or []:
            if isinstance(mid, str) and self.store.remove(mid):
                dropped += 1
        try:
            self.store.touch("last_reflect_at")
            self.store.bump("reflects")
        except Exception:
            pass
        note = str(data.get("note") or "").strip()
        log.info("记忆反思完成：画像 %d 项，合并/删除 %d 条。%s", len(self.store.profile), dropped, note)

    # ---------- 状态 ----------
    def status(self) -> Dict[str, Any]:
        return {
            "running": bool(self._worker is not None and not self._worker.done()),
            "busy": bool(self.busy),
            "queued": int(self._queue.qsize()) if self._queue is not None else 0,
            "auto": self._enabled(),
            "last_error": self.last_error,
        }
