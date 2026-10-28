"""大模型对话：OpenAI 兼容接口，支持任意自定义 base_url/model。"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict, List

log = logging.getLogger("llm")

_WEEKDAYS = "一二三四五六日"


def now_block() -> str:
    """实时时间块：让模型直接回答时间/日期类问题，省掉一次工具往返（1.4~2.6s）。

    刻意放在系统提示的最后：前面的「人设 + 记忆 + 工具规则」在分钟变化时
    仍能命中前缀缓存，只有末尾这一小段每轮变化。
    """
    now = datetime.now()
    return (
        f"【当前时间】{now.strftime('%Y年%m月%d日')} 星期{_WEEKDAYS[now.weekday()]} "
        f"{now.strftime('%H:%M')}。"
        "问到时间、日期、星期、月份、季节这类问题时，直接用这个时间回答，不要调用工具。"
    )

# 进程级共享的 SSL 上下文与 httpx 客户端。
# 起因：httpx.AsyncClient(verify=True) 每次构造都要重新解析 certifi 证书链，
# 本机（Windows + 杀软 HTTPS 扫描）实测单次 3.8~4.4 秒；openai SDK 每 new 一个
# AsyncOpenAI 就会 new 一个 httpx 客户端，于是「试一句」凭空多出 4 秒。
# 只建一次上下文并复用同一个客户端，可彻底消除这段损耗，顺带享受连接复用。
_SSL_CONTEXT: Any = None
_HTTP_CLIENT: Any = None


def _shared_http_client():
    """懒加载一个进程级共享的 httpx.AsyncClient。"""
    global _SSL_CONTEXT, _HTTP_CLIENT
    import ssl

    import httpx

    if _HTTP_CLIENT is not None and not _HTTP_CLIENT.is_closed:
        return _HTTP_CLIENT
    if _SSL_CONTEXT is None:
        _SSL_CONTEXT = ssl.create_default_context()
    _HTTP_CLIENT = httpx.AsyncClient(
        verify=_SSL_CONTEXT,
        timeout=httpx.Timeout(60.0, connect=10.0),
        limits=httpx.Limits(max_connections=32, max_keepalive_connections=8),
    )
    return _HTTP_CLIENT


def _persona_text(persona: Dict[str, Any]) -> str:
    """把 persona 配置拼成 system message 的前半段（人设）。"""
    from .config import persona_prompt

    return persona_prompt(persona or {})


# 首包对冲：上游偶发 20~120 秒不回包（实测 974 次 xiaozhi 请求里 47 次），
# 而客户端 15 秒就放弃、孩子只能听到兜底话术。首包超过这个秒数就并发再开一路，
# 两路各自按网关轮询选号，同时卡住的概率远小于一路。
HEDGE_DELAY_SECONDS = 3.5
# 对冲之后还愿意再等多久（整体首包预算），超了就兜底。
HEDGE_BUDGET_SECONDS = 12.0
# 非流式调用（试一句 / 记忆抽取）的对冲节奏：正常整轮 2~4 秒，等 4 秒还没回就再开一路
HEDGE_DELAY_COMPLETE = 4.0
HEDGE_BUDGET_COMPLETE = 12.0


class LLMClient:
    """OpenAI 兼容 chat/completions 客户端。"""

    def __init__(self, preset: Dict[str, Any], persona: Dict[str, Any] | None = None):
        self.preset = preset
        self.base_url = (preset.get("base_url") or "").rstrip("/")
        self.api_key = preset.get("api_key") or ""
        self.model = preset.get("model") or "echo"
        self.temperature = float(preset.get("temperature", 0.7) or 0.7)
        self.max_tokens = int(preset.get("max_tokens", 512) or 512)
        # persona 可以显式传入，也可以跟着 preset 一起带下来
        self.persona = persona if isinstance(persona, dict) else (preset.get("persona") or {})
        self.persona_prompt = _persona_text(self.persona)
        self.system_prompt = preset.get("system_prompt") or ""
        self.history_turns = int(preset.get("history_turns", 8) or 8)
        self._client = None
        self.is_echo = (not self.base_url) or self.model == "echo"
        # 工具调用：tool_names 为空表示不给模型工具（保持纯聊天行为）
        self.tool_names: List[str] = list(preset.get("tools") or [])
        self.max_tool_rounds = int(preset.get("max_tool_rounds", 3) or 3)
        # 意图快速通道：工具结果本身就是要说的话时，跳过第二次模型调用直接播报
        self.direct_reply = bool(preset.get("direct_reply", True))
        # 思考强度：low / medium / high / none，空字符串表示不传（用服务端默认）
        self.reasoning_effort = str(preset.get("reasoning_effort") or "").strip()
        self.extra_body = preset.get("extra_body") or {}
        # 首包对冲延迟：<=0 关闭（配置项 hedge_delay_seconds，默认 3.5 秒）
        self.hedge_delay = float(preset.get("hedge_delay_seconds", HEDGE_DELAY_SECONDS) or 0)
        self._tool_ctx: Dict[str, Any] = {}
        # 本轮真正调用过的工具名（chat/chat_stream 每次开头重置），
        # 会话层用它判断「模型是不是只嘴上答应了、其实没动手」。
        self.last_tool_names: List[str] = []
        # 长期记忆段落：由会话层在每次对话前注入
        self.memory_block = ""

    def set_memory_block(self, block: str) -> None:
        """注入长期记忆段落（为空则等价于没有记忆）。"""
        self.memory_block = str(block or "").strip()

    def build_system_message(self) -> str:
        """人设在前，长期记忆次之，然后是自定义提示词，最后是工具使用规则。"""
        parts = [
            p
            for p in (
                self.persona_prompt,
                self.memory_block,
                self.system_prompt.strip(),
            )
            if p
        ]
        rule = self._tool_rule()
        if rule:
            parts.append(rule)
        parts.append(now_block())
        return "\n\n".join(parts)

    async def complete(
        self,
        system: str,
        user: str,
        max_tokens: int = 600,
        temperature: float = 0.2,
        timeout: float = 30.0,
    ) -> str:
        """一次性补全（给记忆抽取这类后台任务用）：不带工具、不注入人设与记忆。"""
        if self.is_echo:
            return ""
        client = self._ensure_client()
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": str(system or "")},
                {"role": "user", "content": str(user or "")},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        self._apply_reasoning(kwargs)
        budget = min(float(timeout or HEDGE_BUDGET_COMPLETE), HEDGE_BUDGET_COMPLETE)
        resp = await self._call_hedged(
            lambda: client.chat.completions.create(**kwargs),
            min(HEDGE_DELAY_COMPLETE, budget / 3.0),
            budget,
        )
        return (resp.choices[0].message.content or "").strip()

    def _tool_rule(self) -> str:
        """生成工具使用规则，逼模型在时效性/事实性问题上先查再答。"""
        if not self.tool_names:
            return ""
        has_search = "web_search" in self.tool_names
        lines = ["【工具使用规则】"]
        if has_search:
            lines.append(
                "遇到以下情况必须先用 web_search 查证，再根据结果回答，不要凭记忆猜："
                "①新闻、时事、比赛结果、发布会、最新型号；②具体日期、数字、价格；"
                "③你不确定或可能过时的事实；④孩子明确说「查一下」「搜一下」。"
            )
            lines.append(
                "搜索后要用自己的话把结果讲给孩子听，不要说网址、不要说「根据搜索结果」这类词。"
                "如果搜索没查到，就直说没查到，不要编。"
            )
        if "play_music" in self.tool_names:
            lines.append(
                "孩子说「播放……」「放一首……」「我想听……」「换首歌」「继续放」时，"
                "必须调用 play_music，把歌名或歌手原样作为 keyword 传进去。"
                "不许只回答「这就给你放」却不调用工具——那样设备根本不会出声。"
                "play_music 自己会说「好，给你放……」，所以调完后你只补一句很短的话，别重复歌名。"
                "没搜到就照实说没找到，不要编歌名。"
            )
        lines.append(
            "简单常识、聊天、成语接龙、口算、背诗、讲故事这类，直接用对应工具或直接回答，不必搜索。"
        )
        return "\n".join(lines)

    def _apply_reasoning(self, kwargs: Dict[str, Any]) -> None:
        """把 reasoning_effort / extra_body 合进请求参数（都没配就什么都不做）。"""
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
        if isinstance(self.extra_body, dict) and self.extra_body:
            merged = dict(kwargs.get("extra_body") or {})
            merged.update(self.extra_body)
            kwargs["extra_body"] = merged

    def _ensure_client(self):
        if self._client is None:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(
                base_url=self.base_url or "http://127.0.0.1:1/v1",
                api_key=self.api_key or "none",
                # 交互轮超时：原来 60 秒 + 1 次重试 = 孩子要干等 122 秒才听到一句兜底。
                # 正常首包 2 秒级，15 秒已是 7 倍余量；不重试，失败就快速兜底。
                timeout=15.0,
                max_retries=0,
                # 复用进程级 httpx 客户端：openai SDK 每次构造都会新建 httpx.AsyncClient，
                # 而 httpx 在 verify=True 时会重新解析 certifi 证书链，本机实测约 4 秒。
                # 复用后首次只付一次，且请求之间还能复用 TCP 连接。
                http_client=_shared_http_client(),
            )
        return self._client

    async def _call_hedged(self, factory, delay: float, budget: float):
        """非流式请求的首包对冲：delay 秒没回来就并发再开一路，先成功的赢。

        上游偶发「完全不回包」时，单路要等 SDK 超时才失败；两路请求走网关
        轮询选号（无会话键时 _rotate_pick），大概率落在不同账号上。
        """
        tasks = [asyncio.create_task(factory())]
        try:
            done, _ = await asyncio.wait(
                tasks, timeout=delay, return_when=asyncio.FIRST_COMPLETED
            )
            if not done:
                log.warning("响应 %.1f 秒未到，并发再开一路（对冲）", delay)
                tasks.append(asyncio.create_task(factory()))
                done, _ = await asyncio.wait(
                    tasks, timeout=max(3.0, budget - delay), return_when=asyncio.FIRST_COMPLETED
                )
            if not done:
                raise asyncio.TimeoutError("对冲超时：两路请求都没有响应")
            winner = None
            for task in done:
                if not task.cancelled() and task.exception() is None:
                    winner = task
                    break
            if winner is None:
                for task in done:
                    exc = task.exception()
                    if exc is not None:
                        raise exc
                raise asyncio.TimeoutError("对冲请求失败")
            return winner.result()
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()

    async def _open_stream_once(self, client, kwargs: Dict[str, Any]):
        """开一路流并把第一个 chunk 拉到手，返回 (stream, iterator, chunk)。"""
        stream = await client.chat.completions.create(**kwargs)
        iterator = stream.__aiter__()
        try:
            first = await iterator.__anext__()
        except StopAsyncIteration:
            first = None
        return stream, iterator, first

    @staticmethod
    async def _close_stream(stream) -> None:
        close = getattr(stream, "close", None)
        if close is None:
            return
        try:
            await close()
        except Exception:
            pass

    async def _stream_chunks(self, client, kwargs: Dict[str, Any]):
        """带首包对冲的流式迭代：慢的那一路被丢弃，先到的那一路说话。

        网关对同一个 key 是无亲和轮询选号（不带会话键走 _rotate_pick），
        所以两路请求大概率落在不同账号上；一路被上游卡住时另一路还能按时回。
        """
        if self.hedge_delay <= 0:
            stream, iterator, first = await self._open_stream_once(client, kwargs)
            async for chunk in self._iterate_stream(stream, iterator, first):
                yield chunk
            return

        tasks = [asyncio.create_task(self._open_stream_once(client, kwargs))]
        extra_streams = []
        try:
            done, _ = await asyncio.wait(
                tasks, timeout=self.hedge_delay, return_when=asyncio.FIRST_COMPLETED
            )
            if not done:
                log.warning("首包 %.1f 秒未到，并发再开一路（对冲）", self.hedge_delay)
                tasks.append(asyncio.create_task(self._open_stream_once(client, kwargs)))
                budget = max(3.0, HEDGE_BUDGET_SECONDS - self.hedge_delay)
                done, _ = await asyncio.wait(
                    tasks, timeout=budget, return_when=asyncio.FIRST_COMPLETED
                )
            if not done:
                raise asyncio.TimeoutError("首包超时：两路请求都没有响应")

            winner = None
            for task in done:
                if task.cancelled():
                    continue
                if task.exception() is not None:
                    continue
                winner = task
                break
            if winner is None:
                for task in done:
                    exc = task.exception()
                    if exc is not None:
                        raise exc
                raise asyncio.TimeoutError("首包失败")

            for task in tasks:
                if task is winner:
                    continue
                if task.done() and not task.cancelled() and task.exception() is None:
                    extra_streams.append(task.result()[0])
                else:
                    task.cancel()

            stream, iterator, first = winner.result()
            async for chunk in self._iterate_stream(stream, iterator, first):
                yield chunk
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            for stream in extra_streams:
                await self._close_stream(stream)

    async def _iterate_stream(self, stream, iterator, first):
        """先吐出已经拉到的首包，再继续迭代，结束时关掉这一路。"""
        try:
            if first is not None:
                yield first
            async for chunk in iterator:
                yield chunk
        finally:
            await self._close_stream(stream)

    def set_tool_context(self, ctx: Dict[str, Any]) -> None:
        """注入工具运行上下文（hub / session 等），由会话层在每次调用前设置。"""
        self._tool_ctx = dict(ctx or {})

    def _openai_tools(self) -> List[Dict[str, Any]]:
        """按配置的工具名单生成 OpenAI tools 参数；空则返回 []。"""
        if not self.tool_names:
            return []
        from .tools import to_openai_tools

        return to_openai_tools(self.tool_names)

    async def chat(self, history: List[Dict[str, str]], user_text: str) -> str:
        """history 为 [{role, content}]，返回助手回复文本。

        配了 tools 时走 function calling 循环：模型要调工具就执行、
        把结果回灌，直到它给出最终文本或轮数用尽。
        """
        if self.is_echo:
            return f"你说的是：{user_text}" if user_text else "我在。"

        self.last_tool_names = []
        messages: List[Dict[str, Any]] = []
        system_text = self.build_system_message()
        if system_text:
            messages.append({"role": "system", "content": system_text})
        keep = max(0, self.history_turns * 2)
        if keep:
            messages.extend(history[-keep:])
        messages.append({"role": "user", "content": user_text})

        client = self._ensure_client()
        tools = self._openai_tools()
        try:
            for round_no in range(max(1, self.max_tool_rounds)):
                kwargs: Dict[str, Any] = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                }
                if tools:
                    kwargs["tools"] = tools
                    kwargs["tool_choice"] = "auto"
                self._apply_reasoning(kwargs)

                resp = await self._call_hedged(
                    lambda: client.chat.completions.create(**kwargs),
                    HEDGE_DELAY_COMPLETE,
                    HEDGE_BUDGET_COMPLETE,
                )
                choice = resp.choices[0]
                msg = choice.message
                calls = getattr(msg, "tool_calls", None)

                # 没有工具调用，直接返回文本
                if not calls:
                    text = (msg.content or "").strip()
                    return text or "我没想好怎么回答。"

                self.last_tool_names.extend(c.function.name for c in calls)
                # 把 assistant 的 tool_calls 原样回灌，保持协议完整
                messages.append(
                    {
                        "role": "assistant",
                        "content": msg.content or "",
                        "tool_calls": [c.model_dump() for c in calls],
                    }
                )
                from .tools import call_tool

                direct_text = ""
                for call in calls:
                    name = call.function.name
                    raw = call.function.arguments or "{}"
                    try:
                        import json

                        args = json.loads(raw) if raw.strip() else {}
                        if not isinstance(args, dict):
                            args = {}
                    except Exception:
                        args = {}
                    # 注入运行上下文（hub/session），工具实现按需取用
                    if self._tool_ctx:
                        args = {**args, "_ctx": self._tool_ctx}
                    log.info("调用工具 %s(%s)", name, raw[:120])
                    result = await call_tool(name, args)
                    if (
                        len(calls) == 1
                        and self.direct_reply
                        and result.get("direct")
                        and not result.get("is_error")
                    ):
                        direct_text = str(result.get("text") or "").strip()
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.id,
                            "content": result.get("text") or "",
                        }
                    )
                if direct_text:
                    log.info("快速通道：工具结果直接播报，跳过模型第二轮")
                    return direct_text
                log.info("工具调用完成第 %d 轮，继续生成回复", round_no + 1)

            # 轮数用尽仍要工具：强制要一次纯文本回答
            kwargs2 = {
                "model": self.model,
                "messages": messages
                + [{"role": "user", "content": "请直接回答，不要再调用工具。"}],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }
            self._apply_reasoning(kwargs2)
            resp2 = await self._call_hedged(
                lambda: client.chat.completions.create(**kwargs2),
                HEDGE_DELAY_COMPLETE,
                HEDGE_BUDGET_COMPLETE,
            )
            text = (resp2.choices[0].message.content or "").strip()
            return text or "我没想好怎么回答。"
        except Exception as exc:
            log.error("LLM 调用失败 (%s @ %s): %s", self.model, self.base_url, exc)
            return "模型服务这次没响应，再点一次通常就好。"


    async def chat_stream(self, history: List[Dict[str, Any]], user_text: str):
        """流式生成：工具轮与最终回复全程流式，尽早吐出第一个字。

        旧实现先用「非流式」跑工具轮，再流式跑最终回复——模型不调工具时
        那次非流式调用本身就是全部首字延迟（实测 6.4s vs 流式 1.9s）。
        现在每一轮都开 stream：正文增量立刻 yield，tool_calls 增量边收边拼，
        多工具并行执行。
        """
        self.last_tool_names = []
        if self.is_echo:
            reply = f"你说的是：{user_text}" if user_text else "我在。"
            for ch in reply:
                yield ch
            return

        messages: List[Dict[str, Any]] = []
        system_text = self.build_system_message()
        if system_text:
            messages.append({"role": "system", "content": system_text})
        keep = max(0, self.history_turns * 2)
        if keep:
            messages.extend(history[-keep:])
        messages.append({"role": "user", "content": user_text})

        client = self._ensure_client()
        tools = self._openai_tools()
        try:
            for round_no in range(max(1, self.max_tool_rounds)):
                kwargs: Dict[str, Any] = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    "stream": True,
                }
                if tools:
                    kwargs["tools"] = tools
                    kwargs["tool_choice"] = "auto"
                self._apply_reasoning(kwargs)

                content_parts: List[str] = []
                slots: Dict[int, Dict[str, str]] = {}
                async for chunk in self._stream_chunks(client, kwargs):
                    if not getattr(chunk, "choices", None):
                        continue
                    delta = chunk.choices[0].delta
                    piece = getattr(delta, "content", None)
                    if piece:
                        content_parts.append(piece)
                        yield piece
                    for tc in getattr(delta, "tool_calls", None) or []:
                        idx = getattr(tc, "index", 0) or 0
                        slot = slots.setdefault(idx, {"id": "", "name": "", "args": ""})
                        if getattr(tc, "id", None):
                            slot["id"] = tc.id
                        fn = getattr(tc, "function", None)
                        if fn is not None:
                            if getattr(fn, "name", None):
                                slot["name"] += fn.name
                            if getattr(fn, "arguments", None):
                                slot["args"] += fn.arguments

                if not slots:
                    # 没有工具调用：正文已在上面边流边 yield，本轮结束即收工
                    return

                tool_calls = [
                    {
                        "id": slot["id"] or f"call_{round_no}_{idx}",
                        "type": "function",
                        "function": {
                            "name": slot["name"],
                            "arguments": slot["args"] or "{}",
                        },
                    }
                    for idx, slot in sorted(slots.items())
                ]
                messages.append(
                    {
                        "role": "assistant",
                        "content": "".join(content_parts),
                        "tool_calls": tool_calls,
                    }
                )
                log.info(
                    "工具轮 %d：%s",
                    round_no + 1,
                    ",".join(c["function"]["name"] for c in tool_calls),
                )
                self.last_tool_names.extend(
                    c["function"]["name"] for c in tool_calls
                )
                from .tools import call_tool

                async def _exec(call: Dict[str, Any]) -> Dict[str, Any]:
                    name = call["function"]["name"]
                    raw = call["function"]["arguments"] or "{}"
                    try:
                        args = json.loads(raw) if raw.strip() else {}
                        if not isinstance(args, dict):
                            args = {}
                    except Exception:
                        args = {}
                    if self._tool_ctx:
                        args = {**args, "_ctx": self._tool_ctx}
                    log.info("调用工具 %s(%s)", name, raw[:120])
                    result = await call_tool(name, args)
                    return result if isinstance(result, dict) else {"text": str(result)}

                # 多个工具并行执行，省掉串行等待
                results = await asyncio.gather(
                    *(_exec(c) for c in tool_calls), return_exceptions=True
                )
                direct_text = ""
                for pos, (call, res) in enumerate(zip(tool_calls, results)):
                    if isinstance(res, BaseException):
                        log.warning("工具 %s 失败: %s", call["function"]["name"], res)
                        res = {"text": "工具执行失败", "is_error": True}
                    if (
                        pos == 0
                        and len(tool_calls) == 1
                        and self.direct_reply
                        and res.get("direct")
                        and not res.get("is_error")
                    ):
                        direct_text = str(res.get("text") or "").strip()
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call["id"],
                            "content": res.get("text") or "",
                        }
                    )
                if direct_text:
                    # 快速通道：省掉「把工具结果再喂回模型」的那一轮（实测 1.0~1.7s）
                    log.info("快速通道：工具结果直接播报，跳过模型第二轮")
                    yield direct_text
                    return

            # 轮数用尽仍要工具：加一条约束再流式收尾
            messages.append({"role": "user", "content": "请直接回答，不要再调用工具。"})
            skwargs: Dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "stream": True,
            }
            self._apply_reasoning(skwargs)
            async for chunk in self._stream_chunks(client, skwargs):
                if not getattr(chunk, "choices", None):
                    continue
                piece = getattr(chunk.choices[0].delta, "content", None)
                if piece:
                    yield piece
        except Exception as exc:
            log.error("LLM 流式调用失败 (%s @ %s): %s", self.model, self.base_url, exc)
            yield "我这会儿有点忙，你再说一遍好不好？"


def build_llm(preset: Dict[str, Any], persona: Dict[str, Any] | None = None) -> LLMClient:
    return LLMClient(preset, persona)


def strip_for_speech(text: str) -> str:
    """把不适合朗读的 markdown 符号清掉。"""
    import re

    t = text or ""
    t = re.sub(r"```.*?```", "", t, flags=re.S)
    t = re.sub(r"`([^`]*)`", r"\1", t)
    t = re.sub(r"[*_#>]+", "", t)
    t = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", t)
    t = re.sub(r"^\s*[-•]\s*", "", t, flags=re.M)
    t = re.sub(r"\n{2,}", "\n", t)
    return t.strip()
