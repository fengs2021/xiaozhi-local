"""bedbook 睡前故事 MCP（stdio 传输）客户端。

fengs2021/bedbook 是个用 Node 写的 MCP 服务，自带 148 篇中文儿童故事，
对外只有两个工具：listStories / searchStory（searchStory 的 bestMatch.content
就是整篇正文）。

它的 HTTP 传输在本机实测不可用：initialize 返回 200，之后任何一次
tools/list 或 tools/call 都 500 且服务端无日志（不是我们这边请求写错，
无状态串行调用就是不支持）。所以这里走 stdio：起一个 node 子进程，
按行 JSON-RPC 通信，一次启动多次调用，省掉每次几百毫秒的 Node 冷启动。
"""
from __future__ import annotations

import asyncio
import json
import logging
import random
from typing import Any, Dict, List, Optional

log = logging.getLogger("bedbook")

DEFAULT_NODE = "/opt/node22/bin/node"
DEFAULT_PACKAGE_DIR = "/opt/node22/lib/node_modules/bedbook"
DEFAULT_TIMEOUT = 20.0
PROTOCOL_VERSION = "2025-06-18"
ENTRY = "dist/index.js"


class BedbookError(Exception):
    """故事 MCP 调用失败（进程没起来、超时、工具报错）。"""


def _cfg() -> Dict[str, Any]:
    """读取 config.yaml 的 bedbook 段；读不到就给安全默认值。"""
    from .config import load_config

    try:
        cfg = load_config()
    except Exception as exc:  # pragma: no cover
        log.warning("读取配置失败：%s", exc)
        return {}
    section = cfg.get("bedbook") or {}
    return section if isinstance(section, dict) else {}


class BedbookClient:
    """bedbook 故事服务的最小 MCP 客户端（stdio，进程内单例）。"""

    def __init__(self, node: str = "", package_dir: str = "", timeout: float = 0.0):
        cfg = _cfg()
        # node 不在 systemd 的 PATH 里，必须写绝对路径，否则子进程起不来
        self.node = node or str(cfg.get("node") or DEFAULT_NODE)
        self.package_dir = package_dir or str(cfg.get("package_dir") or DEFAULT_PACKAGE_DIR)
        self.timeout = float(timeout or cfg.get("timeout") or DEFAULT_TIMEOUT)
        self.enabled = cfg.get("enabled", True) is not False
        self._proc: Optional[asyncio.subprocess.Process] = None
        self._reader: Optional[asyncio.Task] = None
        self._err_reader: Optional[asyncio.Task] = None
        self._pending: Dict[int, asyncio.Future] = {}
        self._next_id = 0
        self._start_lock = asyncio.Lock()  # 只保护启动过程，调用之间可并发

    # ---------- 进程管理 ----------
    async def _ensure(self) -> None:
        if not self.enabled:
            raise BedbookError("故事服务未启用（config.yaml bedbook.enabled=false）")
        if self._proc is not None and self._proc.returncode is None:
            return
        async with self._start_lock:
            if self._proc is not None and self._proc.returncode is None:
                return
            await self._start()

    async def _start(self) -> None:
        try:
            self._proc = await asyncio.create_subprocess_exec(
                self.node,
                ENTRY,
                cwd=self.package_dir,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                limit=4 * 1024 * 1024,  # listStories 一次吐 25KB，留足余量
            )
        except Exception as exc:  # noqa: BLE001
            self._proc = None
            raise BedbookError(f"启动故事服务失败：{exc}") from exc
        self._pending = {}
        self._reader = asyncio.create_task(self._read_loop(), name="bedbook-read")
        self._err_reader = asyncio.create_task(self._drain_stderr(), name="bedbook-err")
        await self._request(
            "initialize",
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "xiaozhi-local", "version": "1.0"},
            },
        )
        await self._notify("notifications/initialized")
        log.info("故事服务已启动：%s", self.package_dir)

    async def _kill(self) -> None:
        proc, self._proc = self._proc, None
        for task in (self._reader, self._err_reader):
            if task is not None and not task.done():
                task.cancel()
        self._reader = self._err_reader = None
        if proc is not None and proc.returncode is None:
            try:
                proc.kill()
            except Exception:  # noqa: BLE001
                pass
            try:
                await asyncio.wait_for(proc.wait(), timeout=3.0)
            except Exception:  # noqa: BLE001
                pass
        for fut in list(self._pending.values()):
            if not fut.done():
                fut.set_exception(BedbookError("故事服务已退出"))
        self._pending = {}

    # ---------- 底层 JSON-RPC ----------
    async def _read_loop(self) -> None:
        proc = self._proc
        try:
            while proc is not None and proc.stdout is not None:
                line = await proc.stdout.readline()
                if not line:
                    break
                text = line.decode("utf-8", "replace").strip()
                if not text:
                    continue
                try:
                    msg = json.loads(text)
                except Exception:  # noqa: BLE001
                    log.debug("故事服务输出无法解析：%s", text[:200])
                    continue
                mid = msg.get("id")
                if mid is None:  # 服务端通知，忽略
                    continue
                fut = self._pending.pop(mid, None)
                if fut is None or fut.done():
                    continue
                if msg.get("error"):
                    fut.set_exception(BedbookError(str(msg.get("error"))))
                else:
                    fut.set_result(msg.get("result") or {})
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            log.warning("故事服务读取中断：%s", exc)
        finally:
            for fut in list(self._pending.values()):
                if not fut.done():
                    fut.set_exception(BedbookError("故事服务已退出"))
            self._pending = {}

    async def _drain_stderr(self) -> None:
        proc = self._proc
        if proc is None or proc.stderr is None:
            return
        try:
            while True:
                line = await proc.stderr.readline()
                if not line:
                    break
                log.debug("故事服务 stderr：%s", line.decode("utf-8", "replace").strip()[:300])
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            pass

    def _write(self, payload: Dict[str, Any]) -> None:
        proc = self._proc
        if proc is None or proc.stdin is None:
            raise BedbookError("故事服务没有启动")
        proc.stdin.write((json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8"))

    async def _notify(self, method: str, params: Optional[Dict[str, Any]] = None) -> None:
        payload: Dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            payload["params"] = params
        try:
            self._write(payload)
            proc = self._proc
            if proc is not None and proc.stdin is not None:
                await proc.stdin.drain()
        except Exception as exc:  # noqa: BLE001
            await self._kill()
            raise BedbookError(f"写入故事服务失败：{exc}") from exc

    async def _request(
        self, method: str, params: Optional[Dict[str, Any]] = None, timeout: float = 0.0
    ) -> Dict[str, Any]:
        self._next_id += 1
        mid = self._next_id
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending[mid] = fut
        payload: Dict[str, Any] = {"jsonrpc": "2.0", "id": mid, "method": method}
        if params is not None:
            payload["params"] = params
        try:
            self._write(payload)
            proc = self._proc
            if proc is not None and proc.stdin is not None:
                await proc.stdin.drain()
        except Exception as exc:  # noqa: BLE001
            self._pending.pop(mid, None)
            await self._kill()
            raise BedbookError(f"写入故事服务失败：{exc}") from exc
        try:
            return await asyncio.wait_for(fut, timeout or self.timeout)
        except asyncio.TimeoutError as exc:
            self._pending.pop(mid, None)
            await self._kill()
            raise BedbookError("故事服务响应超时") from exc

    # ---------- 工具调用 ----------
    async def call_tool(self, name: str, args: Optional[Dict[str, Any]] = None) -> Any:
        """调一个 MCP 工具，返回解析后的 JSON（解析不了就返回原文本）。"""
        await self._ensure()
        result = await self._request("tools/call", {"name": name, "arguments": args or {}})
        if result.get("isError"):
            raise BedbookError(str(result.get("content") or f"{name} 调用失败"))
        texts = [
            str(c.get("text") or "")
            for c in (result.get("content") or [])
            if isinstance(c, dict) and c.get("type") == "text"
        ]
        raw = "\n".join(texts).strip()
        if not raw:
            return None
        try:
            return json.loads(raw)
        except Exception:  # noqa: BLE001
            return raw

    # ---------- 业务封装 ----------
    async def stories(self) -> List[Dict[str, Any]]:
        """列出全部故事（元数据，不含正文）。"""
        data = await self.call_tool("listStories", {})
        items = data.get("stories") if isinstance(data, dict) else data
        if not isinstance(items, list):
            return []
        return [i for i in items if isinstance(i, dict) and str(i.get("title") or "").strip()]

    async def search(
        self, title: str = "", keyword: str = "", age: str = ""
    ) -> Optional[Dict[str, Any]]:
        """按名称/关键词找一篇故事，返回带正文的 bestMatch（找不到就是 None）。"""
        args: Dict[str, Any] = {}
        if title:
            args["title"] = title
        if keyword:
            args["keyword"] = keyword
        if age:
            args["age"] = age
        if not args:
            return None
        data = await self.call_tool("searchStory", args)
        best = data.get("bestMatch") if isinstance(data, dict) else None
        if isinstance(best, dict) and str(best.get("content") or "").strip():
            return best
        return None

    async def random_story(self) -> Optional[Dict[str, Any]]:
        """随机来一篇（孩子说「随便讲一个」时用）。"""
        items = await self.stories()
        if not items:
            return None
        random.shuffle(items)
        for item in items[:5]:
            try:
                story = await self.search(title=str(item.get("title") or ""))
            except BedbookError:
                raise
            if story is not None:
                return story
        return None

    async def close(self) -> None:
        await self._kill()


_CLIENT: Optional[BedbookClient] = None


def client() -> BedbookClient:
    """进程内单例：子进程和 Node 运行时都只建一次。"""
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = BedbookClient()
    return _CLIENT


def reset() -> None:
    """配置改动后丢弃旧客户端（下次调用重新拉起子进程）。"""
    global _CLIENT
    _CLIENT = None
