"""学习激励（习惯养成）小程序客户端。

把腾讯云上自建的「习惯养成 - 亲子任务奖励系统」（xuexi）接进小智：
查积分、看今天还剩哪些任务、申请完成任务、申请兑换奖励。

鉴权走 POST /api/auth/login/device，靠孩子账号的 openid（device_id）换 JWT，
不需要陪伴密码；token 过期用 refreshToken 续，续不上就重新登录。
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

log = logging.getLogger("study")

DEFAULT_BASE_URL = "https://xuexi.fengs.fun"
DEFAULT_TIMEOUT = 8.0


class StudyError(Exception):
    """学习激励接口调用失败（网络或业务错误）。"""


def _cfg() -> Dict[str, Any]:
    """读取 config.yaml 的 study 段；读不到就给安全默认值。"""
    from .config import load_config

    try:
        cfg = load_config()
    except Exception as exc:  # pragma: no cover
        log.warning("读取配置失败：%s", exc)
        return {}
    section = cfg.get("study") or {}
    return section if isinstance(section, dict) else {}


class StudyClient:
    """xuexi 后端的最小客户端（只用到孩子端的几个接口）。"""

    def __init__(self, base_url: str = "", device_id: str = "", timeout: float = 0.0):
        self.base_url = (base_url or _cfg().get("base_url") or DEFAULT_BASE_URL).rstrip("/")
        self.device_id = device_id or str(_cfg().get("device_id") or "")
        self.timeout = float(timeout or _cfg().get("timeout") or DEFAULT_TIMEOUT)
        self._token: str = ""
        self._refresh: str = ""
        self._user: Dict[str, Any] = {}
        self._lock = asyncio.Lock()

    # ---------- 底层请求 ----------
    async def _raw(self, method: str, path: str, payload: Optional[Dict[str, Any]] = None,
                   token: str = "") -> Dict[str, Any]:
        import httpx

        headers = {"Accept": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        url = f"{self.base_url}{path}"
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.request(method, url, json=payload, headers=headers)
        except Exception as exc:
            raise StudyError(f"连不上学习激励服务：{exc}") from exc
        if resp.status_code >= 500:
            raise StudyError(f"学习激励服务出错（{resp.status_code}）")
        try:
            data = resp.json()
        except Exception as exc:
            raise StudyError("学习激励服务返回的不是 JSON") from exc
        if not isinstance(data, dict):
            raise StudyError("学习激励服务返回格式异常")
        return data

    async def _login(self) -> None:
        if not self.device_id:
            raise StudyError("没配置孩子账号（study.device_id）")
        data = await self._raw("POST", "/api/auth/login/device", {"deviceId": self.device_id})
        if data.get("code") != 0:
            raise StudyError(data.get("message") or "孩子账号登录失败")
        body = data.get("data") or {}
        self._token = str(body.get("token") or "")
        self._refresh = str(body.get("refreshToken") or "")
        self._user = body.get("user") or {}
        if not self._token:
            raise StudyError("孩子账号登录没有拿到 token")

    async def _refresh_token(self) -> bool:
        if not self._refresh:
            return False
        try:
            data = await self._raw("POST", "/api/auth/refresh", {"refreshToken": self._refresh})
        except StudyError:
            return False
        if data.get("code") != 0:
            return False
        body = data.get("data") or {}
        token = str(body.get("token") or "")
        if not token:
            return False
        self._token = token
        self._refresh = str(body.get("refreshToken") or self._refresh)
        return True

    async def _call(self, method: str, path: str, payload: Optional[Dict[str, Any]] = None) -> Any:
        """带自动登录/续期的业务请求，返回 data 字段。"""
        async with self._lock:
            if not self._token:
                await self._login()
            data = await self._raw(method, path, payload, token=self._token)
            code = data.get("code")
            if code in (1002, 1003, 401):  # 未登录 / token 过期
                ok = await self._refresh_token()
                if not ok:
                    await self._login()
                data = await self._raw(method, path, payload, token=self._token)
                code = data.get("code")
            if code != 0:
                raise StudyError(str(data.get("message") or "调用失败"))
            return data.get("data")

    # ---------- 业务接口 ----------
    async def me(self) -> Dict[str, Any]:
        data = await self._call("GET", "/api/auth/me")
        if isinstance(data, dict) and data:
            self._user = data
        return self._user if not data else data

    async def tasks(self) -> List[Dict[str, Any]]:
        """今天的任务清单（daily / weekly / once 拍平）。"""
        data = await self._call("GET", "/api/tasks") or {}
        out: List[Dict[str, Any]] = []
        if isinstance(data, dict):
            for key in ("daily", "weekly", "once"):
                for item in (data.get(key) or []):
                    if isinstance(item, dict):
                        out.append(item)
        elif isinstance(data, list):
            out = [x for x in data if isinstance(x, dict)]
        return out

    async def rewards(self) -> List[Dict[str, Any]]:
        data = await self._call("GET", "/api/rewards") or []
        return [x for x in data if isinstance(x, dict)] if isinstance(data, list) else []

    async def complete_task(self, task_id: str) -> Dict[str, Any]:
        data = await self._call("POST", f"/api/tasks/{task_id}/complete") or {}
        return data if isinstance(data, dict) else {}

    async def exchange(self, reward_id: str) -> Dict[str, Any]:
        data = await self._call("POST", "/api/exchanges", {"rewardId": reward_id}) or {}
        return data if isinstance(data, dict) else {}


_CLIENT: Optional[StudyClient] = None


def client() -> StudyClient:
    """进程内单例，复用 token。"""
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = StudyClient()
    return _CLIENT


def reset() -> None:
    """配置改动后丢弃旧客户端（重新读 config + 重新登录）。"""
    global _CLIENT
    _CLIENT = None
