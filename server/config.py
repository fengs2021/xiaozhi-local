"""配置加载与热更新。"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.yaml"

# 人设默认值：老配置没有 persona 段时自动补齐
DEFAULT_PERSONA: Dict[str, str] = {
    "assistant_name": "小智",
    "user_name": "主人",
    "language": "zh",
    "character": "",
    "speech_speed": "+0%",
    "pitch": "+0Hz",
}

# 天气工具默认城市
DEFAULT_WEATHER_CITY = "上海"

# 使用限制默认值（儿童场景保护）：0 一律表示不限制
DEFAULT_LIMITS: Dict[str, Any] = {
    "max_output_chars_per_day": 3000,  # 单台设备每天最多听到多少字回复
    "idle_bye_seconds": 120,           # 多久没说话就主动告别并断开
    "wake_guard_seconds": 2.0,         # 开始聆听后多少秒内做「短段免疫」
    "wake_guard_min_ms": 800,          # 免疫期内短于这个长度的语音段直接丢弃
}


def _all_tool_names() -> list:
    """取内置工具全名列表（延迟导入避免循环依赖）。"""
    try:
        from .tools import TOOL_NAMES

        return list(TOOL_NAMES)
    except Exception:  # pragma: no cover - 工具模块异常时退回固定列表
        return ["get_time", "get_weather", "calculator", "get_device_status"]


def _mask(key: str) -> str:
    if not key:
        return ""
    if len(key) <= 8:
        return "****"
    return key[:4] + "****" + key[-4:]


def load_config(path: Path | None = None) -> Dict[str, Any]:
    """读取 config.yaml，并叠加环境变量覆盖。"""
    p = Path(path) if path else CONFIG_PATH
    cfg: Dict[str, Any] = {}
    if p.exists():
        cfg = yaml.safe_load(p.read_text(encoding="utf-8")) or {}

    cfg.setdefault("server", {})
    cfg.setdefault("ota", {})
    cfg.setdefault("llm", {})
    cfg.setdefault("asr", {})
    cfg.setdefault("tts", {})
    cfg.setdefault("vad", {})
    cfg.setdefault("audio", {})
    cfg.setdefault("limits", {})
    cfg["llm"].setdefault("presets", {})
    cfg["llm"].setdefault("active", "echo")

    # 人设段：缺字段逐个补默认值，保证老配置能平滑升级
    persona = cfg.get("persona")
    if not isinstance(persona, dict):
        persona = {}
    for key, default in DEFAULT_PERSONA.items():
        value = persona.get(key)
        persona[key] = default if value is None else str(value).strip()
    cfg["persona"] = persona

    # MCP 段：enabled_tools 缺省为全部内置工具
    mcp = cfg.get("mcp")
    if not isinstance(mcp, dict):
        mcp = {}
    tools = mcp.get("enabled_tools")
    if not isinstance(tools, list) or not tools:
        tools = _all_tool_names()
    mcp["enabled_tools"] = [str(x).strip() for x in tools if str(x).strip()]
    mcp["weather_city"] = str(mcp.get("weather_city") or DEFAULT_WEATHER_CITY).strip()
    cfg["mcp"] = mcp

    # 使用限制：缺字段逐个补默认值，保证老配置平滑升级
    limits = cfg.get("limits")
    if not isinstance(limits, dict):
        limits = {}
    for key, default in DEFAULT_LIMITS.items():
        raw = limits.get(key, default)
        try:
            limits[key] = type(default)(raw)
        except (TypeError, ValueError):
            limits[key] = default
    cfg["limits"] = limits

    # 环境变量覆盖
    if os.environ.get("XIAOZHI_LLM_BASE_URL"):
        cfg["llm"].setdefault("presets", {}).setdefault(cfg["llm"]["active"], {})[
            "base_url"
        ] = os.environ["XIAOZHI_LLM_BASE_URL"]
    if os.environ.get("XIAOZHI_LLM_MODEL"):
        cfg["llm"].setdefault("presets", {}).setdefault(cfg["llm"]["active"], {})[
            "model"
        ] = os.environ["XIAOZHI_LLM_MODEL"]

    cfg["_root"] = str(ROOT)
    return cfg


def save_config(cfg: Dict[str, Any], path: Path | None = None) -> None:
    """把配置写回 config.yaml（去掉内部字段）。"""
    p = Path(path) if path else CONFIG_PATH
    data = {k: v for k, v in cfg.items() if not k.startswith("_")}
    p.write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False, width=200),
        encoding="utf-8",
    )


def resolve_path(cfg: Dict[str, Any], value: str) -> str:
    """把配置里的相对路径解析成绝对路径。"""
    if not value:
        return value
    p = Path(value)
    if p.is_absolute():
        return str(p)
    return str(Path(cfg["_root"]) / value)


def active_preset(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """取当前生效的 LLM preset（合并环境变量里的 key）。"""
    llm = cfg.get("llm", {})
    presets = llm.get("presets", {}) or {}
    name = llm.get("active") or next(iter(presets), "echo")
    preset = dict(presets.get(name) or {})
    preset.setdefault("name", name)
    preset.setdefault("base_url", "")
    preset.setdefault("model", "echo")
    preset.setdefault("temperature", 0.7)
    preset.setdefault("max_tokens", 512)
    if not preset.get("api_key"):
        preset["api_key"] = os.environ.get("XIAOZHI_LLM_API_KEY", "")
    preset["system_prompt"] = llm.get("system_prompt", "")
    preset["history_turns"] = int(llm.get("history_turns", 8) or 8)
    preset["persona"] = public_persona_view(cfg)
    return preset


def public_llm_view(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """给网页用的 LLM 配置视图（api_key 打码）。"""
    llm = cfg.get("llm", {})
    presets = {}
    for name, p in (llm.get("presets") or {}).items():
        q = dict(p or {})
        q["api_key"] = _mask(q.get("api_key", ""))
        presets[name] = q
    return {
        "active": llm.get("active", ""),
        "system_prompt": llm.get("system_prompt", ""),
        "history_turns": llm.get("history_turns", 8),
        "presets": presets,
    }

def sanitize_preset(data: Dict[str, Any]) -> Dict[str, Any]:
    """把前端提交的预设字段规范化。"""
    out: Dict[str, Any] = {}
    out["base_url"] = str(data.get("base_url") or "").strip()
    out["model"] = str(data.get("model") or "").strip()
    out["api_key"] = str(data.get("api_key") or "").strip()
    try:
        out["temperature"] = max(0.0, min(2.0, float(data.get("temperature", 0.7))))
    except (TypeError, ValueError):
        out["temperature"] = 0.7
    try:
        out["max_tokens"] = max(16, min(32768, int(data.get("max_tokens", 512))))
    except (TypeError, ValueError):
        out["max_tokens"] = 512
    # 思考强度：只接受已知档位，其余当空（不传）
    effort = str(data.get("reasoning_effort") or "").strip().lower()
    out["reasoning_effort"] = effort if effort in ("low", "medium", "high", "none") else ""
    eb = data.get("extra_body")
    out["extra_body"] = eb if isinstance(eb, dict) else {}
    return out


def upsert_preset(
    cfg: Dict[str, Any], name: str, preset: Dict[str, Any], old_name: str = ""
) -> Dict[str, Any]:
    """新增/更新一个预设；old_name 非空表示重命名。"""
    llm = cfg.setdefault("llm", {})
    presets = llm.setdefault("presets", {})
    name = (name or "").strip()
    if not name:
        raise ValueError("预设名称不能为空")
    old_name = (old_name or "").strip()
    src = old_name or name
    cur = dict(presets.get(src) or {})
    newp = sanitize_preset(preset)
    # api_key 里出现 * 说明是打码回显值，保留原 key
    if "*" in newp.get("api_key", ""):
        newp["api_key"] = cur.get("api_key", "")
    cur.update(newp)
    if old_name and old_name != name:
        presets.pop(old_name, None)
        if llm.get("active") == old_name:
            llm["active"] = name
    presets[name] = cur
    if not llm.get("active") or llm["active"] not in presets:
        llm["active"] = name
    return cur


def delete_preset(cfg: Dict[str, Any], name: str) -> None:
    llm = cfg.setdefault("llm", {})
    presets = llm.setdefault("presets", {})
    if name not in presets:
        raise ValueError("预设不存在")
    if len(presets) <= 1:
        raise ValueError("至少要保留一个预设")
    presets.pop(name)
    if llm.get("active") == name:
        llm["active"] = next(iter(presets))


def set_active(cfg: Dict[str, Any], name: str) -> None:
    presets = cfg.setdefault("llm", {}).setdefault("presets", {})
    if name not in presets:
        raise ValueError("预设不存在")
    cfg["llm"]["active"] = name


# ---------- 人设 ----------
def sanitize_persona(data: Dict[str, Any]) -> Dict[str, str]:
    """把前端提交的人设字段规范化（全部转 str 并 strip）。"""
    src = data if isinstance(data, dict) else {}
    out: Dict[str, str] = {}
    for key, default in DEFAULT_PERSONA.items():
        value = src.get(key)
        text = "" if value is None else str(value).strip()
        out[key] = text if text else default
    # 语速/音调必须是 edge-tts 认识的格式，非法值回退默认，避免合成直接报错
    if not _looks_like_percent(out["speech_speed"]):
        out["speech_speed"] = DEFAULT_PERSONA["speech_speed"]
    if not _looks_like_hz(out["pitch"]):
        out["pitch"] = DEFAULT_PERSONA["pitch"]
    return out


def _looks_like_percent(value: str) -> bool:
    """形如 +0% / -20% / 10% 的语速值。"""
    v = (value or "").strip()
    if not v.endswith("%"):
        return False
    try:
        float(v[:-1])
    except ValueError:
        return False
    return True


def _looks_like_hz(value: str) -> bool:
    """形如 +0Hz / -20Hz / 10Hz 的音调值。"""
    v = (value or "").strip()
    if not v.lower().endswith("hz"):
        return False
    try:
        float(v[:-2])
    except ValueError:
        return False
    return True


def public_persona_view(cfg: Dict[str, Any]) -> Dict[str, str]:
    """给网页和 LLM 用的人设视图（字段齐全，字符串已 strip）。"""
    persona = cfg.get("persona")
    if not isinstance(persona, dict):
        persona = {}
    return sanitize_persona(persona)


def persona_prompt(persona: Dict[str, Any]) -> str:
    """把 persona 拼成 system message 的前半段。"""
    p = sanitize_persona(persona if isinstance(persona, dict) else {})
    name = p["assistant_name"]
    user = p["user_name"]
    lines = [f"你叫{name}，称呼用户为{user}。"]
    character = p.get("character") or ""
    if character:
        lines.append(f"你的性格：{character}")
    if p.get("language") and p["language"].lower() not in ("zh", "zh-cn", "chinese", "中文"):
        lines.append(f"请优先使用语言代码 {p['language']} 与用户交流。")
    lines.append("说话要口语化、简短自然，像真人聊天一样。")
    return "\n".join(lines)


# ---------- MCP 工具开关 ----------
def enabled_tools(cfg: Dict[str, Any]) -> list:
    """取当前启用的工具名列表（缺省全开）。"""
    mcp = cfg.get("mcp") or {}
    tools = mcp.get("enabled_tools")
    if not isinstance(tools, list) or not tools:
        return _all_tool_names()
    return [str(x).strip() for x in tools if str(x).strip()]


def weather_city(cfg: Dict[str, Any]) -> str:
    """取天气工具默认城市。"""
    mcp = cfg.get("mcp") or {}
    return str(mcp.get("weather_city") or DEFAULT_WEATHER_CITY).strip()
