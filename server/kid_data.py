"""小学教育内容库的统一数据加载层。

数据文件全部在 <项目根>/data/ 下，构建脚本 _build_data.py 生成：
  idioms_dict.txt   45776 条四字成语，每行 word|pinyin|explanation
  poems.json        319 首古诗，{title, author, type, lines[]}
  words_index.json  9139 个字 -> 组词列表
  english_words.json 819 个人教小学单词
  stories.json      626 篇故事（伊索 426 + 格林 200）

所有加载都做了缓存与缺文件降级，缺文件时返回空结构由调用方兜底。
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Dict, List, Tuple

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"

_lock = threading.Lock()
_cache: Dict[str, Any] = {}


def _read_json(name: str, default: Any) -> Any:
    if name in _cache:
        return _cache[name]
    with _lock:
        if name in _cache:
            return _cache[name]
        path = _DATA_DIR / name
        try:
            with path.open("r", encoding="utf-8") as f:
                val = json.load(f)
        except (OSError, ValueError):
            val = default
        _cache[name] = val
        return val


def load_idioms() -> List[Tuple[str, str, str]]:
    """返回 [(成语, 拼音, 释义)]，按文件顺序。"""
    if "idioms" in _cache:
        return _cache["idioms"]
    out: List[Tuple[str, str, str]] = []
    try:
        text = (_DATA_DIR / "idioms_dict.txt").read_text(encoding="utf-8")
    except OSError:
        text = ""
    for line in text.splitlines():
        parts = line.split("|")
        if len(parts) < 3:
            continue
        w, py, ex = parts[0].strip(), parts[1].strip(), parts[2].strip()
        if len(w) == 4:
            out.append((w, py, ex))
    _cache["idioms"] = out
    return out


def load_poems() -> List[Dict[str, Any]]:
    return _read_json("poems.json", [])


def load_words() -> Dict[str, List[str]]:
    return _read_json("words_index.json", {})


def load_english() -> List[Dict[str, str]]:
    return _read_json("english_words.json", [])


def load_stories() -> List[Dict[str, str]]:
    return _read_json("stories.json", [])


def data_stats() -> Dict[str, int]:
    """给控制台展示用的数据规模。"""
    return {
        "idioms": len(load_idioms()),
        "poems": len(load_poems()),
        "words": len(load_words()),
        "english": len(load_english()),
        "stories": len(load_stories()),
    }
