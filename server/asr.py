"""语音识别：sherpa-onnx 离线模型（SenseVoice / Paraformer）。

默认走 SenseVoice：中文准确率比 paraformer-small 高，自带标点与数字规范化，
且短音频不会像 Paraformer 那样吞字。Paraformer 作为可选引擎保留。
"""
from __future__ import annotations

import logging
import os
import re
import threading
from typing import Any, Dict, Optional

import numpy as np

log = logging.getLogger("asr")

# SenseVoice 会在结果前后加 <|zh|><|NEUTRAL|> 这类标签，识别后必须剥掉
_TAG_RE = re.compile(r"<\|[^|]*\|>")

# 纯语气词：孩子口齿不清、环境噪音、VAD 切到半个字，常被识别成这些。
# 交给 LLM 只会白跑一轮两秒多的往返，所以在这里直接丢掉。
FILLER_CHARS = "嗯啊哦呃唉诶哎呀噢喔额唔呣"
FILLER_MAX_LEN = 3
_PUNCT = "。，、！？…,.!?；;：: \t\r\n\"'“”‘’"


def clean_text(text: str) -> str:
    """剥掉 SenseVoice 的标签，压掉多余空白。"""
    return _TAG_RE.sub("", text or "").strip()


def is_filler(text: str) -> bool:
    """整句只有语气词、且不是在发问时，认为不是有效输入。"""
    t = clean_text(text)
    if not t:
        return True
    core = "".join(ch for ch in t if ch not in _PUNCT)
    if not core or len(core) > FILLER_MAX_LEN:
        return False
    if not all(ch in FILLER_CHARS for ch in core):
        return False
    # 「啊？」「哦？」是真的在问，别吞掉
    if "?" in t or "？" in t:
        return False
    return True


class _SherpaBase:
    """共用的解码骨架：加锁、建流、解码、取文本。"""

    sample_rate = 16000

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self._lock = threading.Lock()

    def _decode(self, pcm: np.ndarray) -> str:
        if pcm is None or pcm.size < self.sample_rate // 10:
            return ""
        audio = pcm.astype(np.float32) / 32768.0
        with self._lock:
            stream = self._rec.create_stream()
            stream.accept_waveform(self.sample_rate, audio)
            self._rec.decode_stream(stream)
            raw = stream.result.text or ""
        return clean_text(raw)

    def transcribe(self, pcm: np.ndarray) -> str:
        return self._decode(pcm)


class SenseVoiceASR(_SherpaBase):
    """SenseVoice 离线识别器，16kHz 单声道 int16。

    相比 Paraformer-small 的主要好处：
    - 中文同音字错得少（实测「腰疼」→「潦草」、「加楼」→「词语接龙」）
    - use_itn 打开后数字直接是阿拉伯数字、句子自带标点，TTS 分句更准
    - 必须显式指定 language='zh'：自动检测会把短促的「啊」判成日语
    """

    def __init__(
        self,
        model_dir: str,
        num_threads: int = 2,
        language: str = "zh",
        use_itn: bool = True,
        sample_rate: int = 16000,
    ):
        import sherpa_onnx

        super().__init__(sample_rate)
        model = os.path.join(model_dir, "model.int8.onnx")
        if not os.path.exists(model):
            model = os.path.join(model_dir, "model.onnx")
        tokens = os.path.join(model_dir, "tokens.txt")
        self._rec = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=model,
            tokens=tokens,
            use_itn=bool(use_itn),
            language=str(language or "zh"),
            num_threads=int(num_threads),
            debug=False,
            provider="cpu",
        )
        log.info(
            "sherpa-onnx SenseVoice 已加载: %s (language=%s itn=%s)",
            model_dir,
            language,
            use_itn,
        )


class SherpaASR(_SherpaBase):
    """Paraformer 离线识别器（保留，可在 config 里切回去）。"""

    def __init__(self, model_dir: str, num_threads: int = 2, sample_rate: int = 16000):
        import sherpa_onnx

        super().__init__(sample_rate)
        self._rec = sherpa_onnx.OfflineRecognizer.from_paraformer(
            paraformer=os.path.join(model_dir, "model.int8.onnx"),
            tokens=os.path.join(model_dir, "tokens.txt"),
            num_threads=num_threads,
            sample_rate=sample_rate,
            feature_dim=80,
            decoding_method="greedy_search",
            debug=False,
            provider="cpu",
        )
        log.info("sherpa-onnx Paraformer 已加载: %s", model_dir)


class NullASR:
    """未启用 ASR 时的占位实现。"""

    def transcribe(self, pcm: np.ndarray) -> str:  # noqa: D401
        return ""


def build_asr(cfg: Dict[str, Any], root: str) -> Any:
    """按配置构造 ASR。"""
    from .config import resolve_path

    a = cfg.get("asr") or {}
    engine = (a.get("engine") or "none").strip().lower()
    if engine in ("sensevoice", "sense_voice", "sherpa"):
        model_dir = resolve_path(cfg, a.get("model_dir", ""))
        num_threads = int(a.get("num_threads", 2))
        try:
            if engine == "sherpa":
                return SherpaASR(model_dir, num_threads)
            return SenseVoiceASR(
                model_dir,
                num_threads,
                language=str(a.get("language", "zh") or "zh"),
                use_itn=bool(a.get("use_itn", True)),
            )
        except Exception as exc:  # pragma: no cover
            log.error("加载 %s 模型失败: %s", engine, exc)
            return NullASR()
    return NullASR()
