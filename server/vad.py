"""语音活动检测（VAD）：sherpa-onnx Silero VAD，用于判断一句话结束。"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

log = logging.getLogger("vad")


class SileroVAD:
    """把持续到来的 16kHz PCM 切成一句一句话。"""

    def __init__(
        self,
        model_path: str,
        threshold: float = 0.5,
        min_silence_ms: int = 700,
        min_speech_ms: int = 250,
        buffer_seconds: float = 60.0,
        sample_rate: int = 16000,
    ):
        import sherpa_onnx

        cfg = sherpa_onnx.VadModelConfig()
        cfg.silero_vad.model = model_path
        cfg.silero_vad.threshold = threshold
        cfg.silero_vad.min_silence_duration = min_silence_ms / 1000.0
        cfg.silero_vad.min_speech_duration = min_speech_ms / 1000.0
        cfg.silero_vad.max_speech_duration = 20.0
        cfg.sample_rate = sample_rate
        cfg.num_threads = 1
        cfg.provider = "cpu"
        self._vad = sherpa_onnx.VoiceActivityDetector(cfg, buffer_size_in_seconds=buffer_seconds)
        self.sample_rate = sample_rate
        self._speaking = False
        log.info(
            "Silero VAD 已加载 threshold=%.2f min_silence=%dms min_speech=%dms",
            threshold,
            min_silence_ms,
            min_speech_ms,
        )

    def push(self, pcm: np.ndarray) -> List[np.ndarray]:
        """喂入任意长度 PCM，返回已经完整的一句话列表。"""
        segments: List[np.ndarray] = []
        if pcm is None or pcm.size == 0:
            return segments
        samples = (pcm.astype(np.float32) / 32768.0).tolist()
        self._vad.accept_waveform(samples)
        while not self._vad.empty():
            seg = self._vad.front
            samples = np.asarray(seg.samples, dtype=np.float32)
            if samples.size:
                segments.append((np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16))
            self._vad.pop()
        return segments

    def flush(self) -> List[np.ndarray]:
        """会话结束时把残留语音取出。"""
        self._vad.flush()
        segments: List[np.ndarray] = []
        while not self._vad.empty():
            seg = self._vad.front
            samples = np.asarray(seg.samples, dtype=np.float32)
            if samples.size:
                segments.append((np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16))
            self._vad.pop()
        return segments

    def reset(self) -> None:
        self._vad.reset()

    @property
    def is_speech(self) -> bool:
        return bool(self._vad.is_speech_detected())


class EnergyVAD:
    """无模型兜底：按能量 + 静音时长切句。"""

    def __init__(self, threshold: float = 500.0, min_silence_ms: int = 700, sample_rate: int = 16000):
        self.threshold = threshold
        self.min_silence_samples = int(sample_rate * min_silence_ms / 1000)
        self.sample_rate = sample_rate
        self._buf = np.zeros(0, dtype=np.int16)
        self._silence = 0
        self._speaking = False

    def push(self, pcm: np.ndarray) -> List[np.ndarray]:
        out: List[np.ndarray] = []
        if pcm is None or pcm.size == 0:
            return out
        self._buf = np.concatenate([self._buf, pcm.astype(np.int16)])
        frame = self.sample_rate // 50  # 20ms
        consumed = 0
        while self._buf.size - consumed >= frame:
            chunk = self._buf[consumed : consumed + frame]
            consumed += frame
            rms = float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2)))
            if rms >= self.threshold:
                self._speaking = True
                self._silence = 0
            else:
                if self._speaking:
                    self._silence += frame
                    if self._silence >= self.min_silence_samples:
                        seg = self._buf[: consumed - self._silence]
                        if seg.size:
                            out.append(seg.copy())
                        self._buf = self._buf[consumed:]
                        consumed = 0
                        self._speaking = False
                        self._silence = 0
        if consumed:
            self._buf = self._buf[consumed:]
        return out

    def flush(self) -> List[np.ndarray]:
        seg = self._buf
        self._buf = np.zeros(0, dtype=np.int16)
        return [seg] if seg.size else []

    def reset(self) -> None:
        self._buf = np.zeros(0, dtype=np.int16)
        self._silence = 0
        self._speaking = False

    @property
    def is_speech(self) -> bool:
        return self._speaking


def build_vad(cfg: Dict[str, Any]) -> Any:
    """按配置构造 VAD，模型缺失时退化为能量检测。"""
    from .config import resolve_path

    v = cfg.get("vad") or {}
    model = resolve_path(cfg, v.get("model", ""))
    params = dict(
        threshold=float(v.get("threshold", 0.5)),
        min_silence_ms=int(v.get("min_silence_ms", 700)),
        min_speech_ms=int(v.get("min_speech_ms", 250)),
        buffer_seconds=float(v.get("buffer_seconds", 60)),
    )
    try:
        import os

        if model and os.path.exists(model):
            return SileroVAD(model, **params)
        log.warning("VAD 模型不存在（%s），退化为能量检测", model)
    except Exception as exc:  # pragma: no cover
        log.error("加载 VAD 失败: %s，退化为能量检测", exc)
    return EnergyVAD(threshold=500.0, min_silence_ms=params["min_silence_ms"])
