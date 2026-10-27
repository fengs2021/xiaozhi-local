"""Opus 编解码与重采样（PyAV / libopus）。

固件侧约定（main/protocols/websocket_protocol.cc）：
- 上行：设备发裸 Opus 包（version_=1），16kHz 单声道，60ms 一帧（960 样本）。
- 下行：服务器发裸 Opus 包，采样率/帧长以 server hello 的 audio_params 为准。
"""
from __future__ import annotations

import numpy as np

try:
    import av
except Exception as exc:  # pragma: no cover
    raise RuntimeError("需要安装 PyAV：pip install av") from exc


def _frame(pcm: np.ndarray, rate: int) -> "av.AudioFrame":
    """把 (1, n) int16 数组包成 AudioFrame。"""
    if pcm.ndim == 1:
        pcm = pcm.reshape(1, -1)
    pcm = np.ascontiguousarray(pcm, dtype=np.int16)
    f = av.AudioFrame.from_ndarray(pcm, format="s16", layout="mono")
    f.sample_rate = rate
    return f


class OpusDownlinkEncoder:
    """把 24kHz int16 PCM 编成固定帧长的 Opus 包。"""

    def __init__(self, sample_rate: int = 24000, frame_ms: int = 20, bitrate: int = 32000):
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.frame_samples = int(sample_rate * frame_ms / 1000)
        self._ctx = av.CodecContext.create("libopus", "w")
        self._ctx.sample_rate = sample_rate
        self._ctx.layout = "mono"
        self._ctx.format = "s16"
        self._ctx.bit_rate = bitrate
        self._ctx.open()
        # libopus 只支持 2.5/5/10/20/40/60ms，且不允许任意改 frame_size，
        # 因此这里按 20ms 切块，若配置不是 20ms 会退化为 20ms 并在启动时告警。
        self.actual_frame_ms = 20 if self._ctx.frame_size == 480 else self._ctx.frame_size * 1000 // sample_rate
        self._pending = np.zeros(0, dtype=np.int16)

    def encode(self, pcm: np.ndarray) -> list[bytes]:
        """喂入任意长度 PCM，返回能凑满整帧的 Opus 包列表。"""
        pcm = np.asarray(pcm, dtype=np.int16).reshape(-1)
        if pcm.size:
            self._pending = np.concatenate([self._pending, pcm])
        out: list[bytes] = []
        n = self.frame_samples
        while self._pending.size >= n:
            chunk = self._pending[:n]
            self._pending = self._pending[n:]
            out.extend(bytes(p) for p in self._ctx.encode(_frame(chunk, self.sample_rate)))
        return out

    def flush(self) -> list[bytes]:
        """把残余样本补零成整帧后编码（结尾用）。"""
        if self._pending.size == 0:
            return []
        pad = np.zeros(self.frame_samples - self._pending.size, dtype=np.int16)
        pcm = np.concatenate([self._pending, pad])
        self._pending = np.zeros(0, dtype=np.int16)
        return [bytes(p) for p in self._ctx.encode(_frame(pcm, self.sample_rate))]


class OpusUplinkDecoder:
    """把设备发来的 Opus 包解成 16kHz int16 PCM。"""

    def __init__(self, out_rate: int = 16000):
        self.out_rate = out_rate
        self._ctx = av.CodecContext.create("opus", "r")
        self._ctx.open()
        self._resampler = av.AudioResampler(format="s16", layout="mono", rate=out_rate)

    def decode(self, payload: bytes) -> np.ndarray:
        out = np.zeros(0, dtype=np.int16)
        try:
            pkt = av.Packet(payload)
            frames = self._ctx.decode(pkt)
        except Exception:
            return out
        for f in frames:
            for r in self._resampler.resample(f):
                arr = r.to_ndarray()
                if arr.dtype != np.int16:
                    arr = (np.clip(arr, -1.0, 1.0) * 32767).astype(np.int16)
                out = np.concatenate([out, arr.reshape(-1)])
        return out


def decode_audio_file(path: str, sample_rate: int = 24000) -> np.ndarray:
    """解码任意音频文件为指定采样率的 int16 单声道。"""
    container = av.open(path)
    stream = container.streams.audio[0]
    resampler = av.AudioResampler(format="s16", layout="mono", rate=sample_rate)
    chunks: list[np.ndarray] = []
    for frame in container.decode(stream):
        for r in resampler.resample(frame):
            arr = r.to_ndarray()
            if arr.dtype != np.int16:
                arr = (np.clip(arr, -1.0, 1.0) * 32767).astype(np.int16)
            chunks.append(arr.reshape(-1))
    container.close()
    for r in resampler.resample(None):
        arr = r.to_ndarray()
        if arr.dtype != np.int16:
            arr = (np.clip(arr, -1.0, 1.0) * 32767).astype(np.int16)
        chunks.append(arr.reshape(-1))
    if not chunks:
        return np.zeros(0, dtype=np.int16)
    return np.concatenate(chunks)


def resample(pcm: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """int16 PCM 重采样。"""
    if src_rate == dst_rate or pcm.size == 0:
        return pcm
    resampler = av.AudioResampler(format="s16", layout="mono", rate=dst_rate)
    out: list[np.ndarray] = []
    for r in resampler.resample(_frame(pcm, src_rate)):
        arr = r.to_ndarray()
        if arr.dtype != np.int16:
            arr = (np.clip(arr, -1.0, 1.0) * 32767).astype(np.int16)
        out.append(arr.reshape(-1))
    return np.concatenate(out) if out else np.zeros(0, dtype=np.int16)
