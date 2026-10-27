"""语音合成：edge-tts（在线）与 Windows SAPI（离线）。"""
from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from typing import Any, Dict

import numpy as np

from .audio import decode_audio_file
from .tts_maoxiang import (
    EMOTIONS as MAOXIANG_EMOTIONS,
    MAOXIANG_SAMPLE_RATE,
    MaoxiangTTS,
    load_maoxiang_voices,
)

log = logging.getLogger("tts")

# edge-tts 常用中文音色表（前端下拉用；均为微软公开音色，无需 key）
EDGE_VOICES = [
    {"id": "zh-CN-XiaoxiaoNeural", "name": "晓晓", "gender": "女", "style": "温暖亲和，通用助手"},
    {"id": "zh-CN-XiaoyiNeural", "name": "晓伊", "gender": "女", "style": "活泼俏皮，年轻"},
    {"id": "zh-CN-YunxiNeural", "name": "云希", "gender": "男", "style": "阳光少年，清朗"},
    {"id": "zh-CN-YunjianNeural", "name": "云健", "gender": "男", "style": "浑厚有力，解说感"},
    {"id": "zh-CN-YunyangNeural", "name": "云扬", "gender": "男", "style": "专业播报，新闻腔"},
    {"id": "zh-CN-YunxiaNeural", "name": "云夏", "gender": "男", "style": "少年音，活泼"},
    {"id": "zh-CN-YunzeNeural", "name": "云泽", "gender": "男", "style": "成熟稳重，故事感"},
    {"id": "zh-CN-XiaochenNeural", "name": "晓辰", "gender": "女", "style": "自然随和"},
    {"id": "zh-CN-XiaohanNeural", "name": "晓涵", "gender": "女", "style": "温柔舒缓"},
    {"id": "zh-CN-XiaomoNeural", "name": "晓墨", "gender": "女", "style": "沉稳大气"},
    {"id": "zh-CN-XiaoruiNeural", "name": "晓睿", "gender": "女", "style": "知性成熟"},
    {"id": "zh-CN-liaoning-XiaobeiNeural", "name": "晓北", "gender": "女", "style": "东北口音，接地气"},
    {"id": "zh-CN-shaanxi-XiaoniNeural", "name": "晓妮", "gender": "女", "style": "陕西口音，爽利"},
    {"id": "zh-HK-HiuMaanNeural", "name": "曉曼", "gender": "女", "style": "粤语，柔和"},
    {"id": "zh-TW-HsiaoChenNeural", "name": "曉臻", "gender": "女", "style": "台湾国语，甜美"},
]

EDGE_VOICE_IDS = [v["id"] for v in EDGE_VOICES]


class _StreamFallback:
    """流式合成的默认实现：整句合成后一次性产出。

    不支持真流式的引擎（edge / sapi / null）继承它即可，调用方统一走 stream()，
    不必判断引擎类型；拿到的是「一整块音频」，行为与改造前完全一致。
    """

    async def stream(self, text: str):
        pcm = await self.synthesize(text)
        if getattr(pcm, "size", 0):
            yield pcm


class EdgeTTS(_StreamFallback):
    """微软 Edge 在线语音，返回 24kHz int16 PCM。"""

    def __init__(
        self,
        voice: str = "zh-CN-XiaoxiaoNeural",
        rate: str = "+0%",
        volume: str = "+0%",
        sample_rate: int = 24000,
        pitch: str = "+0Hz",
    ):
        self.voice = voice
        self.rate = rate
        self.volume = volume
        self.sample_rate = sample_rate
        self.pitch = pitch or "+0Hz"

    async def synthesize(self, text: str) -> np.ndarray:
        import edge_tts

        if not text.strip():
            return np.zeros(0, dtype=np.int16)
        communicate = edge_tts.Communicate(
            text, self.voice, rate=self.rate, volume=self.volume, pitch=self.pitch
        )
        buf = bytearray()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                buf.extend(chunk["data"])
        if not buf:
            return np.zeros(0, dtype=np.int16)
        # edge-tts 输出 mp3，落临时文件后交给 PyAV 解码
        fd, path = tempfile.mkstemp(suffix=".mp3")
        os.close(fd)
        try:
            with open(path, "wb") as f:
                f.write(buf)
            return await asyncio.get_running_loop().run_in_executor(
                None, decode_audio_file, path, self.sample_rate
            )
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass


class PreviewTTS(EdgeTTS):
    """试听专用：用指定音色/语速/音调合成一句，可临时覆盖 voice。"""

    async def synthesize_with(self, text: str, voice: str = "") -> np.ndarray:
        if voice and voice != self.voice:
            old = self.voice
            self.voice = voice
            try:
                return await self.synthesize(text)
            finally:
                self.voice = old
        return await self.synthesize(text)


class SapiTTS(_StreamFallback):
    """Windows 内置 SAPI 语音（离线，音质一般）。"""

    def __init__(self, voice: str = "", sample_rate: int = 24000):
        self.voice = voice
        self.sample_rate = sample_rate

    def _synth_sync(self, text: str) -> np.ndarray:
        import win32com.client

        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            speaker = win32com.client.Dispatch("SAPI.SpVoice")
            if self.voice:
                for v in speaker.GetVoices():
                    if self.voice.lower() in v.GetDescription().lower():
                        speaker.Voice = v
                        break
            stream = win32com.client.Dispatch("SAPI.SpFileStream")
            stream.Open(path, 3, False)
            speaker.AudioOutputStream = stream
            speaker.Speak(text)
            stream.Close()
            return decode_audio_file(path, self.sample_rate)
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass

    async def synthesize(self, text: str) -> np.ndarray:
        if not text.strip():
            return np.zeros(0, dtype=np.int16)
        return await asyncio.get_running_loop().run_in_executor(None, self._synth_sync, text)


class MaoxiangPreview(MaoxiangTTS):
    """猫剪试听：可临时覆盖音色。"""

    async def synthesize_with(self, text: str, voice: str = "") -> np.ndarray:
        if voice and voice != self.voice:
            old = self.voice
            self.voice = voice
            try:
                return await self.synthesize(text)
            finally:
                self.voice = old
        return await self.synthesize(text)


class NullTTS(_StreamFallback):
    async def synthesize(self, text: str) -> np.ndarray:
        return np.zeros(0, dtype=np.int16)


def speech_rate_factor(cfg: Dict[str, Any]) -> float:
    """把 persona.speech_speed / tts.rate（形如 +20% / -10% / 1.2）转成倍率。"""
    t = cfg.get("tts") or {}
    persona = cfg.get("persona") or {}
    raw = str(persona.get("speech_speed") or t.get("rate") or "+0%").strip()
    try:
        if raw.endswith("%"):
            return max(0.2, min(3.0, 1.0 + float(raw[:-1]) / 100.0))
        val = float(raw)
        return max(0.2, min(3.0, val if val > 0 else 1.0))
    except (TypeError, ValueError):
        return 1.0


def pitch_value(cfg: Dict[str, Any]) -> float:
    """把 persona.pitch / tts.pitch（形如 +2Hz）转成猫剪的 -5..5 档位。"""
    t = cfg.get("tts") or {}
    persona = cfg.get("persona") or {}
    raw = str(persona.get("pitch") or t.get("pitch") or "+0Hz").strip()
    try:
        if raw.lower().endswith("hz"):
            raw = raw[:-2]
        hz = float(raw)
    except (TypeError, ValueError):
        return 0.0
    return max(-5.0, min(5.0, hz / 10.0))


def build_tts(cfg: Dict[str, Any]) -> Any:
    """按配置构造 TTS；语速/音调优先取 persona，缺省回落到 tts 段。"""
    t = cfg.get("tts") or {}
    persona = cfg.get("persona") or {}
    engine = t.get("engine", "none")
    sr = int((cfg.get("audio") or {}).get("downlink_sample_rate", 24000))
    # persona.speech_speed -> edge-tts rate；persona.pitch -> edge-tts pitch
    rate = str(persona.get("speech_speed") or t.get("rate") or "+0%").strip()
    pitch = str(persona.get("pitch") or t.get("pitch") or "+0Hz").strip()
    if engine == "maoxiang":
        return MaoxiangTTS(
            voice=t.get("voice", "zh_female_vv_uranus_bigtts"),
            sample_rate=sr,
            speech_rate=speech_rate_factor(cfg),
            pitch=pitch_value(cfg),
            loudness=float(t.get("loudness") or 0),
            emotion=str(t.get("emotion") or ""),
            emotion_scale=float(t.get("emotion_scale") or 0),
        )
    if engine == "edge":
        return EdgeTTS(
            t.get("voice", "zh-CN-XiaoxiaoNeural"),
            rate,
            t.get("volume", "+0%"),
            sr,
            pitch,
        )
    if engine == "sapi":
        return SapiTTS(t.get("sapi_voice", ""), sr)
    return NullTTS()


def voice_catalog(cfg: Dict[str, Any], maoxiang_cache: str = "") -> Dict[str, Any]:
    """返回按引擎分组的音色目录，供控制中心下拉使用。"""
    from pathlib import Path

    mx = load_maoxiang_voices(Path(maoxiang_cache) if maoxiang_cache else None)
    return {
        "edge": {"label": "微软 Edge（在线）", "voices": EDGE_VOICES},
        "maoxiang": {"label": "火山豆包 / 猫剪（在线）", "voices": mx},
        "emotions": MAOXIANG_EMOTIONS,
        "maoxiang_sample_rate": MAOXIANG_SAMPLE_RATE,
    }


def pcm_to_wav_bytes(pcm: np.ndarray, sample_rate: int) -> bytes:
    """把 int16 单声道 PCM 编码成内存里的 WAV（标准库 wave，浏览器可直接播）。"""
    import io
    import wave

    data = np.asarray(pcm, dtype=np.int16).reshape(-1)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(data.tobytes())
    return buf.getvalue()
