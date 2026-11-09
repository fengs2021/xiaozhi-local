"""语音识别：sherpa-onnx 离线模型（SenseVoice / Paraformer）。

默认走 SenseVoice：中文准确率比 paraformer-small 高，自带标点与数字规范化，
且短音频不会像 Paraformer 那样吞字。Paraformer 作为可选引擎保留。
"""
from __future__ import annotations

import logging
import os
import re
import threading
import time
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


# SenseVoice 面对纯噪音/静音时偶尔整句吐这些「幻觉词」，同样没有实义
# 只喊名字/称呼（热词「小爱同学」太强时，短促的一声「那好啊」也会被强行解成「小爱同学」）
_ADDRESS_ONLY = frozenset({"小爱同学", "小智同学", "小智", "瑶瑶", "宝贝"})

NOISE_TEXTS = frozenset({
    "谢谢观看", "谢谢收看", "谢谢大家观看", "感谢观看", "字幕志愿者",
    "请不吝点赞订阅转发打赏支持", "订阅我的频道",
})


def is_filler(text: str) -> bool:
    """整句只有标点/语气词、且不是在发问时，认为不是有效输入。

    纯标点也要挡：说完一句话设备回到聆听态，环境噪音被 VAD 当成一声
    300~400ms 的人声切出来，SenseVoice 只给一个「。」。送进 LLM 只会换回
    一句「嗯？怎么不说话了」，白等两秒多，孩子还以为小智在犯傻。
    """
    t = clean_text(text)
    if not t:
        return True
    core = "".join(ch for ch in t if ch not in _PUNCT)
    if not core:
        return True                      # 「。」「，。」「。。」这类纯标点
    if core in _ADDRESS_ONLY:
        return True
    if core in NOISE_TEXTS:
        return True
    if len(core) > FILLER_MAX_LEN:
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


class FunASRNanoASR(_SherpaBase):
    """Fun-ASR-Nano（阿里 FunAudioLLM）离线识别器，sherpa-onnx 版模型。

    与 SenseVoice 的关系：同门，但这套是「encoder-adaptor + LLM 自回归解码」，
    中文长句、口语连读、专有名词的成句能力更强；代价是解码慢（要逐 token 生成），
    不像 SenseVoice 一次前向就出整句。

    模型目录里需要有 encoder_adaptor / llm / embedding / tokenizer 四个文件，
    sherpa-onnx 按 int8 → fp16 → 原始精度 的顺序自动挑。
    """

    def __init__(
        self,
        model_dir: str,
        num_threads: int = 4,
        language: str = "中文",
        use_itn: bool = True,
        max_new_tokens: int = 256,
        sample_rate: int = 16000,
    ):
        import sherpa_onnx

        super().__init__(sample_rate)
        files = _pick_funasr_files(model_dir)
        self._rec = sherpa_onnx.OfflineRecognizer.from_funasr_nano(
            encoder_adaptor=files["encoder_adaptor"],
            llm=files["llm"],
            embedding=files["embedding"],
            tokenizer=files["tokenizer"],
            num_threads=int(num_threads),
            sample_rate=sample_rate,
            feature_dim=80,
            decoding_method="greedy_search",
            debug=False,
            provider="cpu",
            language=str(language or ""),
            itn=bool(use_itn),
            max_new_tokens=int(max_new_tokens),
        )
        log.info(
            "sherpa-onnx Fun-ASR-Nano 已加载: %s (language=%s itn=%s threads=%d)",
            model_dir,
            language,
            use_itn,
            num_threads,
        )


def _pick_funasr_files(model_dir: str) -> Dict[str, str]:
    """在模型目录里认出 Fun-ASR-Nano 需要的四个文件。

    官方压缩包命名是 encoder_adaptor.int8.onnx / llm.int8.onnx / embedding.int8.onnx /
    tokenizer.json（不同版本可能带 fp16 或无后缀），这里按关键字扫一遍并优先 int8，
    免得模型一升级就得改代码。
    """
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"模型目录不存在: {model_dir}")
    names = sorted(os.listdir(model_dir))
    onnx = [n for n in names if n.endswith(".onnx")]

    def best(keyword: str) -> str:
        cand = [n for n in onnx if keyword in n.lower()]
        if not cand:
            return ""

        def rank(name: str) -> int:
            low = name.lower()
            if ".int8." in low:
                return 0
            if "fp16" in low:
                return 1
            return 2

        return os.path.join(model_dir, sorted(cand, key=rank)[0])

    tok = ""
    for name in ("tokenizer.json", "tokenizer.model", "tokens.txt"):
        if name in names:
            tok = os.path.join(model_dir, name)
            break
    files = {
        "encoder_adaptor": best("encoder_adaptor") or best("encoder"),
        "llm": best("llm"),
        "embedding": best("embedding"),
        "tokenizer": tok,
    }
    missing = [k for k, v in files.items() if not v]
    if missing:
        raise FileNotFoundError(f"模型目录缺文件 {missing}（现有: {names}）")
    return files


class FunASRGpuASR:
    """通过本机 HTTP 服务调用 GPU 版 Fun-ASR-Nano（模型常驻显存）。

    Fun-ASR-Nano 是「encoder-adaptor + LLM 自回归」，CPU 上每句要好几秒，跑不动；
    所以模型常驻在 /root/funasr-gpu/serve.py 那个独立进程里（RTX 3070，bf16，
    约 1.6GB 显存），这边只发原始 PCM 做 HTTP 调用。

    服务不可用（显卡被占、进程重启、超时）时自动回退到本地 SenseVoice，
    孩子不会因为后台服务抽风就听不到回应。
    """

    sample_rate = 16000

    def __init__(
        self,
        url: str = "http://127.0.0.1:9411/asr",
        timeout: float = 4.0,
        keywords: Any = None,
        language: str = "zh",
        max_new_tokens: int = 96,
        fallback: Any = None,
    ):
        self.url = str(url or "http://127.0.0.1:9411/asr")
        self.timeout = float(timeout or 4.0)
        self.language = str(language or "zh")
        self.max_new_tokens = int(max_new_tokens or 96)
        # 热词：孩子名字、常看书名、课本词语。Fun-ASR 是 LLM 解码，给了提示词
        # 就不会把「瑶瑶」听成「摇摇」、「尼尔斯骑鹅旅行记」听成别的东西。
        self.keywords = [str(k).strip() for k in (keywords or []) if str(k).strip()]
        self.fallback = fallback
        self._fails = 0
        self._down_until = 0.0
        log.info("Fun-ASR-Nano(GPU) 走 HTTP: %s（超时 %.1fs，热词 %d 个，回退 %s）",
                 self.url, self.timeout, len(self.keywords),
                 type(fallback).__name__ if fallback else "无")

    def _http(self, pcm: np.ndarray, timeout: float) -> str:
        import json
        import urllib.parse
        import urllib.request

        q = "?language=%s&max_new_tokens=%d" % (self.language, self.max_new_tokens)
        if self.keywords:
            q += "&keywords=" + urllib.parse.quote(",".join(self.keywords))
        req = urllib.request.Request(
            self.url + q,
            data=pcm.astype(np.int16).tobytes(),
            headers={"Content-Type": "application/octet-stream"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
        return clean_text(data.get("text") or "")

    def _use_fallback(self, pcm: np.ndarray, why: str) -> str:
        if self.fallback is None:
            return ""
        try:
            return self.fallback.transcribe(pcm)
        except Exception as exc:  # pragma: no cover
            log.error("SenseVoice 回退也失败: %s（原因 %s）", exc, why)
            return ""

    def transcribe(self, pcm: np.ndarray) -> str:  # noqa: D401
        if pcm is None or pcm.size < self.sample_rate // 10:
            return ""
        now = time.time()
        if now < self._down_until:          # 刚失败过，30 秒内直接走本地，别再等超时
            return self._use_fallback(pcm, "服务熔断中")
        try:
            text = self._http(pcm, self.timeout)
            self._fails = 0
            return text
        except Exception as exc:
            self._fails += 1
            if self._fails >= 2:            # 连续两次失败 → 熔断 30 秒
                self._down_until = time.time() + 30.0
                log.warning("GPU ASR 连续失败 %d 次，%s → 30 秒内改用 SenseVoice: %s",
                            self._fails, self.url, exc)
            else:
                log.warning("GPU ASR 调用失败，本次用 SenseVoice: %s", exc)
            return self._use_fallback(pcm, str(exc))


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
    if engine in ("funasr_gpu", "funasr-gpu", "funasr_http", "funasr_gpu_http"):
        url = a.get("gpu_url") or "http://127.0.0.1:9411/asr"
        fallback: Any = None
        fb_dir = resolve_path(cfg, a.get("fallback_model_dir", "") or
                              "models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17")
        try:
            fallback = SenseVoiceASR(fb_dir, int(a.get("num_threads", 2)), language="zh")
            log.info("GPU ASR 的回退引擎就绪: SenseVoice %s", fb_dir)
        except Exception as exc:  # pragma: no cover
            log.warning("回退引擎 SenseVoice 加载失败（不影响 GPU 主路）: %s", exc)
        words = a.get("hotwords") or []
        if isinstance(words, str):
            words = [w.strip() for w in words.replace("|", ",").split(",") if w.strip()]
        return FunASRGpuASR(
            url=str(url),
            keywords=words,
            timeout=float(a.get("gpu_timeout", 4.0) or 4.0),
            language=str(a.get("language", "zh") or "zh"),
            max_new_tokens=int(a.get("max_new_tokens", 96) or 96),
            fallback=fallback,
        )
    if engine in ("funasr", "funasr_nano", "fun-asr", "fun_asr"):
        model_dir = resolve_path(cfg, a.get("model_dir", ""))
        try:
            return FunASRNanoASR(
                model_dir,
                num_threads=int(a.get("num_threads", 4)),
                language=str(a.get("language", "中文") or "中文"),
                use_itn=bool(a.get("use_itn", True)),
                max_new_tokens=int(a.get("max_new_tokens", 256)),
            )
        except Exception as exc:  # pragma: no cover
            log.error("加载 Fun-ASR-Nano 模型失败: %s", exc)
            return NullASR()
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
