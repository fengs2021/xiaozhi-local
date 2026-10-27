# -*- coding: utf-8 -*-
"""A/B 基准：TTS 预连接池（复用 WSS 握手）开/关 对首包延迟的影响。

复刻 server/session.py 的真实流水线：
    LLM 流式出字 -> 切句（句末标点 / 逗号类攒够 9 字）-> 并发合成（信号量 3）-> 按序播放
唯一变量：TTS 是否提前握手（WSS 预连接池）。

用法：
    python3 _bench_prewarm_ab.py cold 3
    python3 _bench_prewarm_ab.py warm 3
"""
import asyncio
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from server.config import load_config
from server.llm import build_llm, strip_for_speech
from server.tts import build_tts

QUESTION = "给我讲一个关于小猫的短故事"
SENT_END = "\u3002\uff01\uff1f\uff1b\n!?;"
SENT_SOFT = "\uff0c\uff1a,"


def cut_sentences(buf):
    """按 session.py 的规则从 buf 里切出完整句子，返回 (句子列表, 剩余 buf)。"""
    out = []
    while True:
        cut = -1
        for i, ch in enumerate(buf):
            if ch in SENT_END:
                cut = i
                break
            if ch in SENT_SOFT and i >= 9:
                cut = i
                break
        if cut < 0:
            break
        out.append(buf[: cut + 1])
        buf = buf[cut + 1 :]
    return out, buf


async def run_once(cfg, llm, tts, warm: bool):
    """跑一轮，warm=True 时启用预连接池。"""
    real_prewarm = getattr(tts, "prewarm", None)
    if not warm:
        # 关掉预热，并把池里可能残留的连接清空，保证真的走冷连接
        tts._idle.clear()
        tts.prewarm = lambda *a, **k: None
    else:
        tts.prewarm = real_prewarm
        real_prewarm(2)

    t0 = time.time()
    first_text_ms = None
    first_audio_ms = None
    synth_ms = []
    sem = asyncio.Semaphore(3)

    async def synth(sent):
        async with sem:
            ts = time.time()
            pcm = await tts.synthesize(sent)
            return pcm, int((time.time() - ts) * 1000)

    tasks = []
    buf = ""

    async for piece in llm.chat_stream([], QUESTION):
        if first_text_ms is None:
            first_text_ms = int((time.time() - t0) * 1000)
        buf += piece
        sentences, buf = cut_sentences(buf)
        for sent in sentences:
            s = strip_for_speech(sent).strip()
            if not s:
                continue
            tasks.append(asyncio.create_task(synth(s)))
            if warm:
                real_prewarm(3)
    if buf.strip():
        s = strip_for_speech(buf).strip()
        if s:
            tasks.append(asyncio.create_task(synth(s)))
            if warm:
                real_prewarm(3)

    # 按序播放（等第 i 句合成完才算首包）
    for i, task in enumerate(tasks):
        pcm, ms = await task
        synth_ms.append(ms)
        if first_audio_ms is None and pcm.size > 0:
            first_audio_ms = int((time.time() - t0) * 1000)

    total_ms = int((time.time() - t0) * 1000)
    if not warm:
        tts.prewarm = real_prewarm
    return {
        "first_text_ms": first_text_ms,
        "first_audio_ms": first_audio_ms,
        "synth_ms": synth_ms,
        "total_ms": total_ms,
        "sentences": len(synth_ms),
    }


async def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "warm"
    rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    cfg = load_config()
    preset = (cfg.get("llm") or {}).get("presets", {})[(cfg.get("llm") or {}).get("active")]
    llm = build_llm(preset, cfg.get("persona"))
    llm.system_prompt = (cfg.get("llm") or {}).get("system_prompt", "")
    tts = build_tts(cfg)

    results = []
    for i in range(rounds):
        r = await run_once(cfg, llm, tts, warm=(mode == "warm"))
        results.append(r)
        print(
            f"[{mode} #{i+1}] 首字 {r['first_text_ms']}ms | 首音频 {r['first_audio_ms']}ms | "
            f"总 {r['total_ms']}ms | 句数 {r['sentences']} | 每句合成 {r['synth_ms']}",
            flush=True,
        )
        await asyncio.sleep(1.0)

    def med(key):
        vals = [r[key] for r in results if r[key] is not None]
        return int(statistics.median(vals)) if vals else -1

    flat = [m for r in results for m in r["synth_ms"]]
    print(
        f"== {mode} 汇总(中位数) 首字 {med('first_text_ms')}ms 首音频 {med('first_audio_ms')}ms "
        f"总 {med('total_ms')}ms 每句合成中位 {int(statistics.median(flat)) if flat else -1}ms",
        flush=True,
    )
    try:
        await tts.aclose()
    except Exception:
        pass


asyncio.run(main())
