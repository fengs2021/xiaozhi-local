# -*- coding: utf-8 -*-
"""TTS 预连接池专项基准：只量 TTS，剥掉 LLM 抖动。

对比两种取连接方式（同一批句子、同一台机器、交替跑）：
  cold: 每句现连 WSS（池是空的）
  warm: 先 prewarm(N) 握手好，每句直接取现成连接
输出：每句合成耗时、裸握手耗时、首句额外开销。
"""
import asyncio
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from server.config import load_config
from server.tts import build_tts

SENTS = [
    "从前有一只特别爱冒险的小猫咪。",
    "它总爱跑到山顶上看远方的白云。",
    "有一天它终于看见了那座传说中的城堡。",
]


async def handshake_ms(tts, n=3):
    """裸握手耗时：直接连一次猫剪 WSS，不发任务。"""
    out = []
    for _ in range(n):
        import websockets

        t0 = time.time()
        ws = await websockets.connect(tts._build_url(), max_size=None, open_timeout=tts.timeout)
        out.append(int((time.time() - t0) * 1000))
        await ws.close()
    return out


async def run(cfg, warm: bool):
    tts = build_tts(cfg)
    if warm:
        tts.prewarm(3)
        # 等池子填满
        for _ in range(60):
            if len(getattr(tts, "_idle", [])) >= 3:
                break
            await asyncio.sleep(0.1)
    else:
        tts._idle.clear()

    ms = []
    t0 = time.time()
    for s in SENTS:
        ts = time.time()
        pcm = await tts.synthesize(s)
        ms.append(int((time.time() - ts) * 1000))
    total = int((time.time() - t0) * 1000)
    try:
        await tts.aclose()
    except Exception:
        pass
    return ms, total


async def main():
    cfg = load_config()
    print("裸握手耗时(ms):", await handshake_ms(build_tts(cfg), 4), flush=True)
    for mode in ("cold", "warm", "cold", "warm"):
        ms, total = await run(cfg, warm=(mode == "warm"))
        print(
            f"[{mode}] 每句 {ms} | 中位 {int(statistics.median(ms))}ms | 合计 {total}ms",
            flush=True,
        )
        await asyncio.sleep(0.5)


asyncio.run(main())
