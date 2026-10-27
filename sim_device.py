
"""模拟小智设备，验证本地服务端全链路（不需要真实硬件）。"""
import asyncio, json, sys, time
import numpy as np
import av
import aiohttp, soundfile as sf

URL = "ws://127.0.0.1:8003/xiaozhi/v1/"
TOKEN = "xiaozhi-local"
WAV = r"D:\Users\liqih\Documents\deepseek-harness\default-workspace\xiaozhi-local\models\sherpa-onnx-paraformer-zh-small-2024-03-09\test_wavs\0.wav"

def encode_opus_16k(pcm, frame_ms=60):
    c = av.CodecContext.create("libopus", "w")
    c.sample_rate = 16000; c.layout = "mono"; c.format = "s16"; c.open()
    n = int(16000 * frame_ms / 1000)
    out = []
    for i in range(0, len(pcm) - n + 1, n):
        chunk = np.ascontiguousarray(pcm[i:i+n], dtype=np.int16).reshape(1, -1)
        f = av.AudioFrame.from_ndarray(chunk, format="s16", layout="mono"); f.sample_rate = 16000
        out += [bytes(p) for p in c.encode(f)]
    return out

async def main():
    data, sr = sf.read(WAV, dtype="int16")
    if data.ndim > 1: data = data[:, 0]
    assert sr == 16000, sr
    # 真实场景里用户说完会有静音尾巴，补 1.5s 静音让 VAD 收句
    data = np.concatenate([data, np.zeros(16000 * 3 // 2, dtype=np.int16)])
    packets = encode_opus_16k(data)
    print(f"上行音频 {len(data)/16000:.2f}s -> {len(packets)} 个 Opus 包")

    headers = {
        "Authorization": f"Bearer {TOKEN}",
        "Protocol-Version": "1",
        "Device-Id": "90:70:69:f7:e7:ac",
        "Client-Id": "2dd93c1e-3a3c-42d5-87e4-146847730577",
    }
    async with aiohttp.ClientSession() as sess:
        async with sess.ws_connect(URL, headers=headers, max_msg_size=0) as ws:
            await ws.send_json({"type":"hello","version":1,"features":{"mcp":True},
                                "transport":"websocket",
                                "audio_params":{"format":"opus","sample_rate":16000,"channels":1,"frame_duration":60}})
            hello = await asyncio.wait_for(ws.receive_json(), 10)
            print("服务器 hello:", json.dumps(hello, ensure_ascii=False))
            assert hello.get("transport") == "websocket"

            await ws.send_json({"type":"listen","state":"start","mode":"auto"})
            t0 = time.time()
            for p in packets:
                await ws.send_bytes(p)
                await asyncio.sleep(0.06)
            print(f"上行完成 {time.time()-t0:.1f}s")

            # 收集事件
            got = {"stt":None,"llm":None,"tts_start":False,"tts_stop":False,"audio":0,"audio_bytes":0}
            deadline = time.time() + 40
            dec = av.CodecContext.create("opus","r"); dec.open()
            rs = av.AudioResampler(format="s16", layout="mono", rate=24000)
            total_samples = 0
            while time.time() < deadline:
                try:
                    msg = await asyncio.wait_for(ws.receive(), 20)
                except asyncio.TimeoutError:
                    print("(20 秒无消息，结束)")
                    break
                if msg.type == aiohttp.WSMsgType.TEXT:
                    d = json.loads(msg.data)
                    if d.get("type") == "stt":
                        got["stt"] = d.get("text"); print("STT:", d.get("text"))
                    elif d.get("type") == "llm":
                        got["llm"] = d.get("text"); print("LLM:", d.get("text"))
                    elif d.get("type") == "tts":
                        if d.get("state") == "start": got["tts_start"] = True; print("TTS start")
                        if d.get("state") == "stop": got["tts_stop"] = True; print("TTS stop"); break
                        if d.get("state") == "sentence_start": print("  句:", d.get("text"))
                elif msg.type == aiohttp.WSMsgType.BINARY:
                    got["audio"] += 1; got["audio_bytes"] += len(msg.data)
                    for f in dec.decode(av.Packet(msg.data)):
                        for r in rs.resample(f):
                            total_samples += r.samples
            print(f"\n下行: {got['audio']} 包 / {got['audio_bytes']} 字节 / 解码 {total_samples/24000:.2f}s @24k")
            print("结果:", json.dumps({k:v for k,v in got.items()}, ensure_ascii=False))
            ok = got["stt"] and got["llm"] and got["audio"] > 0 and got["tts_stop"]
            print("\n=== 链路验证:", "通过" if ok else "失败", "===")

asyncio.run(main())
