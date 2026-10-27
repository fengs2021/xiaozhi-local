# 小智本地控制中心

把「小智 AI」ESP32 硬件从官方服务器（api.tenclass.net）接到你自己的电脑上，
用自己配置的大模型对话，不依赖官方云。

## 一、它做了什么

设备开机 → 请求 OTA 地址 → 本服务返回 WebSocket 地址 → 设备连上来 →
说一句话 → 本机 ASR 识别 → 你的大模型回答 → 本机 TTS 合成 → Opus 编码发回设备播报。

```
ESP32 设备 ──HTTP POST /xiaozhi/ota/──> 本服务（返回 ws://<你的服务器IP>:8003/xiaozhi/v1/）
         ──WebSocket 音频──> ASR(sherpa-onnx) → LLM(OpenAI 兼容) → TTS(edge-tts) → Opus
```

## 二、启动

双击 `启动控制中心.bat`，或：

```bash
cd xiaozhi-local
python run.py
```

打开控制台：<http://127.0.0.1:8003>

## 三、让设备连到本机

设备的 OTA 地址存在 NVS 里，优先级高于固件内置地址，有两种改法：

### 方法 A：配网 AP 高级配置（推荐，不用刷机）

1. 设备上长按 / 点击 BOOT 键进入配网模式（屏幕提示「进入配网模式」）
2. 手机连设备的 WiFi 热点（形如 `Xiaozhi-xxxx`）
3. 浏览器打开 <http://192.168.4.1>，进入「高级配置」
4. 把 **OTA 地址** 改成：`http://<你的服务器IP>:8003/xiaozhi/ota/`
5. 保存并重启设备

### 方法 B：重新烧录固件时写死 OTA 地址

在 `menuconfig` 里改 `Xiaozhi Assistant → Default OTA URL`，或用 NVS 分区工具写入。

> 注意：本机 IP 必须固定（或在路由器里给这台电脑绑定静态 IP），否则设备会连不上。

## 四、换模型

控制台「模型配置」里直接改，保存即生效，不用重启：

| 预设 | 用途 |
|---|---|
| `deepseek` | DeepSeek 官方，需 API Key |
| `openai` | OpenAI 官方 |
| `local-ollama` | 本机 Ollama（先 `ollama serve`，模型如 `qwen2.5:7b`） |
| `echo` | 不回模型，只复述识别结果，用来验证链路 |

任何 **OpenAI 兼容** 的服务都能用（vLLM、LM Studio、OneAPI、硅基流动、月之暗面等），
只要填对 `base_url` + `model` + `api_key`。

API Key 也可以放环境变量 `XIAOZHI_LLM_API_KEY`，避免写进配置文件。

## 五、组件

| 模块 | 实现 | 说明 |
|---|---|---|
| ASR | sherpa-onnx Paraformer 中文 small | 完全本地离线 |
| LLM | OpenAI 兼容 HTTP | 任意模型 |
| TTS | edge-tts（默认）/ Windows SAPI | edge 音质好，需联网 |
| VAD | sherpa-onnx Silero VAD | 判断一句话说完 |
| 编解码 | PyAV / libopus | 上行 16k，下行 24k |

## 六、目录

```
xiaozhi-local/
├── config.yaml              全部配置（模型、端口、TTS 声音、VAD 灵敏度）
├── run.py                   启动入口
├── sim_device.py            模拟设备，无需硬件即可验证全链路
├── server/
│   ├── app.py               HTTP/WS 路由、OTA 端点、控制台 API
│   ├── session.py           单设备会话：收音频 → 管线 → 发音频
│   ├── audio.py             Opus 编解码与重采样
│   ├── asr.py / tts.py / llm.py / vad.py
│   └── static/index.html    网页控制台
├── data/                    运行时数据（成语/古诗/单词/故事/音色表/对话记录）
├── raw/                     原始语料（成语/古诗/单词/故事），构建脚本的输入
├── models/                  离线模型
├── _pylibs/                 随项目走的依赖（httpx / websockets / pypinyin…）
├── _build_data.py           从 raw/ 重建 data/ 下的词库
├── _build_poems.py          重建 data/poems.json（小学必背 + 扩展）
└── _bench_prewarm_ab.py / _bench_tts_pool.py   延迟复测脚本
```

## 七、自检

不开硬件也能验证：

```bash
python sim_device.py
```

它会用模型自带的测试音频模拟设备上行，然后打印 STT/LLM/TTS 结果和下行音频时长。

## 八、常见问题

**设备连不上**
- 确认电脑和设备的 IP 在同一网段，防火墙放行 8003 端口
- 控制台「实时日志」里看有没有 `OTA 请求` 和 `设备接入`

**能识别但没声音**
- 看日志有没有 `TTS 无输出`（edge-tts 需要联网）
- 换 `tts.engine: sapi` 试试本地合成

**识别不准**
- 换更大的 paraformer 模型，或调 `vad.min_silence_ms`

**说完没反应**
- 调小 `vad.min_silence_ms`（`config.yaml` 里当前是 450ms），让它更快判定说完

## 九、运行依赖（缺包会 500）

服务要 `httpx`（LLM 的 OpenAI 兼容客户端）、`websockets`（猫剪 TTS 的 WSS）、
`pypinyin`（查字拼音的工具 `pinyin_lookup`）。这三个包在 `server/llm.py`、
`server/tts_maoxiang.py`、`server/kid_tools.py` 里都是**函数内懒加载**的，
所以缺包时服务照常启动、网页也能打开，但一调 LLM/TTS 就失败：

- `/api/test-preset`、`/api/test-llm` 报 500
- `/api/voices/test` 直接返回 `{"ok": false, "error": "合成失败：No module named 'websockets'"}`
- 查字拼音时工具报 `No module named 'pypinyin'`

如果用 WSL 的系统 Python（`/usr/bin/python3`）跑，site-packages 里没有这两个包，
`pip install` 也装不进去（PEP 668 externally-managed + 根文件系统只读）。
所以依赖随项目放在 `_pylibs/`（约 4MB），由 `server/__init__.py` 自动挂到 `sys.path`：

| 路径 | 内容 |
|---|---|
| `_pylibs/httpx` | httpx 0.28.1 |
| `_pylibs/httpcore` | httpcore 1.0.9 |
| `_pylibs/websockets` | websockets 15.0.1 |
| `_pylibs/h11`、`_pylibs/sniffio` | 上面两个的依赖 |
| `_pylibs/pypinyin` | pypinyin 0.55.0（纯 Python） |
| 项目根目录的 `httpx`/`websockets`/`pypinyin`/… 软链接 | 让**已经在跑**的进程不用重启也能用上 |

两种启动方式都验证过：

- Windows Python 3.11（`启动控制中心.bat`）：系统自带 httpx 0.28.1 / websockets 15.0.1，直接可用；
- WSL `/usr/bin/python3`：靠 `_pylibs/` 补齐（重启后由 `server/__init__.py` 引导生效）。

`_pylibs/` 丢了就在项目根目录重装一份：

```bash
cd xiaozhi-local
python3 -m pip install --no-deps --target ./_pylibs httpx==0.28.1 websockets==15.0.1 httpcore h11 sniffio pypinyin==0.55.0
for p in httpx httpcore websockets h11 sniffio pypinyin; do ln -sfn _pylibs/$p $p; done
```

## 十、当前实测延迟（2026-10-24，WSL）

| 环节 | 耗时 |
|---|---|
| 控制中心「试一句」`/api/test-preset` | 中位 **1.6s**（改造前 7.58s） |
| 同一 prompt 直连网关 `127.0.0.1:9100` | 1.65~1.81s（控制中心已无额外损耗） |
| LLM 首字（流式） | 1.6~2.5s（主要瓶颈在模型本身） |
| TTS 单句合成 | 0.9~1.0s（裸 WSS 握手约 0.1s） |
| 语音首包（LLM 首句 + 合成） | 2.7~3.5s |

复测脚本：`_bench_prewarm_ab.py`（全链路 A/B）、`_bench_tts_pool.py`（TTS 连接池 A/B）。

## 十一、词库重建（data/ 与 raw/）

儿童内容工具（成语链、背古诗、查单词、讲故事）读的是 `data/` 下的成品文件，
它们的**原始语料**放在 `raw/`，由两个构建脚本生成：

| 脚本 | 产出 | 数据源（raw/ 下） |
|---|---|---|
| `python3 _build_data.py` | `data/idioms_dict.txt`、`data/english_words.json`、`data/words_index.json`、`data/stories.json` | `idiom.json`、`word.json`、PEP 小学单词 8 册、`aesop.csv`、`grimms/` |
| `python3 _build_poems.py` | `data/poems.json` | `k12_poems.json`、`poem300.json` |

两个脚本都用 `__file__` 定位项目根，Windows / WSL 都能直接跑，**幂等**：
同样的语料重复构建结果完全一致。注意 `raw/k12_poems.json` 是 10-24 下午换过的版本，
所以重建后的 `data/poems.json`（354 首）与旧版略有出入，属正常。

目录里 `_` 前缀的文件只有上面这几个是「留着的」，其余 `_` 开头的脚本/中间产物
（探针、基准、中间 JSON）都是调试残留，可以随时删。

## 开源版配置说明

1. cp config.example.yaml config.yaml
2. 按需修改：server.public_host（对外地址）、server.token（设备鉴权）、llm.presets.*.api_key、mcp.weather_city
3. pip install -r requirements.txt
4. python run.py（默认监听 0.0.0.0:8003）
5. 固件侧把 OTA 地址指向 http://<你的服务器IP>:8003/xiaozhi/ota/

config.yaml、data/、models/ 均不入库。
