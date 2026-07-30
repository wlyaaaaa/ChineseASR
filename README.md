# ChineseASR

`ChineseASR` 是一个本地优先的中文语音转文字项目，目标是把中文录音转成可审计、低幻觉、可复现的文本。它面向 Windows + CUDA 工作站，默认 quick 使用 `SenseVoiceSmall`，strict 使用 `Qwen3-ASR-1.7B + SenseVoiceSmall`。可选的 `FireRedASR2-LLM` 是证据级词汇主引擎，不会因为安装完成而自动取代默认 strict 组合。

这个项目优先解决三件事：

- **中文准确性优先**：strict 模式默认以 `Qwen3-ASR-1.7B` 为主引擎，`SenseVoiceSmall` 为对照锚点。
- **低幻觉和可复核**：双模型分歧、静音出字、模板废话、异常重复、繁体残留、超长无标点等都会进入 audit / metrics / review。
- **适合 AI Agent 调用**：`scripts\asr-smart.ps1` 通过本地 API 提交任务，快速返回 job 状态，避免长时间卡住命令行或上层 Agent。

它不是云转写服务，不会默认上传音频。模型、输出、wheelhouse 和私人评测数据都保留在本机。

## 当前状态

`personal-use v0.1` 已完成收尾，进入维护态。关闭标准是：

1. `scripts\doctor.ps1` 能确认无代理、CUDA、模型配置和依赖状态。
2. 单元测试全通过。
3. `scripts\smoke-asr-smart.ps1 -Json` 能完成 strict smart job。
4. 公开仓库只包含源码、脚本、配置、测试和文档，不包含模型权重、用户音频、生成转写或 wheelhouse 大文件。

后续真实录音 benchmark、模型组合微调、Ollama 仲裁启用、VAD 边界切片优化都属于使用阶段校准，不是当前版本的关闭阻塞项。

## 适合场景

- 微信语音、会议录音、口述笔记、中文播客、课程录音的本地转写。
- 对“幻觉低、能复查证据”要求高于“只要快”的中文 ASR。
- 需要让 Codex、脚本或其他本地 Agent 调用 ASR，又不希望命令行长时间阻塞。
- 需要保留 raw JSON、audit、metrics、manifest，方便以后追溯模型版本和输入 hash。

## 不适合场景

- 只想要一个极简 GUI。
- 不愿意本地安装模型权重和 Python 环境。
- 需要直接把音频交给云厂商转写。
- 需要英文、多语种或字幕生产工具链作为主目标。

## 默认模型策略

模型注册在 `configs\models.yaml`，实现和模型选择解耦。

| 用途 | 默认引擎 | 说明 |
| --- | --- | --- |
| `strict` 主引擎 | `qwen3-asr-1.7b` | 准确率优先的中文主转写线，基于 Qwen3-ASR 权重和 `qwen-asr` runtime |
| `strict` 对照引擎 | `sensevoice` | 快速中文声学锚点，用于发现分歧和疑似幻觉 |
| `quick` | `sensevoice` | 单模型快速转写 |
| 可选证据级词汇主引擎 | `fireredasr2-llm` | 隔离在 WSL 中运行；仅在显式选择时作为 strict 主引擎 |
| baseline | `paraformer` | 保守普通话基线，可用于回归对照 |
| fallback/comparison | `whisper-large-v3` | 已注册为备用/对照，不作为中文 strict 默认路径 |

strict 模式即使一路模型失败，也会保留可用输出并生成审计包。正文会标记 `[疑似]`，`strict.audit.md` 会记录失败引擎、异常摘要和复核理由。两路都失败时输出 `[听不清]`。

## 一分钟使用

在仓库根目录打开 PowerShell：

```powershell
cd <repo-root>
.\scripts\doctor.ps1
.\scripts\asr-smart.ps1 -Audio C:\path\to\audio.wav -Mode strict -WaitSec 15 -Json
```

推荐日常入口是 `asr-smart.ps1`，因为它会走本地 API/job 队列，不会把调用方长时间卡死。

常用模式：

```powershell
# 严格双模型，推荐默认
.\scripts\asr-smart.ps1 -Audio C:\path\to\audio.wav -Mode strict -WaitSec 15 -Json

# 长音频，自适应引擎上限切片 + 断点续跑
.\scripts\asr-smart.ps1 -Audio C:\path\to\long.wav -Mode long-strict -WaitSec 15 -Json

# 显式用 FireRedASR2-LLM 作为词汇主引擎；不会改变默认配置
.\scripts\asr-smart.ps1 -Audio C:\path\to\long.mp3 -Mode long-strict -PrimaryEngine fireredasr2-llm -SecondaryEngine sensevoice -WaitSec 15 -Json

# 快速单模型，只在明确接受较少审计时使用
.\scripts\asr-smart.ps1 -Audio C:\path\to\audio.wav -Mode quick -WaitSec 15 -Json

# 批量转写文件夹
.\scripts\transcribe-folder.ps1 -InputDir C:\path\to\audio-folder
```

固定端到端 smoke：

```powershell
.\scripts\smoke-asr-smart.ps1 -Json
```

## 安装与模型下载

先安装 CUDA 版 PyTorch 和核心依赖：

```powershell
.\scripts\install-torch-cu128-direct.ps1
.\scripts\setup-core.ps1
```

下载 quick / secondary 默认需要的 SenseVoice：

```powershell
.\scripts\download-models.ps1 -Engine sensevoice
```

strict 主线需要 Qwen ASR runtime 和权重：

```powershell
.\scripts\setup-qwen.ps1
.\scripts\download-models.ps1 -Engine qwen3-asr-1.7b
```

可选的 FireRedASR2-LLM 使用独立 WSL runtime，安装顺序如下：

```powershell
.\scripts\setup-firered.ps1
.\scripts\download-models.ps1 -Engine fireredasr2-llm
```

默认 WSL 虚拟环境是 `/opt/chineseasr/firered/.venv`。源码和权重分别放在 Git 忽略的 `models/firered/FireRedASR2S`、`models/firered/FireRedASR2-LLM`：

- 源码固定为 commit `4e7d9aaf4482a47cec1724807026b9b151926eb5`。
- 模型固定为 revision `2c5e0f415b9afb8f67cb8b00ea4c54959f70e824`。
- 下载完成后生成 `MODEL_RECEIPT.json`，记录 14 项必要权重文件的规范顺序、路径、大小和 SHA-256。运行时逐项校验 receipt、固定 revision、源码 HEAD 和干净工作树，任一不一致都会拒绝装载。

FireRed 输入契约是 16 kHz、16-bit、mono PCM WAV；MP3 等输入由前端经 ffmpeg 生成派生 WAV。单输入硬上限为 40 秒，长音频的推荐有效切片为 35 秒。

启用 `use_half: true` 且使用 CUDA 时，隔离 worker 会让 Qwen 基座首次装载即使用 BF16（GPU 不支持 BF16 时使用 FP16），避免官方实现先完整物化 FP32 权重、随后才转半精度造成的内存峰值。临时装载桥在成功或异常后都会恢复官方绑定，固定源码 checkout 不会被改写；raw 结果会记录 `llm_initial_load_dtype`。

FireRed 的 CPU 装载峰值仍高于普通 ASR。本机 64GB Windows + RTX 5090 D 的已验证 WSL 配置是：

```ini
[wsl2]
memory=32GB
swap=8GB

[experimental]
autoMemoryReclaim=gradual
```

修改 `%UserProfile%\.wslconfig` 后需在没有重要 WSL/Docker 任务时执行 `wsl --shutdown` 再重新启动。隔离 worker 会在大权重哈希和装载前同时检查配置容量与当前可用容量，并在无法读取 Linux CUDA 内存信息时 fail-closed。半精度装载的最低配置门槛为 28 GiB RAM、34 GiB RAM+swap，启动当下还需至少 18 GiB `MemAvailable`、22 GiB `MemAvailable+SwapFree`；不满足时会区分“配置不足”和“当前占用过高”并给出可操作错误。这里的 32GB/8GB 是本机验证值，不是对所有硬件的统一承诺。

常规模型下载与 FireRed runtime setup 会清理代理环境；FireRed 的 Hugging Face 权重下载保留调用进程当前代理设置，以适应实际联网环境。常规模型缓存位于 `models\modelscope`；这些模型目录都不进入 Git。

## 输出文件

strict 模式会把最终稿和证据拆开：

| 文件 | 用途 |
| --- | --- |
| `*.strict.md` | 给人看的最终转写正文，尽量保持干净 |
| `*.strict.audit.md` | 模型文本、相似度、分歧、flags、候选文本和判断依据 |
| `*.strict.audit.json` | 机器可读审计报告 |
| `*.qwen3-asr-1.7b.raw.json` | 主模型原始输出 |
| `*.sensevoice.raw.json` | 对照模型原始输出 |
| `*.fireredasr2-llm.raw.json` | 显式选择 FireRed 主引擎时的原始输出 |

长音频和评测流程还会写：

| 文件 | 用途 |
| --- | --- |
| `manifest.json` | 输入 hash、模型配置 hash、chunk 参数、chunk 状态 |
| `metrics.json` | 耗时、相似度、风险标记、模型和运行时信息 |
| `review.md` | 最值得人工复核的片段队列，按 P0/P1/P2 排序 |
| `benchmark.md` / `benchmark.json` | 和 truth 文本对齐后的评测结果 |

不要把 raw JSON 当最终稿。常规读取顺序是：先看 `outputs.final` 或 `*.strict.md`，再看 `audit.md`、`audit_json`、`metrics.json` 和 `review.md`。

顶层 job 的 `succeeded` 只表示流程产出了结果；机器消费者必须同时读取 `evidence_status`：

- `verified`：要求的双引擎链完整执行，但不表示转写逐字准确，仍需看 `status`、分歧和人工复核项；
- `provisional`：至少一路引擎失败，现有文本只是带失败证据的回退结果；
- `unavailable`：证据链或必要产物不完整；
- `pending`：仍在处理；
- `not_applicable`：quick 单引擎任务不适用证据级双引擎状态。

显式使用 FireRed 时，若 FireRed 失败而对照引擎成功，流程仍会生成带 `[疑似]` 的回退文本，但 job、长音频 manifest 和对应 chunk 都会标为 `provisional`，并列出 `evidence_failures`。状态判定会交叉验证 final、audit、audit JSON 和两路 raw JSON 均存在且可解析，并核对 audit 中的引擎身份、执行状态、错误与 raw 结果一致；声称 `verified` 但缺少或损坏必要产物时会降为 `unavailable`。证据级验收还必须确认每个 FireRed raw JSON 文本非空、`error=null`、运行时 dtype 正确，并且逐段审计不含 `engine_failure`；`verified` 不能替代对原始录音的人工核听。

## 本地 API 与 Smart Wrapper

本地 API 只绑定 `127.0.0.1`：

```powershell
.\.venv\Scripts\python.exe -m zh_asr serve --host 127.0.0.1 --port 18666 --state-dir outputs\api
```

主要端点：

- `GET /health`
- `GET /jobs`
- `GET /jobs/{job_id}`
- `GET /observer/jobs`
- `GET /observer/jobs/{job_id}`
- `POST /jobs/transcribe`
- `POST /jobs/{job_id}/cancel`

`/observer/*` 是只读安全投影，供本机统一观察台读取。它只返回任务状态、模式、逻辑模型名、时间、可用的长音频 chunk 计数与终态 RTF；ASR 不产生通用 LLM token 指标，因此 token 状态固定为 `not_applicable`。投影不返回音频/输出路径、PID、命令、stdout/stderr、识别正文、证据完整性状态或 GPU Broker 信息。证据消费者必须读取 `/jobs` 或 `/jobs/{job_id}` 中的 `evidence_status` 与 `evidence_failures`，不能用 observer 投影替代证据验收。

`asr-smart.ps1` 会在需要时启动本地 API，提交 job，并在 `WaitSec` 内等待结果。如果任务仍在运行，它会返回 job id 和下一步查询命令，而不是无限等待。

为了避免和 Ollama、LocalOCR、LM Studio 或其他 Python 模型抢 GPU，服务会读取 `nvidia-smi --query-compute-apps`。发现外部 CUDA compute 进程时默认返回 `blocked`；只有显式使用 `-AllowGpuConflicts` 或 API 参数 `allow_gpu_conflicts=true` 才会继续排队。

默认端口是 `18666`，刻意避开 LocalOCR 的 `18665`。

RTX 5090D 32GB 显存较大时，这个 GPU 排他锁仍然是保守默认值，不代表硬件不能并发。确认当前任务可以和 Ollama、LocalOCR 或其他 CUDA 进程共用显存时，再显式使用 `-AllowGpuConflicts` / `allow_gpu_conflicts=true`。

默认入口还会向 `http://127.0.0.1:32100/_gpu_broker/*` 申请全机 GPU 租约。Broker
会在 ASR 启动前卸载空闲 Ollama/LocalOCR，并在 ASR 运行期间阻止新的 Ollama 或 OCR
重型推理。`-AllowGpuConflicts` 同时绕过旧进程检测和统一 Broker，只能在用户明确接受
多模型并发时使用；Broker 不限制单个 ASR 任务本身的显存或内存占用。

## 长音频与断点续跑

长音频入口：

```powershell
.\scripts\asr-smart.ps1 -Audio C:\path\to\long.mp3 -Mode long-strict -WaitSec 15 -Json
```

`ChunkSec` 是请求值，不是无条件采用的固定值。实际 `effective_chunk_sec` 会取请求值与两路引擎能力上限中的最小值；选择 FireRed 时为 35 秒，并始终低于其 40 秒单输入硬上限。每个 chunk 都跑 strict 双引擎。

MP3 等非标准输入先统一为 16 kHz、16-bit、mono PCM WAV。schema 2 的 `manifest.json` 记录源文件与派生文件 SHA-256、转换 provenance、模型配置 hash、请求/有效切片参数、已解析引擎和 chunk 状态。内容与运行参数共同形成 fingerprint：一致时可跳过已有成功输出，残留 `running` 会转为 `stale` 后重跑；内容或配置变化时不会误用旧结果。

待处理 chunk 按两路引擎最小 `max_request_inputs` 分成有界批。FireRed 配置为每批最多 16 个输入；每个批次中，每个引擎只加载一次，再处理该批所有 chunk，避免逐片重复加载模型。

strict audit v2 保留两路引擎的原始文本、raw JSON 引用、provenance、分歧和人工复核项。选择策略固定为保留主引擎证据：不做多数投票，也不做语义改写。

## LLM 仲裁

LLM 仲裁配置在 `configs\models.yaml` 的 `llm_arbitration`，默认关闭。

当前设计是本地 Ollama evidence-only 仲裁：

- 只读取 ASR audit 证据，不读取音频。
- 只在 chunk 有 `flags`、`needs_review` 或低相似度时触发。
- 仲裁结果写入 merged audit / metrics。
- 不覆盖 raw ASR JSON。
- 默认 `keep_alive=0`，避免长期占用 GPU。

默认关闭是有意设计：基础转写链路必须在没有 Ollama、没有额外 GPU 驻留、没有 LLM 最终猜测时也能稳定工作。

## 评测与 Benchmark

生成隐私友好的本地合成/对抗评测集：

```powershell
.\scripts\eval.ps1 -Generate -GenerateOnly
```

运行内置评测：

```powershell
.\scripts\eval.ps1 -Generate -Force
```

用自己的音频和人工 truth 文本跑 benchmark：

```powershell
.\scripts\benchmark.ps1 -AudioDir C:\path\to\audio -TruthDir C:\path\to\truth
```

benchmark 按文件 stem 匹配音频和 truth，写 `_manifest\manifest.json`，不会复制你的源音频或 truth 文件。真实私人录音 benchmark 是校准手段，不需要提交到公开仓库。

## 离线 Wheelhouse

冻结当前依赖：

```powershell
.\scripts\export-lock.ps1
```

下载 wheelhouse：

```powershell
.\scripts\build-wheelhouse.ps1
.\scripts\verify-wheelhouse.ps1
```

离线安装 smoke：

```powershell
.\scripts\install-offline.ps1 -Venv .venv-offline-smoke
```

`offline\wheelhouse\` 被 Git 忽略。小型 lock/checksum manifest 可以放在 `offline\manifests\` 下追踪。

## 模型替换

同一 adapter 内换模型，优先只改 `configs\models.yaml`：

- `defaults.engine`：quick 默认引擎。
- `strict.primary_engine`：strict 主引擎。
- `strict.secondary_engine`：strict 对照引擎。
- `engines.*.adapter`：运行时适配器，目前有 `funasr`、`qwen-asr` 和 `firered-worker`。
- `llm_arbitration`：本地 Ollama 仲裁配置，默认关闭。

临时使用其他配置：

```powershell
$env:ZH_ASR_MODEL_CONFIG='C:\path\to\models.yaml'
```

换模型后的基本验证：

```powershell
.\scripts\download-models.ps1 -Engine <engine-name>
.\scripts\strict.ps1 -Audio C:\path\to\audio.wav
.\scripts\smoke-asr-smart.ps1 -Json
```

新增不同运行时，例如 Whisper 本地实现、其他 LLM 音频模型或云 API，应新增 adapter，并保持 pipeline 只依赖统一的 `generate(...)` 输出形状。

## 测试

单元测试不加载 ASR 模型：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

环境体检：

```powershell
.\scripts\doctor.ps1
```

端到端 smoke：

```powershell
.\scripts\smoke-asr-smart.ps1 -Json
```

公开发布前建议再跑：

```powershell
git diff --check
git status --ignored=matching --short
```

另外按公开发布流程对 `README.md`、`docs`、`src`、`tests`、`scripts`、`configs` 做 source-only 秘密模式扫描，确认没有 token、私钥、完整 `.env`、OAuth JSON、原始日志或私人输出内容。

## 公开仓库边界

仓库追踪：

- `src\zh_asr\`
- `scripts\`
- `configs\models.yaml`
- `tests\`
- `docs\`
- `offline\manifests\`

仓库不追踪：

- `.venv\`
- `models\`
- `outputs\`
- `eval\corpus\`
- `offline\wheelhouse\`
- Python cache 和 build artifacts

这些目录可能包含模型权重、生成转写、私人音频路径、评测材料或大型 wheel 文件。

## 常见问题

**为什么 strict 比 quick 慢？**

strict 会跑两路模型并写审计文件，目标是低幻觉和可复核；quick 只跑单模型。

**为什么 smart 返回 `blocked`？**

本机已有其他 CUDA compute 进程。默认行为是保护 GPU，不和 Ollama、LocalOCR、LM Studio 等抢资源。确认可以抢占时再加 `-AllowGpuConflicts`。

**为什么正文里有 `[疑似]` 或 `[听不清]`？**

这表示模型证据不足、两路分歧大、某一路失败或命中风险规则。正文不强装确定，细节看 `strict.audit.md` 和 `strict.audit.json`。

**是否会自动使用 LLM 猜最终答案？**

不会。Ollama 仲裁默认关闭，并且只读取 audit 证据。需要时在 `configs\models.yaml` 显式打开。

**换更强模型要改代码吗？**

通常不用。优先改 `configs\models.yaml`，只有新增运行时时才写 adapter。

## 更多文档

- [架构说明](docs/architecture.md)
- [公开发布边界](docs/public-release.md)
