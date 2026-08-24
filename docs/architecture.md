# 架构说明

## 目标

`ChineseASR` 面向中文严谨转写：先保证不幻觉、不乱编，再追求速度和格式化效果。

## 引擎策略

引擎不再硬编码在 Python 实现里，而是注册在 `configs/models.yaml`：

1. `defaults.engine` 决定 quick 模式默认模型。
2. `strict.primary_engine` / `strict.secondary_engine` 决定严格模式双模型组合。
3. `aliases` 记录 VAD、标点、说话人等可复用模型别名；`speaker_verification` 只配置按需的本地 `person:self` CAM++ 锚，不进入默认转写。
4. `engines.*.adapter` 决定运行时适配器；当前已实现 `funasr`、`qwen-asr` 和 `firered-worker`。

当前默认策略仍然是：

1. `qwen3-asr-1.7b`：strict 准确率优先主线。基于 Qwen3-ASR 官方开源权重和 `qwen-asr` runtime。
2. `sensevoice`：quick 默认和 strict 低幻觉锚点。组合 `iic/SenseVoiceSmall`、`fsmn-vad`、`ct-punc`；默认明确**不**加载 `cam++`。
3. `fireredasr2-llm`：可选的证据级词汇主引擎，仅在显式选择时进入 strict；安装或下载不会改变默认组合。
4. `paraformer`：中文保守备用线。可产出时间戳；显式 `cam++` 输出是 diarization/匿名聚类而非用户身份确认，可能过拆或合并。
5. `whisper-large-v3`：只记录为 fallback/comparison，不自动作为主输出。

`speaker-enroll` / `speaker-evidence` 在真实问题需要时才按需读取一个有限音频片段。enroll 只允许创建一个带单句可回查依据、可替换且永远为 `inferred` 的本机私有 `person:self` 向量；目标片段只输出源哈希、时间、模型哈希和相似度。它不建设通用声纹平台，也不让分数、匿名 cluster 或默认下混音频单独产生 `confirmed`。归属投影保留来源、联系人、声道、对话角色、句义、跨录音与声纹的各项理由，但不计算固定权重、合成分数或长期置信等级：具体的来源/语义判断可以解释性地压过相反的弱声学线索，无法合理消解的实质冲突才是 `unknown`；`confirmed` 仍要求权威来源引用。

FireRed 通过 Windows adapter 调用隔离的 WSL worker。默认 WSL Python 为 `/opt/chineseasr/firered/.venv/bin/python`；源码和权重位于 Git 忽略的 `models/firered/FireRedASR2S` 与 `models/firered/FireRedASR2-LLM`。固定来源是：

- FireRedASR2S commit：`4e7d9aaf4482a47cec1724807026b9b151926eb5`
- FireRedASR2-LLM revision：`2c5e0f415b9afb8f67cb8b00ea4c54959f70e824`

`scripts/setup-firered.ps1` 创建隔离环境并检出固定源码；`scripts/download-models.ps1 -Engine fireredasr2-llm` 下载固定模型 revision，并生成带 14 项必要文件大小及 SHA-256 的规范 `MODEL_RECEIPT.json`。运行时校验 receipt schema、仓库、revision、精确文件清单、路径边界、大小、SHA-256，以及源码 HEAD 和 tracked/untracked 干净工作树；任一不一致均 fail-closed。

Qwen runtime 固定为 `qwen-asr==0.0.6`，模型固定为 revision
`a04930dbe5419bfee073f7cade734f572689a3a8`。Qwen 的规范
`MODEL_RECEIPT.json` 绑定 13 个必要模型文件；runtime 版本、revision、收据、大小或
SHA-256 任一漂移都会在 `Qwen3ASRModel.from_pretrained` 前 fail-closed。重要录音使用
`FireRedASR2-LLM + Qwen3-ASR-1.7B`，默认日常 strict 仍保持
`Qwen3-ASR-1.7B + SenseVoiceSmall`。

官方 checkpoint 内的 `args.use_fp16=0` 会使 Qwen2-7B 先按 FP32 装载，而运行参数 `use_half` 原本要等整个模型构造完成后才转 BF16。隔离 worker 在 `use_half=true` + CUDA 时，仅在官方 `from_pretrained` 调用窗口内将首次 LLM 装载 dtype 约束为 BF16（不支持时 FP16），并在成功或异常后恢复官方绑定。这不修改固定源码或模型 checkpoint，raw 结果记录实际 `llm_initial_load_dtype`。

FireRed worker 在校验大权重及装载前读取 WSL `/proc/meminfo`，同时检查配置总量和启动当下可用量；Linux CUDA 环境无法读取完整内存信息时 fail-closed。半精度装载要求至少 28 GiB RAM、34 GiB RAM+swap，并要求 `MemAvailable` 至少 18 GiB、`MemAvailable+SwapFree` 至少 22 GiB；FP32 对应门槛为 40/48 GiB 与 36/44 GiB。错误会区分配置不足和当前占用过高，并给出 `.wslconfig`、关闭占用进程与 `wsl --shutdown` 提示，避免模型装载后才被内核 OOM 直接 SIGKILL。本机 64GB Windows 工作站已用 WSL 32GB RAM + 8GB swap 完成真实 CUDA 验收；这是一项本机运行事实，而不是跨硬件统一阈值承诺。

## 数据流

```text
audio -> VAD split -> ASR engine -> raw JSON -> Markdown transcript
```

严格模式：

```text
audio
  -> Qwen3-ASR raw JSON
  -> SenseVoice raw JSON
  -> normalized text comparison
  -> strict.md + strict.audit.md + strict.audit.json
  -> strict.review.json + strict.receipt.json
```

`strict.md` 是给人读的最终稿，正文尽量干净；当两个模型严重冲突、都为空、或出现常见幻觉套话时才写入 `[疑似]` / `[听不清]`。audit v2 保存两路 raw JSON 引用、逐引擎原文与 provenance、分歧、风险规则和人工复核项。选择策略是 `primary_preserving_no_majority_vote_no_semantic_rewrite`：不做多数投票，不把语义改写冒充声学证据。

如果 strict 中某个引擎抛错，流程不会直接丢弃整次任务。成功的一路会继续进入最终猜测，正文标记 `[疑似]`，审计报告状态为 `engine_failure`，并在 raw JSON 中保留失败引擎、异常类型和错误摘要。如果两路都失败，则输出 `[听不清]` 并要求人工复核。

空转写的声学语义由独立的 `media.objective-result.v1` sidecar 表达，不增加旧 strict `engine_evidence` 条目。`speech_transcribed`、`no_speech_detected`、`speech_detected_but_not_transcribable` 和 `indeterminate` 是唯一的客观结果；空文本、空数组、零字节或 `expect_empty` 本身不能证明无语音。sidecar 的 execution/coverage/quality 正式状态分别是 `completed|failed|unsupported|corrupt`、`complete|partial|unknown`、`sufficient|low_confidence|unknown`，调用方治理绑定只通过 `caller_binding` 透传。

任务执行状态和证据完整性状态相互独立。顶层 `succeeded` 只说明命令成功产出；`evidence_status` 在 strict audit、长音频 manifest/chunk 和 API job 上统一使用 `pending / verified / provisional / unavailable / not_applicable`。其中 `verified` 要求双引擎链完整执行，并要求 final、audit、review、两路 raw 通过 `strict.receipt.json` 的路径、大小、SHA-256 与语义交叉验证，但仍不保证文本逐字正确；有引擎失败但仍产出回退文本时必须是 `provisional`，必要产物缺失、未同步重建收据的修改、语义错配或任务失败时必须是 `unavailable`，quick 单引擎任务为 `not_applicable`。该 receipt 只证明包内一致性，不是数字签名或可信时间戳，外部真实性仍以原始录音和独立保全链为准。

因此，显式选择 FireRed 的证据级验收不能只检查顶层 job 为 `succeeded`。还必须逐段确认 `evidence_status=verified`、FireRed raw 文本非空、`error=null`、初始装载 dtype 符合配置，且 strict audit 不含 `engine_failure`；否则只算有审计记录的对照引擎回退。即使状态为 `verified`，精确引用前仍要回听原始录音。

长音频模式：

```text
long audio
  -> MP3/other input -> 16 kHz, 16-bit, mono PCM WAV
  -> capability-bounded chunks + schema 2 manifest.json
  -> bounded batches; each engine loads once per batch
  -> each chunk writes strict audit v2
  -> skip completed chunks on resume
  -> optional uncertain-only Ollama arbitration
  -> transcript.md + audit.md + metrics.json
```

`chunk_sec` 是请求值。planner 使用两路引擎能力计算 `effective_chunk_sec`；选择 FireRed 时，单输入硬上限为 40 秒，推荐有效切片为 35 秒，因此不会沿用请求中的 300 秒作为实际切片。overlap 必须小于有效切片。

前端通过 ffmpeg 把 MP3 等输入转换为 16 kHz、16-bit、mono PCM WAV，并保留源文件与派生文件 SHA-256、ffmpeg 版本和转换命令 provenance。schema 2 manifest 还记录模型配置 hash、固定模型/runtime 收据、运行代码身份、请求/有效切片、已解析引擎、chunk 状态和 run fingerprint。resume 会重新运行统一 bundle verifier，不信任 manifest 中缓存的 `evidence_status`；只有运行身份一致且收据覆盖的内容仍完整时才跳过。残留 `running`、收据缺失、哈希/语义错配都会标记为 `stale` 后重跑。

待处理 chunk 按两路引擎中较小的 `max_request_inputs` 分批。FireRed 的上限是 16；每个批次先加载主引擎一次并处理整批，再加载对照引擎一次并处理整批。该边界控制内存与 IPC 规模，也避免逐 chunk 重复加载模型。

请求 fingerprint 包含音频内容 hash、设备、GPU 冲突策略、规范化缓存目录、模型配置、固定源码/模型 revision、模型/runtime receipt SHA-256 和运行代码 SHA-256。相同长音频请求在失败或取消后复用稳定输出目录；schema 1 或身份变化均安全重跑。Windows 子进程树、带 job token 的 WSL worker、ffmpeg 和 LocalGpuBroker lease 都有明确的超时、取消和终态释放路径。

LLM 仲裁默认关闭，配置在 `configs/models.yaml` 的 `llm_arbitration`。当前 provider 是本地 Ollama，默认模型 `qwen-main-v1:latest`，`keep_alive=0`。仲裁只读取 chunk audit 证据，不读取音频；只在 `flags`、`needs_review` 或低相似度 chunk 上触发。仲裁结果写入 merged audit / metrics，不覆盖原始 strict raw JSON。

## 本地 API / Smart 调用层

面向 Codex 和自动化调用时，不直接长时间等待 `strict.ps1`。推荐路径是：

```text
scripts\asr-smart.ps1
  -> 127.0.0.1 local API
  -> in-memory job queue
  -> single GPU worker
  -> child process: python -m zh_asr strict/transcribe
  -> outputs\api\<job_id>\
```

API 入口由 `python -m zh_asr serve --host 127.0.0.1 --port 18666` 提供，主要端点是 `/health`、`/jobs`、`/jobs/{job_id}`、`/jobs/transcribe` 和 `/jobs/{job_id}/cancel`。`/observer/jobs` 与 `/observer/jobs/{job_id}` 提供稳定的只读安全投影，只暴露调度元数据、可验证的 chunk 计数和终态 RTF，不暴露输入/输出路径、进程与命令信息、日志、转写正文、证据完整性状态或 Broker 细节；证据消费者必须改读 `/jobs*` 的 `evidence_status/evidence_failures`。

服务层只负责调度，不在 HTTP 请求线程中加载模型。每个任务在独立 Python 子进程里运行现有 CLI，这样 API 可以快速返回、任务可以取消、模型显存也不会长期留在 API 进程里。

为避免和其他本地模型互相抢 GPU，所有公开 CLI 和 smart/API 路径统一取得
LocalGpuBroker 租约并失败关闭。服务父进程持有、续期和释放租约，工作子进程必须携带
opaque token 并向 Broker 验证当前 live owner；裸环境标记不能绕过。直接 CLI 由持租约
的监督进程启动可终止工作子进程。任一路径续租失败都会立即终止完整子进程树；服务任务
写入 `gpu_broker_lost`，不会吞掉续期错误后继续运行。旧
`allow_gpu_conflicts=true` / `scripts\asr-smart.ps1 -AllowGpuConflicts` 只保留为无
机器级 Broker 嵌入场景的外部 CUDA 进程检测兼容字段，不能绕过正式 LocalGpuBroker。
正式 Broker 的协调域是已接入的 Ollama、LocalOCR 与 ChineseASR，不把 LM Studio 或任意
未接入的 CUDA 进程误报为已受管。

当前机器是 RTX 5090D 32GB，默认排他锁属于受管工作负载之间的保守调度策略，不是硬件
能力判断；未接入 Broker 的 CUDA 工作负载由调用者另行协调。

固定端到端验收入口是：

```powershell
.\scripts\smoke-asr-smart.ps1 -Json
```

它会使用本地模型缓存自带的中文样例音频，强制跑一次 strict smart job，并校验 `final`、`audit`、`audit_json`、`primary_raw_json` 和 `secondary_raw_json` 都存在。

重要录音的完整证据链验收入口是：

```powershell
.\scripts\smoke-evidence-asr.ps1 -Audio C:\path\to\important.mp3 -Json
```

该入口固定使用 FireRed + Qwen，逐 chunk 要求 `succeeded + verified`、完整
strict receipt、两路成功且非空、无 `engine_failure`，并检查 FireRed 初始装载 dtype。

## 模型替换边界

同一 adapter 内替换模型时，优先只改 `configs/models.yaml`，然后运行：

```powershell
.\scripts\download-models.ps1 -Engine <engine-name>
.\scripts\strict.ps1 -Audio C:\path\to\audio.wav
```

新增不同运行时，例如 Whisper 本地实现、其他 LLM 音频模型或云 API 时，应新增 adapter，并保持 pipeline 只依赖统一的 `build_model(...)` / `generate(...)` 形状。当前已有 `funasr`、`qwen-asr` 和隔离 WSL 的 `firered-worker`。

## 已实现的审计闭环

当前个人使用版已经闭合以下能力，作为已交付范围记录：

- 输入与运行可追踪：`manifest.json` / `metrics.json` 记录音频 hash、truth hash、模型配置 hash、选中模型、命令、运行时和耗时。
- 双模型审计：strict audit v2 输出 `strict.md`、`strict.audit.md`、`strict.audit.json`、`strict.review.json`、`strict.receipt.json` 和两路 raw JSON，保留 provenance、分歧和人工复核项；收据绑定全部内容，不做多数投票或语义改写。新包使用包内相对引用，整目录复制后仍能复验；旧绝对引用仅在原位置兼容。
- 幻觉规则库：静音出字、常见模板废话、异常重复、双模型大分歧、繁体残留、超长无标点都会进入 audit / metrics / review。
- 评测与 benchmark：内置隐私友好的合成/对抗评测集；用户私有音频可通过同名 audio/truth 目录跑 `benchmark.md`、`benchmark.json`、`review.md`。
- 人工复核队列：`review.md` 按 P0/P1/P2 排序，给出复核原因、建议动作、截断证据和源文件路径。
- 可选 LLM 仲裁：长音频模式可接本地 Ollama，只对不确定 chunk 做 evidence-only 仲裁，结果写入 audit / metrics，不改写 raw ASR 证据。

LLM 仲裁刻意默认关闭。这是资源和可信度边界：默认转写链路必须在没有 Ollama、没有额外 GPU 驻留、没有顶级模型猜测的情况下稳定工作；需要最终猜测时再显式打开。

长音频 planner 已按引擎能力限制实际切片，并以 schema 2 manifest 支持内容寻址的断点续跑。后续如升级为 VAD 静音边界切片，应继续保留请求值、有效值、provenance、内容 hash 和 resume 判定，不得退回无来源的固定 300 秒描述。

个人使用版关闭标准：

1. `doctor.ps1` 能确认无代理、CUDA、模型配置和依赖状态。
2. 单元测试全通过。
3. `smoke-asr-smart.ps1 -Json` 能完成默认 strict smart job；`smoke-evidence-asr.ps1 -Audio <path> -Json` 能对重要录音完成 FireRed + Qwen 证据验收。
4. 公开仓库只包含源码、脚本、配置、测试和文档，不包含模型权重、用户音频、输出转写或 wheelhouse 大文件。

本机已用一段超过 40 秒的真实中文电话录音完成 FireRed + Qwen 四切片验收：API、manifest 和四个 chunk 均为 `evidence_status=verified`，四个 FireRed raw 均为非空、无错误、初始装载为 BF16，逐段审计无 `engine_failure`；同一请求随后断点复用为 0 个处理、4 个跳过。默认 Qwen + SenseVoice strict smoke 另行通过，说明可选 FireRed 安装未改变默认组合。私人音频、转写正文和收据保留在 Git 忽略的本地归档，不进入公开仓库。

## 下载策略

常规模型下载会清空代理变量，模型默认走 ModelScope ID，并缓存到项目根目录下的 `models\modelscope`。FireRed runtime setup 同样清理代理环境，但 Hugging Face 权重下载保留调用进程当前代理设置。

运行时会优先检查缓存目录：如果 YAML 中的模型 ID 已在本地缓存中，就把它们作为本地路径传给 FunASR，减少日常转写时的网络探测。

Qwen3-ASR 权重必须先用 `scripts\download-models.ps1 -Engine qwen3-asr-1.7b` 预取。该脚本使用 ModelScope 的 `Qwen/Qwen3-ASR-1.7B`，本地目录是 `models\modelscope\Qwen\Qwen3-ASR-1.7B`。

FireRed 采用独立流程：

```powershell
.\scripts\setup-firered.ps1
.\scripts\download-models.ps1 -Engine fireredasr2-llm
```

源码、权重和 `MODEL_RECEIPT.json` 都位于 ignored `models/firered` 树，不进入仓库。receipt 记录必要文件的 SHA-256，供后续核对；不要把 receipt 当作模型文件本身，也不要绕过固定 revision。
