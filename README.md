# ChineseASR

`ChineseASR` 是一个本地优先的中文语音转文字项目，目标是把中文录音转成可审计、低幻觉、可复现的文本。它面向 Windows + CUDA 工作站，默认 quick 使用 `SenseVoiceSmall`，strict 使用 `Qwen3-ASR-1.7B + SenseVoiceSmall`。FunASR 官方 GPU flagship `Fun-ASR-Nano-2512` 已作为显式 `fun-asr-nano` profile 提供，但不会因为安装完成而改变 quick 默认；可选的 `FireRedASR2-LLM` 是证据级词汇主引擎，也不会自动取代默认 strict 组合。

这个项目优先解决三件事：

- **中文准确性优先**：strict 模式默认以 `Qwen3-ASR-1.7B` 为主引擎，`SenseVoiceSmall` 为对照锚点。
- **低幻觉和可复核**：双模型分歧、静音出字、模板废话、异常重复、繁体残留、超长无标点等都会进入 audit / metrics / review。
- **适合 AI Agent 调用**：`scripts\asr-smart.ps1` 通过本地 API 提交任务，快速返回 job 状态，避免长时间卡住命令行或上层 Agent。

它是本地优先而不是云转写服务：默认路径不会上传音频。只有调用独立的“重要录音专业云入口”并同时声明录音重要、授权本次云上传时，才会把本机切片发送给阿里云百炼；模型、输出、wheelhouse 和私人评测数据仍保留在本机。

## 当前状态

`personal-use v0.1` 已完成收尾，进入维护态。关闭标准是：

1. `scripts\doctor.ps1` 能确认无代理、CUDA、模型配置和依赖状态。
2. 单元测试全通过。
3. `scripts\smoke-asr-smart.ps1 -Json` 能完成默认 strict smart job；重要录音另用
   `scripts\smoke-evidence-asr.ps1 -Audio <path> -Json` 验收 FireRed + Qwen 完整证据链。
4. 公开仓库只包含源码、脚本、配置、测试和文档，不包含模型权重、用户音频、生成转写、模型收据或 wheelhouse 大文件。

后续真实录音 benchmark、模型组合微调、Ollama 仲裁启用、VAD 边界切片优化都属于使用阶段校准，不是当前版本的关闭阻塞项。

## 适合场景

- 微信语音、会议录音、口述笔记、中文播客、课程录音的本地转写。
- 对“幻觉低、能复查证据”要求高于“只要快”的中文 ASR。
- 需要让 Codex、脚本或其他本地 Agent 调用 ASR，又不希望命令行长时间阻塞。
- 需要保留 raw JSON、audit、metrics、manifest，方便以后追溯模型版本和输入 hash。

## 不适合场景

- 只想要一个极简 GUI。
- 不愿意本地安装模型权重和 Python 环境。
- 希望普通录音或整个文件夹自动上传云端转写；云入口只接受明确的重要录音。
- 需要英文、多语种或字幕生产工具链作为主目标。

## 默认模型策略

模型注册在 `configs\models.yaml`，实现和模型选择解耦。

| 用途 | 默认引擎 | 说明 |
| --- | --- | --- |
| `strict` 主引擎 | `qwen3-asr-1.7b` | 准确率优先的中文主转写线，基于 Qwen3-ASR 权重和 `qwen-asr` runtime |
| `strict` 对照引擎 | `sensevoice` | 快速中文声学锚点，用于发现分歧和疑似幻觉 |
| `quick` | `sensevoice` | 单模型快速转写 |
| 显式 GPU flagship | `fun-asr-nano` | `FunAudioLLM/Fun-ASR-Nano-2512`；需要 GPU，作为较重的 LLM-ASR 候选，不改变 quick 默认 |
| 可选证据级词汇主引擎 | `fireredasr2-llm` | 隔离在 WSL 中运行；仅在显式选择时作为 strict 主引擎 |
| 重要录音专业云入口 | `qwen-audio-3.0-asr-flash` | 仅由独立脚本显式调用；Key 经 Password Center SecretRef 注入，普通模式无法触发 |
| 显式时间线/匿名说话人 baseline | `paraformer` | 固定 `speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch@v2.0.4`，输出逐句 `sentence_info` 时间和 CAM++ 匿名聚类；已知两方通话的调用方可传 `--preset-spk-num 2`，省略时自动聚类；不改变 quick/strict 默认 |
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

# 重要录音的证据级组合；不会改变默认配置
.\scripts\asr-smart.ps1 -Audio C:\path\to\long.mp3 -Mode long-strict -PrimaryEngine fireredasr2-llm -SecondaryEngine qwen3-asr-1.7b -WaitSec 15 -Json

# 明确的重要/专业录音才允许使用最强云候选；两个开关缺一即在上传前阻断
.\scripts\asr-professional-cloud.ps1 -Audio C:\path\to\important.wav -Important -CloudUploadAuthorized -Json

# 快速单模型，只在明确接受较少审计时使用
.\scripts\asr-smart.ps1 -Audio C:\path\to\audio.wav -Mode quick -WaitSec 15 -Json

# 显式使用 FunASR GPU flagship；不会改变 quick 默认
.\scripts\asr-smart.ps1 -Audio C:\path\to\audio.wav -Mode quick -Engine fun-asr-nano -Device cuda:0 -WaitSec 15 -Json

# 批量转写文件夹
.\scripts\transcribe-folder.ps1 -InputDir C:\path\to\audio-folder
```

固定端到端 smoke：

```powershell
.\scripts\smoke-asr-smart.ps1 -Json
```

重要录音的完整证据链验收：

```powershell
.\scripts\smoke-evidence-asr.ps1 -Audio C:\path\to\important.mp3 -Json
```

## 重要录音专业云入口

`scripts\asr-professional-cloud.ps1` 是唯一的云上传入口，当前 worker 固定调用阿里云百炼
`qwen-audio-3.0-asr-flash` 同步接口。它与 `quick`、`strict`、`long-strict` 隔离，普通调用、
批量文件夹和仅因录音较长都不会触发云端。阿里云当前对非实时长文件/说话人分离推荐
`qwen-audio-3.0-asr-flash-filetrans`，但该接口要求公网可访问的文件 URL；本项目坚持本地音频边界，
因此尚未把它接入本地 worker。现有入口会在本机切片后调用同步模型。

入口在创建任务和读取音频前依次要求：

1. `-Important`：当前录音已被明确归类为重要或专业录音；
2. `-CloudUploadAuthorized`：调用方确认这次可以把音频切片发送给阿里云百炼；
3. Password Center 的受管目标 `qwen-audio3-asr-important-once` 完整性验证通过。

API Key 只由 Secret Broker 注入固定、哈希绑定的子进程环境，不进入命令行、请求文件、
转写结果或模型上下文。音频先在本机转为 16 kHz 单声道 WAV，再按最多 180 秒切片；
每段使用 HTTPS Base64 同步接口，结果保存到被 Git 忽略的 `outputs\cloud-jobs`。云调用失败会
明确返回失败原因，不会静默冒充本地结果。对于法律、投诉、雇佣等证据录音，云结果是能力优先的
专业候选，同时仍应运行 FireRed + Qwen 本地证据链并人工核听，不能把云转写本身当作证据认证。

## 安装与模型下载

先安装 CUDA 版 PyTorch 和核心依赖：

```powershell
.\scripts\install-torch-cu128-direct.ps1
.\scripts\setup-core.ps1
```

核心依赖文件 `requirements-core.txt` 固定 `funasr==1.4.2`。Fun-ASR-Nano 是面向 GPU 的较重模型，先完成 CUDA/PyTorch 与核心依赖安装，再按需下载；不需要 Nano 的机器无需额外安装模型。

下载 quick / secondary 默认需要的 SenseVoice：

```powershell
.\scripts\download-models.ps1 -Engine sensevoice
```

按需下载官方 GPU flagship `Fun-ASR-Nano-2512`：

```powershell
.\scripts\download-models.ps1 -Engine fun-asr-nano
```

该命令只准备显式 profile 的模型，不会把它设为默认 quick 引擎；profile 固定 ModelScope revision
`05201c46f1c38592b1567f857c0d56eab3d0d8ef` 并启用官方 `trust_remote_code` 加载路径。没有可用 CUDA GPU 时继续使用 `sensevoice`。

strict 主线需要 Qwen ASR runtime 和权重：

```powershell
.\scripts\setup-qwen.ps1
.\scripts\download-models.ps1 -Engine qwen3-asr-1.7b
```

Qwen runtime 固定为 `qwen-asr==0.0.6`，模型固定为 revision
`a04930dbe5419bfee073f7cade734f572689a3a8`。下载脚本会生成并验证
`MODEL_RECEIPT.json`，逐项绑定 13 个必要文件的规范路径、大小和 SHA-256；已有固定
缓存可用 `-ReceiptOnly` 只生成/核验收据，不下载也不加载模型。runtime 版本、revision、
收据或任一权重文件漂移时，adapter 会在模型 loader 运行前 fail-closed。

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
| `*.strict.review.json` | 面向机器和人工复核的结构化投影 |
| `*.strict.receipt.json` | 绑定本次全部严格产物的路径、大小、SHA-256、语义声明与 bundle SHA |
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

## 按需说话人归属

默认转写不是声纹识别：模型给出的 `speaker` 编号只是匿名分段/聚类，不能直接等同“本人”。只有真实问题需要判断一句话是谁说的时，才对**已有的、带起止时间的转写 JSON**做一次小投影。普通 `attribute-speakers` 不读原始音频、不加载模型；结果只下发逐句文本、匿名说话人、状态、角色、单句中文依据和回到原转写的 JSON pointer：

```powershell
python -m zh_asr attribute-speakers C:\private\call.raw.json `
  --context C:\private\call.speaker-context.json `
  --out C:\private\call.speaker-attribution.json
```

### 私有 `person:self` 声纹锚

在本机私有边界内，可以显式保存**唯一一个**用户本人的 `person:self` CAM++ profile。它不是通用声纹库：没有第二数据库、服务、队列、他人 profile 或全库重跑。profile 保存在 Git 忽略的 `outputs\private`（也可显式指定另一条私有路径），包含原始参考音频的路径/哈希/片段/声道、生成时间、CAM++ 配置 revision、runtime 和模型文件哈希，以及可重建、可替换、可删除的本人向量；不会复制原始参考音频。

只对“本人候选”有清晰、无反证依据的有限片段建锚。每次必须写出一条 `--inference-basis`：profile 固定为可撤销的 `inferred`，不能冒充 `confirmed`，可以用更强或更新的参考显式替换。

```powershell
python -m zh_asr speaker-enroll C:\private\known-self.wav `
  --start-ms 0 --end-ms 12000 --channel mix `
  --inference-basis "该片段有可回查的本人候选依据，且可由更强参考替换。" `
  --profile C:\private\person-self.voice-profile.json --device cpu
```

之后只在需要的一个目标片段上生成证据。目标向量只在内存中计算并丢弃；输出仅留下原始目标哈希、精确片段时间、声道提取方式、模型/文件哈希、profile 哈希和相似度/阈值：

```powershell
python -m zh_asr speaker-evidence C:\private\call.m4a `
  --start-ms 18400 --end-ms 23100 --channel mix `
  --profile C:\private\person-self.voice-profile.json `
  --out C:\private\call.18400-23100.person-self-evidence.json --device cpu
```

显式替换或删除也只作用于这一份 `person:self` profile：

```powershell
python -m zh_asr speaker-enroll C:\private\new-known-self.wav `
  --start-ms 0 --end-ms 12000 `
  --inference-basis "新的可回查本人候选依据；替换旧锚。" `
  --profile C:\private\person-self.voice-profile.json --replace
python -m zh_asr speaker-profile-delete `
  --profile C:\private\person-self.voice-profile.json --confirm-delete person:self
```

`--channel mix` 会明确记录为 `mixed_not_channel_evidence`。`left`/`right` 只在**原始输入确为双声道**时由新入口精确提取；默认 quick/strict/long 的单声道准备产物不能倒推为左右声道证据。即使是小米录音，也只有同时满足已验证 cohort、原始右声道精确提取、源文件 SHA-256 和分段时间都匹配时，才有一个可撤销的“本人候选”声道线索。

把一份或多份上述证据传给投影时，`context.recording_audio.sha256` 必须绑定同一原始音频。CLI 会把转写 JSON、context、每份 voice evidence 的实际文件 SHA-256，以及该原始音频 SHA-256 写进顶层 `input_binding`；库调用没有文件时使用同一 JSON 的 canonical SHA-256：

```json
{
  "schema": "chinese-asr.speaker-attribution-context.v2",
  "recording_kind": "mono_call",
  "recording_audio": {
    "sha256": "<原始音频 SHA-256>"
  },
  "segment_evidence": [
    {
      "index": 0,
      "dialogue_role": {
        "candidate_role": "self",
        "reason": "该句在快递员询问后回答了本人持有物的故障。"
      }
    }
  ]
}
```

```powershell
python -m zh_asr attribute-speakers C:\private\call.raw.json `
  --context C:\private\call.speaker-context.json `
  --voice-evidence C:\private\call.18400-23100.person-self-evidence.json `
  --out C:\private\call.speaker-attribution.json
```

`contact_role`、`dialogue_role`、`semantic_role`、`cross_recording_role` 与来源上下文都是软线索，可单独形成可撤销 `inferred`，也会相互融合。当前项目没有独立可信 receipt adapter，因此即使 `source_identity` 带有 `authority_ref` 也只能形成 `inferred`，不能由 caller 自报升为 `confirmed`。输出逐段只保留匿名 `speaker`、结论、单句中文依据和原转写 JSON pointer；不会下发声纹分数、逐项内部线索或目标 embedding。

- CAM++ 相似度、联系人、声道、对话角色、句义和跨录音相同声音都不能单独 `confirmed`；相似度本身只是 `person:self` 的正/负候选线索。
- `attribute-speakers` 只让与当前唯一私有 `person:self` profile 哈希一致的声纹分数参与方向判断；profile 被删除或替换后，旧 evidence 自动失效并保留为 `unknown` 声纹线索，独立的声道、联系人、对话角色和句义依据不受影响。
- 归因器不做固定加权、合成分数或 high/medium/low 等级。单一清晰且无反证的线索、或多项同向线索，都可以直接产生可撤销 `inferred`；声纹/声道只组织注意力，不是低智力终裁。
- 若声纹或声道与有具体理由的来源、联系人、对话或句义判断相反，投影会在内部保留两边证据，并在对外单句依据中说明为何后者暂时压过前者；只有上下文判断本身冲突、或只剩无法解释的相反声学线索时，才输出 `unknown` 和 `speaker_attribution_gap=true`。
- `recording_kind=mono_call` 且目标来自 `mix`/`mixed_not_channel_evidence` 时，投影使用更保守的 `±0.04` 声纹风险带；模型阈值仍固定为 `0.31`，不会拿单通电话重调模型。风险带内的分数只记为 `unknown` 声学线索，仍可由具体联系人、对话角色或句义依据作可撤销判断。其他录音类型继续使用模型证据的 `±0.02` 常规歧义带。
- 这只归属“这段语音可能是谁说的”，**不证明**照片、视频、微信媒体或消息由用户发送、拥有或持有；媒体来源/Owner 必须另有明确来源事实。
- Paraformer 的可选 CAM++ diarization 仍只是匿名聚类，可能过拆/合并，不能替代 `person:self` enrollment，也不能把 cluster ID 解释成用户。

如果原转写没有可用时间戳，应只在真实问题需要时先补这一份录音的时间戳分句；不要批量把历史录音或每条消息转成“本人事件”。

顶层 job 的 `succeeded` 只表示流程产出了结果；机器消费者必须同时读取 `evidence_status`：

- `verified`：要求的双引擎链完整执行，且 final、audit、review、两路 raw 已通过收据哈希和语义交叉校验；不表示转写逐字准确，仍需看 `status`、分歧和人工复核项；
- `provisional`：至少一路引擎失败，现有文本只是带失败证据的回退结果；
- `unavailable`：证据链或必要产物不完整；
- `pending`：仍在处理；
- `not_applicable`：quick 单引擎任务不适用证据级双引擎状态。

显式使用 FireRed 时，若 FireRed 失败而对照引擎成功，流程仍会生成带 `[疑似]` 的回退文本，但 job、长音频 manifest 和对应 chunk 都会标为 `provisional`，并列出 `evidence_failures`。状态判定会重新验证收据覆盖的六项内容产物，核对路径、字节数、SHA-256、两路 raw 独立性、引擎身份、文本、执行状态、错误以及 final/audit/review 对 audit JSON 的投影；未同步重建收据的替换、损坏、缺失或语义错配会使 `verified` 降为 `unavailable`。该收据是自包含的一致性清单，不是数字签名或可信时间戳，不能单独证明外部真实性。证据级验收还必须确认每个 FireRed raw JSON 文本非空、`error=null`、运行时 dtype 正确，并且逐段审计不含 `engine_failure`；原始录音始终是权威来源，`verified` 不能替代人工核听。

### 空转写的客观结果

每个 quick/strict 结果都会旁写一个版本化的 `*.objective-result.json`（long-strict 为根目录的 `objective-result.json`，chunk 也各有一个）。它与旧 strict 两条 `engine_evidence` 和 receipt 分开，机器消费者必须读取 `objective_outcome`，不能把空字符串、空数组、零字节或 `expect_empty` 当作无语音：

- `speech_transcribed`：至少有可观察转写文本；单引擎空文本会附带 `quality_status=low_confidence`，不能把双引擎链的 `verified` 当成准确性证明；
- `no_speech_detected`：只在完整音频覆盖下取得规范 VAD 零区间，或完整有效 PCM 的全零负证据时使用，并绑定 raw SHA-256、处理器/配置/策略/request hash、区间和非空负证据 hash；
- `speech_detected_but_not_transcribable`：检测到语音区间但文本为空，保持 deferred/unknown，可交给音频理解路线；
- `indeterminate`：没有完整 VAD/负证据，或存在预处理、模型、子进程、格式、覆盖等不确定性。

sidecar 的正式正交字段是 `execution.status ∈ {completed, failed, unsupported, corrupt}`、`coverage.status ∈ {complete, partial, unknown}` 和 `quality.status ∈ {sufficient, low_confidence, unknown}`；旧 `*_status` 名称只在 `compatibility` 中保留。`media_kind` 固定为 `audio`。调用方若有 caller-owned 绑定，可通过 API `caller_binding` 原样透传；服务到 CLI 子进程只经 `ZH_ASR_CALLER_BINDING_JSON` 环境变量传递，不进入命令行，ChineseASR 不解释或伪造这些治理字段。long-strict 允许配置的 chunk overlap，但必须从 0 连续覆盖到原音频时长、没有 gap/exclusion、首尾闭合，且所有 child 负证据、区间、raw 引用和幂等身份都绑定时才允许聚合为 `no_speech_detected`。旧 Markdown 或缺 sidecar 的缓存最多是未验证的历史产物，不得据此宣称无语音。

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

为了避免和受管 Ollama、LocalOCR 或其他 ChineseASR 任务抢 GPU，所有公开 CLI 和
smart/API 路径都必须先取得 LocalGpuBroker 租约；Broker 不可用时任务失败关闭，不会
退回到仅凭 `nvidia-smi` 判断后继续重型推理。

默认端口是 `18666`，刻意避开 LocalOCR 的 `18665`。

RTX 5090D 32GB 显存较大时，这个 GPU 排他锁仍然是保守调度边界，不代表硬件不能并发。
旧客户端的 `-AllowGpuConflicts` / `allow_gpu_conflicts=true` 仍可被解析，但只影响没有
机器级 Broker 的嵌入式外部 CUDA 进程检测，不能绕过本机 LocalGpuBroker。正式 Broker
只协调已接入它的 Ollama、LocalOCR 与 ChineseASR，不声称管理 LM Studio 等未接入进程。

默认入口会向 `http://127.0.0.1:32100/_gpu_broker/*` 申请全机 GPU 租约。Broker
会在 ASR 启动前卸载空闲 Ollama/LocalOCR，并在 ASR 运行期间阻止新的 Ollama 或 OCR
重型推理。服务子进程启动时必须携带 opaque lease token 并向 Broker 验证当前 live
owner，父进程随后持续续租；裸环境标记不能证明已持有租约。直接 CLI 同样采用“持租约
监督进程 → 可终止工作子进程”结构。
租约续期一旦失败，运行中的完整子进程树会被立即终止；服务任务以
`gpu_broker_lost` 失败，不能在失去排他性的情况下继续生成貌似成功的证据。

## 长音频与断点续跑

长音频入口：

```powershell
.\scripts\asr-smart.ps1 -Audio C:\path\to\long.mp3 -Mode long-strict -WaitSec 15 -Json
```

`ChunkSec` 是请求值，不是无条件采用的固定值。实际 `effective_chunk_sec` 会取请求值与两路引擎能力上限中的最小值；选择 FireRed 时为 35 秒，并始终低于其 40 秒单输入硬上限。每个 chunk 都跑 strict 双引擎。

MP3 等非标准输入先统一为 16 kHz、16-bit、mono PCM WAV。schema 2 的 `manifest.json` 记录源文件与派生文件 SHA-256、转换 provenance、模型配置 hash、固定模型/runtime 收据、运行代码身份、请求/有效切片参数、已解析引擎和 chunk 状态。内容与运行身份共同形成 fingerprint。续跑时不会只相信 manifest 的缓存状态，而会重新验证每个 chunk 的收据和全部内容；一致时才跳过，残留 `running`、内容篡改或身份变化都会转为重跑。

待处理 chunk 按两路引擎最小 `max_request_inputs` 分成有界批。FireRed 配置为每批最多 16 个输入；每个批次中，每个引擎只加载一次，再处理该批所有 chunk，避免逐片重复加载模型。

strict audit v2 保留两路引擎的原始文本、raw JSON 引用、provenance、分歧和人工复核项。选择策略固定为保留主引擎证据：不做多数投票，也不做语义改写。
新生成的 strict bundle 使用包内相对引用，并由 receipt 绑定六项内容产物；完整目录复制
到归档盘后仍可复验。旧版绝对引用在原位置继续兼容，但不会伪装成可搬迁包。

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

LocalGpuBroker 不可用、已有其他受管重型任务，或无法取得全机租约。等待现有任务结束并
检查 Broker 状态；`-AllowGpuConflicts` 不会绕过机器级 Broker。

**为什么正文里有 `[疑似]` 或 `[听不清]`？**

这表示模型证据不足、两路分歧大、某一路失败或命中风险规则。正文不强装确定，细节看 `strict.audit.md` 和 `strict.audit.json`。

**是否会自动使用 LLM 猜最终答案？**

不会。Ollama 仲裁默认关闭，并且只读取 audit 证据。需要时在 `configs\models.yaml` 显式打开。

**换更强模型要改代码吗？**

通常不用。优先改 `configs\models.yaml`，只有新增运行时时才写 adapter。

## 更多文档

- [架构说明](docs/architecture.md)
- [公开发布边界](docs/public-release.md)
