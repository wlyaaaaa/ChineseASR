# 文件转写质量与长期模型维护

本页描述项目已实现的调用契约，不是本机安装或准确率验收报告。以实际命令、退出码、模型版本、测试报告和原录音回听分别验收；不把一项通过替代其他项。

## 路线与调用

`quick` 仍是 SenseVoiceSmall 单模型初稿；桌面听写仍是单 Qwen3-ASR，不自动保存录音或最近正文，不增加双模型等待与 LLM 润色。

文件转写有两个命名配置，显式引擎参数优先于命名配置：

| 配置 | 主引擎 | 校验引擎 | 用途 |
| --- | --- | --- | --- |
| `baseline` | Qwen3-ASR-1.7B | SenseVoiceSmall | 当前默认的对照与回退路线 |
| `high_quality` | FireRedASR2-LLM | Qwen3-ASR-1.7B | 显式高质量候选路线；不因公开分数或安装成功自动提升为默认 |

```powershell
# 本地 API/job 入口，不上传音频
.\scripts\asr-smart.ps1 -Audio C:\audio\recording.wav -Mode strict -Profile high_quality -Json

# 命令行：超长输入自动转到可恢复的长音频流程
.\.venv\Scripts\python.exe -m zh_asr strict C:\audio\recording.wav --profile high_quality --out-dir .\outputs\example

# 已知双声道分录时，明确选择从 0 开始的声道；不推断说话人身份
.\scripts\asr-smart.ps1 -Audio C:\audio\call.wav -Mode strict -Profile high_quality -ChannelIndex 1 -Json

# 批量文件：长文件分块，短文件保留批处理
.\.venv\Scripts\python.exe -m zh_asr batch C:\audio\folder --profile high_quality --out-dir .\outputs\batch-example
```

配置中存在某个模型，不等于 adapter 已实现、权重已安装或真实推理通过。Whisper 当前仅为配置占位，不是可调用的兜底引擎。Paraformer 的匿名说话人聚类继续保留；强制对齐器不替代说话人分离。

## 从音频到可复核结果

文件流程按模型时长上限和质量参数切块。VAD 用于寻找较自然的切口，保留原时间线覆盖，不把未检测到语音的区间当成应删除的内容。FireRed 仍采用更短的建议切片，输入不能超过其硬限制；上下文重叠也计入实际长度。

Qwen 支持批量输入，批处理异常时隔离失败项，避免一个坏文件吞掉邻居结果。生成长度触及上限、异常短文本及边界风险作为完整性信号保留；这些信号不是经过校准的概率。

差异比较保留小数点、正负号和关键技术符号；否定、数值及明确配置的关键项不能被整段高相似度掩盖。常规 CER 与保留关键符号的 CER 分开统计。正常出现“字幕”“点赞”等词，不单独构成删除或改写理由。

Qwen3-ForcedAligner-0.6B 接收音频与已经提供的文字，输出字词时间范围。它负责定位，不负责证明这些文字正确。对齐不完整时保留警告；只在文字和重叠音频时间共同支持时移除重复覆盖，不能因为连续两次真实说了同样的话而去重。

`quality.review.json` 与 `quality.review.html` 提供分歧位置、上下文音频和补充识别。SenseVoice 只在它不是现有两路之一、且存在复核片段时补听，不作为第三票决定词汇真相。仍有分歧时保留候选，不改写原始审计结果。

HTML 只使用本地音频，不请求外部服务；笔记导出包含对应片段与审计内容的哈希，避免把修订误用于另一段录音。复核功能失败时，原正文与原始结果仍保留，质量状态明确降级。

## 状态不能混为一谈

API 的 `status` 表示任务执行，`evidence_status` 表示原始产物与引用的完整性，`quality_result` 表示补充质量复核。`quality_result.needs_review=null` 表示没有可用的质量结论，不等于无需复核。

两路模型一致、`evidence_status=verified`、`exact_text_coverage=true` 均不能单独证明逐字准确。对齐及复核输出保留 `lexical_truth_verified=false`；输出时间戳也不构成说话人身份确认。

单引擎失败可以留下临时可用正文和明确的失败证据，但不能作为完整 strict 缓存命中。长音频、批量和质量复核分别记录内容与运行身份；代码、模型锁、配置、输入或调用绑定变化时不能静默套用旧结果。

## 模型安装与升级

模型版本、运行时版本、adapter 和切片限制是一组兼容契约。增加同系列版本时更新该组件的固定 revision、文件校验和 runtime pin；新模型家族仍须实现 adapter 及其转换测试，不承诺只改模型名就能兼容任意架构。

```powershell
# 看配置、adapter 与本地模型状态，不执行转写
.\.venv\Scripts\python.exe -m zh_asr.model_lifecycle status

# 按已有不可变版本锁恢复缺失的对齐模型文件
.\.venv\Scripts\python.exe -m zh_asr.model_lifecycle fetch-aligner
.\.venv\Scripts\python.exe -m zh_asr.model_lifecycle verify-aligner

# 显式查询上游 revision；不改变默认、不自动下载、不新建后台任务
.\.venv\Scripts\python.exe -m zh_asr.model_lifecycle check-updates
```

存在锁文件但没有权重的新安装可以恢复。匹配的文件复用，下载文件先在暂存区校验，再进入模型目录。已安装文件与锁不匹配时停止，不用现场文件重新生成“正确”哈希。模型来源、固定版本和文件哈希不明时，不以目录存在判定可用。

候选升级先使用单独配置或显式引擎运行评测，保留旧权重与原配置。评测至少分开记录字错、关键内容错误、静音出字、错误自动放行、复核负担及耗时。公开短样例用于集成检查；真实质量判断需要适用场景的带参考文本音频。不能把同一录音的相邻切片当作独立的训练与最终验收样本。

```powershell
# 使用同一 corpus 与比较口径分别生成基线和候选结果
.\.venv\Scripts\python.exe -m zh_asr eval --corpus-dir .\eval\corpus\holdout --profile baseline --out-dir .\outputs\eval-baseline
.\.venv\Scripts\python.exe -m zh_asr eval --corpus-dir .\eval\corpus\holdout --profile high_quality --out-dir .\outputs\eval-candidate

.\.venv\Scripts\python.exe -m zh_asr.model_lifecycle compare --baseline .\outputs\eval-baseline\metrics.json --candidate .\outputs\eval-candidate\metrics.json

# 仅在比较通过、候选配置及代码身份仍匹配时明确提升文件 strict 默认
.\.venv\Scripts\python.exe -m zh_asr.model_lifecycle activate-profile --profile high_quality --baseline .\outputs\eval-baseline\metrics.json --candidate .\outputs\eval-candidate\metrics.json

# 使用激活命令返回的 switch.json 路径回退
.\.venv\Scripts\python.exe -m zh_asr.model_lifecycle rollback-profile --receipt .\outputs\model-maintenance\<switch-id>\switch.json
```

比较拒绝不同音频/参考文本、指标口径不一致、重复 case ID、无效数值、跳过案例、样本不足、纯合成样例或未改善的候选。通过仍只适用于所给样本，不是通用准确率保证。激活保存前后配置与收据，仅改 strict 的两个默认引擎；回退拒绝覆盖后来发生的默认引擎变更，同时保留无关配置改动。

权重或运行时升级的回退还需恢复对应的旧版本与旧 pin，不能用配置切换代替依赖恢复。应用启动后已经驻留内存的模型不会因为磁盘文件变化而自动变成新版；退出相关作业、按既有运行入口重新加载，并再次核对运行身份。

## 资源、云端和验收边界

ASR 的 GPU 租约默认有效期两分钟，运行中每二十秒续期。继承租约的子进程也不能把期限延长成六小时。丢失租约时终止对应任务，不抢占别的任务，也不绕过共享调度器；已有旧租约需要合法的调度器恢复，修改客户端不会追溯清除它。

云端仍只走既有显式授权入口。普通本地配置、质量分歧或安装新模型都不会自动上传。云端逐片保存成功结果；恢复时复用确认成功的片段，对已发出但结果不明的请求不盲目重复计费。任何取消均为终止，不能因为模型或密码验证被取消而自动重新申请。

本地回归入口为 `python -m unittest discover -s tests`。分别验收源码测试、依赖一致性、实际新模型推理、长音频/声道/断点恢复和正式云端调用；模拟 API 的测试不等于真实云服务已通过。模型权重、原录音、正文、运行日志和私人参考文本不进入公开 Git。

## Desktop dictation reliability

Win+H uses a persistent supervised model subprocess, not the file-quality pipeline. UI and microphone work stay outside native model calls. Audio travels through an anonymous pipe and stays in memory. No new cloud call, transcript history or listening port is introduced. Idle weights remain in RAM; an active GPU lease belongs to the exact model-process creation identity.

Loading is bounded to 180 seconds per attempt, activation/inference to 60 seconds, warmup to 40 seconds, and parking to 20 seconds. Startup gets one recovery attempt; an unreturned phrase gets one retry after confirmed process exit. A warmup timeout reloads without repeating that warmup. Repeated failure becomes actionable instead of an infinite preparation state. Clicking the microphone retries failed initialization without rebooting Windows. An access-denied process query is not proof that another task died.

File ASR and dictation renew short process-bound leases; the broker also caps legacy ASR lease requests to prevent old clients from retaining hours of orphan occupancy. A legitimate file job still has priority until it finishes: the dictation status names that blocker, preserves recorded audio in memory, and allows Esc cancellation. Do not bypass arbitration to hide contention.

Inspect `scripts/dictation.ps1 -Mode Status`, `outputs/dictation/runtime.log`, and `http://127.0.0.1:32100/_gpu_broker/status` separately. Technical logs contain operation timing/error categories, not recognized text. The initial microphone selection can fail independently of a ready model; reconnect the selected device rather than silently recording a different one.

For an approved upgrade, stop through the existing launcher, verify tests and pinned dependencies, then start the installed `ChineseASR Dictation` task to load the host without starting a recording. The launcher's Start/Restart action is the user-facing recording action. Do not send test text to an arbitrary focused application. Source deployment does not update a model already loaded in another process.

ASR and OCR may run together; same-family work is serialized and Ollama heavy work remains mutually exclusive. Helpers are hidden. Neither workflow should call Wallpaper Engine pause/stop, suspend its process, or add broad Python/WSL playback rules. Preserve unrelated fullscreen, maximized-application and user-authored rules. Tests for native timeout/crash/cancellation and orphan leases are in `tests/test_dictation_worker.py` and the PCConfig broker suite; these do not replace actual microphone/UI or wallpaper playback observation.

File-transcription workers adopt their supervisor lease through the live broker. The broker tracks the worker creation identity, not just the launcher. A process-handle watchdog cleans the job-tagged WSL descendants and exits the worker when its supervisor disappears; normal successful worker exit is not misreported as lease loss. This prevents reclaiming GPU ownership while a surviving model worker is still being cleaned up.
