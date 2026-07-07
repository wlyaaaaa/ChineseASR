# 架构说明

## 目标

`ChineseASR` 面向中文严谨转写：先保证不幻觉、不乱编，再追求速度和格式化效果。

## 引擎策略

引擎不再硬编码在 Python 实现里，而是注册在 `configs/models.yaml`：

1. `defaults.engine` 决定 quick 模式默认模型。
2. `strict.primary_engine` / `strict.secondary_engine` 决定严格模式双模型组合。
3. `aliases` 记录 VAD、标点、说话人等可复用模型别名。
4. `engines.*.adapter` 决定运行时适配器；当前已实现 `funasr`。

当前默认策略仍然是：

1. `sensevoice`：默认主线。组合 `iic/SenseVoiceSmall`、`fsmn-vad`、`ct-punc`、`cam++`。
2. `paraformer`：中文保守对照线。适合做交叉识别和基准回归。
3. `whisper-large-v3`：只记录为 fallback/comparison，不自动作为主输出。

## 数据流

```text
audio -> VAD split -> ASR engine -> raw JSON -> Markdown transcript
```

严格模式：

```text
audio
  -> SenseVoice raw JSON
  -> Paraformer raw JSON
  -> normalized text comparison
  -> strict.md + strict.audit.md + strict.audit.json
```

`strict.md` 是给人读的最终稿，正文尽量干净；当两个模型严重冲突、都为空、或出现常见幻觉套话时才写入 `[疑似]` / `[听不清]`。`strict.audit.md` 保存两模型原文、相似度、备选文本和判断依据。后续接入顶级 LLM 仲裁时，应只读取这个审计输入，不覆盖原始 ASR 证据。

## 模型替换边界

同一 `funasr` 适配器内替换模型时，优先只改 `configs/models.yaml`，然后运行：

```powershell
.\scripts\download-models.ps1 -Engine <engine-name>
.\scripts\strict.ps1 -Audio E:\path\to\audio.wav
```

新增不同运行时，例如 Whisper 本地实现、LLM 音频模型或云 API 时，应新增 adapter，并保持 pipeline 只依赖统一的 `build_model(...)` / `generate(...)` 形状。

后续扩展时应增加：

- 音频 hash 和输入元数据。
- 顶级 LLM 仲裁，用于对模型分歧给最终猜测和置信说明。
- 疑似幻觉片段标记。
- CER/WER/CER-like 中文评测集。

## 下载策略

脚本层和 Python 层都会清空代理变量。模型默认走 ModelScope ID，并缓存到 `E:\ChineseASR\models\modelscope`。

运行时会优先检查缓存目录：如果 YAML 中的模型 ID 已在本地缓存中，就把它们作为本地路径传给 FunASR，减少日常转写时的网络探测。
