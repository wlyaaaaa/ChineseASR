# 架构说明

## 目标

`ChineseASR` 面向中文严谨转写：先保证不幻觉、不乱编，再追求速度和格式化效果。

## 引擎策略

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

后续扩展时应增加：

- 音频 hash 和输入元数据。
- 顶级 LLM 仲裁，用于对模型分歧给最终猜测和置信说明。
- 疑似幻觉片段标记。
- CER/WER/CER-like 中文评测集。

## 下载策略

脚本层和 Python 层都会清空代理变量。模型默认走 ModelScope ID，并缓存到 `E:\ChineseASR\models\modelscope`。

运行时会优先检查缓存目录：如果 `iic/SenseVoiceSmall`、`speech_paraformer...`、`fsmn-vad`、`ct-punc` 或 `cam++` 已在本地缓存中，就把它们作为本地路径传给 FunASR，减少日常转写时的网络探测。
