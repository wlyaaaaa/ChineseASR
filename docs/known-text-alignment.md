# 已知文案与音频对齐

用于视频等调用方已经拥有音频与审阅文案，只需要逐词时间位置的场景。它复用本项目对齐器、音频前处理、原有 GPU 租约与可终止 worker，不启动双模型识别、听写服务或新常驻程序，不重新生成音频。

```powershell
.\.venv\Scripts\python.exe -X utf8 -B -m zh_asr alignment-info
.\.venv\Scripts\python.exe -X utf8 -B -m zh_asr align C:\Input\voice.mp3 --text-file C:\Input\voice.txt --output C:\Output\aligned.json --timeout-sec 300
```

`alignment-info` 只读配置和路径，不加载模型；当前模型与版本仍由 configs/models.yaml 和模型锁管理，不在调用方复制模型清单。`align` 默认 cuda:0，需要普通登录用户环境及既有 Broker；CPU 显式选择 `--device cpu`，同样有子进程超时与回收。

输入是单个音频与 UTF-8 文案文件。先在临时目录转换 PCM，再核验模型、对齐结果的覆盖和时间范围，最后重新读取原音频/文案哈希，完整成功才原子替换输出。取消、超时、缺模型、输入变化和覆盖不完整都不覆盖上一份成功结果。调用方不应把输出路径设为任一原件。

结果 `zh_asr.alignment-entry.v1` 包含原音频 hash、文案 hash、逐词 `word/start/end`（秒）、实际模型身份和覆盖；`lexical_truth_verified=false` 始终保留。**强制对齐只找到所给文字在声音中的位置，不证明声音确实逐字正确**。重要内容需另行识别/听审。

单场时长不能超过模型配置的 max_audio_sec（当前 300 秒）；超长时明确报错，应按已审阅分镜切分，不按字符比例猜测音频位置。模型下载、升级与安装仍走现有 model_lifecycle 流程，本入口不会隐式下载。
