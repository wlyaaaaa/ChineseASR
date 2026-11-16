# 实况取数

`docs/live.contract.json` 是公开接入单，`docs/live.snapshot.json` 是按需采样的公开静态快照。
本人完整接入单和带文件名、正文链接的快照保存在私人配套库；本仓不存原音身份、私人路径或正文。

## 按需更新

在本项目运行以下入口；尖括号参数由本机来源登记提供，不提交到公开仓库：

```powershell
.\.venv\Scripts\python.exe -B -m zh_asr live-snapshot --data-root . --output docs/live.snapshot.json --owner-output '<私人配套的本人快照.json>' --sources-output '<本轮来源登记.json>'
```

只读已有结果元数据、模型目录元数据和历史模型调用计时，不打开原音、不运行识别、不联网。
代码规模只统计 Git 已跟踪的自有 Markdown 和 Python、PowerShell、VBS、Shell、GS 文件，排除模型、产物、缓存和第三方。
静态总量、本地及云端段数使用同次扫描；未变的数据保留原时刻。checked_at 单列本次成功核对时刻，云稿总数、时长和无日期旧稿据此判断夜间刷新是否超期；历史听写日期不变。模型占地按需更新，不设过期阈值。
PCConfig 在原每晚自检后运行此生产器；云稿成功落盘和显式模型维护完成也会刷新 `outputs/` 内三份小 JSON。`live` 仅表示读取已有实测，HTTP GET 不扫描。
合成测试结果与业务小输出分开，刷新失败只记录，不改变转写或夜检结论；子进程限时 120 秒。此入口不新增计划任务或常驻进程。
产物共用一个 `asr.transcripts` 目录别名，覆盖输出目录；云稿和本地稿使用正斜杠相对路径，新稿不需改来源登记。
自动刷新只写忽略的 `outputs/`，PCConfig 指向这些文件；私人配套仍保存完整接入单，不再跟踪频繁更新的本人快照。按需导出旧配套路径时仍保留原 manifest 同步能力。
本人本机产物写 open.source 和目录内相对 path，不写 href；共享后端提供“在电脑上打开”的入口，访客份不出本机打开动作。

## 库存口径

- 本地：排明确测试目录及已知合成原音；执行 completed、覆盖 complete、有正文且引用的 raw 文件仍在，才纳入已转好。
- 云端：succeeded 且同名 TXT 仍在并非空，排已知合成原音；同原音多份成功稿合并。缺旧 intent 的稿写“未记名”。
- 总段数是两路原音哈希并集，不相加。回执三组分类保留每组分母，不作去重录音数或准确率。
- 最近二十段按每原音有日期的最新完整结果排序；本地日期取回执 mtime，云端取 completed_utc；缺时长留空。没记完成日期的旧云稿仍计入库存，不伪造日期、不参与最近排序，另列其份数。
- 云稿时长对单份分块区间取并集，同原音多次结果取最大覆盖；不把缺口、重叠或重复调用相加。
- 历史听写只读 `model operation=transcribe seconds`，带最后样本日期，不混入旧内部推理计时。

## 识字与转写共用批量读

`zh_asr.retained_results.read_latest_complete_results(paths, normalize, *, source_ids=None)`
每个 JSON 只读一次。项目适配器 `normalize(path, payload)` 返回含 `source_id`、带时区 `at`、`complete` 的行或 `None`，自行核对其正文仍在。
返回 `{原件ID: 最新完整行}`；`source_ids` 可一次筛多份原件。更新的失败/不完整结果不会覆盖旧完整结果。
读取或解析失败直接报错，不把不完整扫描装成全部；该函数不判断识字或转写文字准确性。
