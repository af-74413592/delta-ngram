# DEV64 no_grad 修复与接续

## 实际状态

`main05` 的 α=1 已完成834组、19,981,748个模型目标，保存了首个20M节点以内的最后完整组；当时α=.99尚未训练。旧评测在main05和main06均完成4/64家族后卡住；用户观察到四卡显存从约68 GB升至设备上限、功耗下降并停止推进。main06已按用户指令停止，四卡释放后启动main07。

main07于 **2026-10-20T15:26:49.711816Z** 启动，PID **86316**，恢复待完成的α=1 DEV64。原修复提交记录36/64家族；随后评测已完成 **64/64家族、16轮、844次原生证据或查询调用**，耗时约452.22秒，生成COMPLETE。未改动的训练runner已自动切到α=.99第一个20M阶段，前3组分别累计9,216、34,816、61,440模型目标，检查点正常保存。15:38:56 UTC回执记录该支已完成152,223个目标。完整机器证据见[修复回执](qwen38_native_eval_no_grad_repair_20261020.json)。

## 修改

- 查询reader改为 `torch.no_grad()`；去掉Δ叶子的 `requires_grad_(True)` 和CE输出必须带梯度的要求。
- 保持原batch大小、补齐、空状态零sentinel、labels、窗口、证据写入和评分顺序。
- `memory_batches.jsonl`记录每次原生调用的四卡allocated、reserved和peak；`memory_rounds.jsonl`记录每轮显存与耗时。不调用backward、`empty_cache`或额外GC实验。
- 训练runner仍为SHA256 `eaf6fd988a77328fc658f426f0145ad291cb29e324b4294c3049118382cca44e`。修复后的eval SHA256为 `46a3cc47d20bd8f4dce103b7aee5654824c95da5375bac08e074f3997b3aead9`。

旧路径开启梯度却不反传，存在保留reader图的风险。已发现Megatron dispatcher会将带图的路由概率保存为模块属性，删除eval局部输出不足以保证图释放。此项源码发现及no_grad后的显存表现支持关闭评测梯度；尚未隔离出所有底层引用，更未证明无限增长一定来自某一种Python引用循环。

## 显存与数值核对

完整16轮的PyTorch峰值allocated为 **71.693 GiB/卡**，峰值reserved为 **72.730 GiB/卡**。各轮结束后的allocated范围为 **65.906–66.482 GiB/卡**，未重现旧路径的持续增长。这里区分PyTorch实际分配与缓存保留量，未把它们混同为nvidia-smi读数。

复用旧partial基线，对前4家族、5个距离、所有条件及方向置换每个seed逐项比较，共220个NLL项目。checkpoint writer SHA、alpha、数据计划、计数、来源、窗口及其他结构字段一致。

| 比较 | NLL精确相同 | 最大绝对差 | RMS差 |
|---|---:|---:|---:|
| 旧05 vs 旧06，均开梯度 | 28/220 | 0.01331581 | 0.00298439 |
| 旧05 vs no_grad07 | 24/220 | 0.01581447 | 0.00393828 |
| 旧06 vs no_grad07 | 22/220 | 0.01716423 | 0.00394071 |

**未达到逐项完全一致，不宣称数值等价。**旧两次运行自身也有差异；本次重跑重新计算H和Δ，未证明它们逐字节相同，因此这些比较无法单独隔离梯度模式的影响。partial只包含聚合分数，没有逐token logits，不能据此宣称logits逐位一致。旧books审计的grad/no_grad逐位一致仅覆盖当时单个样例。

## 检查点与预算

独立CPU工具 `qwen38_native_migrate_eval_only.py` 要求整个代码清单中只允许eval文件改变；训练文件、native runtime、数据计划和初始writer身份仍受原有检查。派生checkpoint只更新 `code_manifest_sha256`、`runtime_identity_sha256`，所有其他字段，包括两臂权重、AdamW、sampler、四rank RNG、稳定性历史、阶段游标及历史预算，均通过typed结构与tensor/NumPy字节摘要保留核对。

原始main06 checkpoint保持不变，派生文件另存，来源与允许差异写入migration receipt。迁移保留的pending attempt为2；原runner正常接续时递增到attempt3。没有重训α=1的前20M，也没有重新初始化α=.99。

恢复携带累计optimizer消费 **21,450,663**、诊断 **59,620**、失败DEV64消费 **286,682**。本次完整DEV64新增 **2,197,313** 次目标访问，累计评测消费 **2,483,995**；旧失败和重跑均不退款。诊断未增加，α=.99的后续更新继续计入100M共享训练消费上限。

## 验证

- 17项CPU迁移测试通过，包括真实checkpoint序列化、状态/RNG完整保留、拒绝其他代码变动、拒绝退款及拒绝身份不一致。
- 远端实际迁移通过，源checkpoint未变，训练状态完整保留。
- main07完成全部64家族，保存评测COMPLETE，自动切换α=.99并实际完成训练更新、保存joint checkpoint。回执保留首次36家族快照时间及完整评测后的补充时间。
