# 本轮实验：全部评估与分析报告

2026-10-20。先读 M1 总报告，再按下面的主题展开。所有图都嵌在 Markdown 中。

**既有实验备份已通过 B2 独立回读：主体 1,733 个对象，补充包 20 个对象，合计 1,753 个对象、25,120,671,034 字节。** 新增的 DEV64 full-context 原始权重参照及 CPU 分析已完成，增量 B2 归档仍待验收；它不在这两组已封存备份中。

## 一、已执行实验的评估结果

| 报告 | 包含内容 | 核心结论 |
|---|---|---|
| **[M1 总评估报告](report.md)** | 实验配置、100M 主结果、全部训练曲线、来源分解、α 对照、DEV64、数学检查、限制与恢复索引；8 张图 | writer 改善后续预测。新事件的主要收益在首个 5M 已出现，随后波动；100M 保持主检查点 |
| [历史留出与六个来源](report.md#historical-curve) | 0M—100M 的 21 点、65M/100M 比较、各来源区间 | 65→100M 的 self 预测继续改善，donor 差距下降不等于自身预测退步 |
| [新事件 DEV64 完整评估](report.md#dev64) | 21 个检查点 × 64 个事件；0/1/4/8/16 篇干扰文章；prefix、方向与完整性检查；4 张图 | 长干扰后的事件相关优势仍为正，但会衰减；后期训练的长距离增益尚不确定 |
| **[DEV64 原始权重完整上下文参照](context_reference.md)** | 同一目标下完整 A/D 上下文与 writer 记忆的 NLL 收益；100M 全文／首窗主表、配对区间、21 点附加表 | 100M 全文收益约为完整源文章上下文收益的 14%–16%，首窗约 22%–26%；不是准确率或事实召回率 |
| [固定预算 5M α 对照](report.md#alpha-ablation) | α=1、0.99、0.95、可学习全局 α、证据 α 的匹配比较与配对区间 | Δ_donor 单独变大不代表整体更好；后续按决定只比较原生固定 1/0.99 |
| [跨文章首块诊断](report.md#cross-article) | 还没写当前文章时，旧文章记忆的作用 | 历史留出上 65→100M 的首块预测改善；这项诊断不属于新 DEV64 |
| [P1 双种子固定开发集诊断](../../p1_diagnostics_results_20261019.md) | seed17/29；self、empty、两个 donor、完整上下文；语言、复现桶与写入深度 | 两个种子的 Δ_donor 均为正；完整真实上下文明显更强 |
| [P2 Base/I 匹配比较与首次新闻检查](../../p2_results_20261019.md) | 相同初始化、5M 预算与 tokenizer 的 B→I / I→I 对比；2 事件、4 道候选题的流程检查 | 本轮保留 I-trained writer；小新闻样例没有显示答案准确率增益 |

P1/P2 文档保留其发表时的状态描述，后续 M1/DEV64 的完成情况以本页与 M1 总报告为准。不同阶段的数据、任务与统计口径不同，不能把表中数值直接横向排名。

## 二、数学与架构分析

| 报告 | 解决的问题 | 结论边界 |
|---|---|---|
| [容量与 writer 数学检查](report.md#capacity-writer-audit) | 终点变化是否来自表写满、过拟合或饱和 | 表未写满；不能排除哈希干扰。现有证据不支持 65→100M 的典型验证损失反弹，也不保证以后每一步都持续受益 |
| [D1 更新算子的形式性质](../../formal_d1_properties_20261019.md) | 块内同一旧快照、平均更新、顺序与状态语义 | 数学性质适用范围见原文；它本身不替代经验评测 |
| [Qwen3.8 原生 writer 架构与复核](../../design/qwen38_native_20261020/architecture.md) | 架构图、160 维行/16 头、输出秩、完整 Jacobian、oracle 的解释边界 | 独立原型完成组件验证；全模型接入与原生训练收益尚未验收 |
| [原始 writer 评议](../../design/qwen38_native_20261020/references/delta_ngram_writer_review_20261020.md) | 用户提供的评议原文 | 保留原文供对照；其中需要更正的数学表述见上面的架构复核 |

## 三、数据、复算与恢复

- [机器可读归档索引](archive_manifest.json)：权重身份、数据对象、代码版本、复算边界。
- [下载与恢复说明](restore.md)：分开下载主体与补充包、固定 SHA 校验、原路径和环境恢复。
- [独立复算收据](reproduction_receipt.json)：分析包内表格与汇总重算结果。
- [本地代码检查](code_validation_20261020.json)：完整测试 **862 项通过**，M1 三个核心实现文件保持原字节。
- [主体 B2 回读收据](backup_receipts/core_restore_verification.json)：1,733 个对象、23,885,567,983 字节全部匹配。
- [补充 B2 回读收据](backup_receipts/supplement_restore_verification.json)：20 个对象、1,235,103,051 字节全部匹配；[封口](backup_receipts/supplement_COMPLETE.json)、[成员清单](backup_receipts/supplement_member_manifest.jsonl)与 [SHA256 清单](backup_receipts/supplement_sha256.txt)保留封存时原件。最终回读收据在封存后产生，保存于 Git，不属于本次 B2 补充包。
- [冻结登记与执行记录](registration/README.md)：100M/65M 身份、21 点追加登记、运行结束记录。
- [Context 参照主表](context_data/context_recovery_primary100m.csv)、[21 点附加表](context_data/context_recovery_curve.csv)、[汇总与校验](context_data/context_recovery_summary.json)：新增结果已完整，增量 B2 收据尚待补齐。
- [历史启动脚本归档](../../../scripts/archive/m1_20261020/README.md)：原样脚本与来源 SHA；不是新启动命令。

## 尚无结果的项目

**FINAL、完整 64 事件 QA/自由生成、原生 Qwen3.8 完整模型训练与 α=1/0.99 正式比较尚未执行。** 本目录不将设计文档、流程检查或 CPU 组件测试当作这些项目的效果报告。
