# Qwen3.8 固定 RMSNorm 的初始有界训练登记

2026-10-20。**实现与本地复核已完成；本文尚无新准备流程通过或optimizer更新的回执。**最终路径选定为sorted router、HC eager及split PLE；RMSNorm eps=1e-6、无log(RMS)、ρ=.01保持。两α各8步作为初始有界训练，计入后续pilot和累计100M，不自动扩展。下一阶段在同一常驻模型内重算H/scale并完成训练前检查；仅全部通过后创建优化器。

仓库[JSON](qwen38_native_normshort_registration_20261020.json)保留数值合同，schema 为 `qwen38_native_rms_stability_registration_v2`；本次说明更新不改写其冻结字段。实际执行须另存绑定当前代码、父账本及准备合同的 `PREPARATION_FROZEN_CONDITIONAL` 候选，通过检查后生成 `FROZEN_EXECUTABLE`。先前四臂草案保存在[历史目录](archive/four_arm_draft_20261020/qwen38_native_normshort_registration_20261020.md)，旧协议没有执行优化；已失败或未定的检查及消费全部保留。

## 1. 固定模型、writer 与输入处理

模型为 `Qwen/Qwen3.8-Flash-Next@de4b8e4d43b917e7706784d8bb445c9af86a3540`，reader 和原生 BF16 T⁰ 冻结。H 取真实 final HC mixer 后 LM-head 输入，2560 维、base-only、detach 后以 FP32 计算：

`H_norm = H / sqrt(mean(H², dim=-1, keepdim=True) + 1e-6)`

逐 token、沿最后一维、无可训练参数，在 writer pooling/right-window 之前执行。reader 自身 norm 不变，log(RMS) 幅度支路本轮不加入。固定 RMSNorm 是用户在优化前的决定，不是试验胜出或梯度验收结果。

两个组为 `alpha100_rmsnorm` 与 `alpha099_rmsnorm`。Writer 16 头、行宽 160、隐层 256、右窗 4、D1；载入同一 seed-17 `writer_initial_state.pt` 文件，其字节 SHA 为 `aa54ccfcdc800d46a145e7dab9787918fbbc96c0a4f662328e7d88dba6d32f11`。该文件含 unit target_scale；最终runtime的scale重新验收后，两组加载相同新向量。

## 2. 真实观测与唯一 ρ

旧compiled路径的12个训练家族无标签观测已完成：18个history frames、8,403个新token，前五来源各1,024，news_event 3,283；两事件完整首篇为2,163/1,120 tokens，分别5/3块。H RMS=3.729683，median=3.757632；T⁰各头RMS约.008008–.008489，RMSNorm条件raw target RMS约.003768，β接近初始.05。这些H和scale当前仅作历史证据；历史none条件不训练。

唯一共同 **ρ=.01** 保持。所选split-precision运行路径固定后，在相同训练家族上重新生成base-only H、raw-target统计及scale，每头仍按 `s_h=.01×RMS(T⁰_h)/RMS(raw_target_h)`；长事件按真实token数贡献，不按来源或family再等权，不重复计TP副本。FP64公式后一次转FP32，两α共用，训练中不再校准。旧16-head数值约.021256–.022531已移入JSON的 `scale_calibration.historical_compiled_runtime_calibration`，active向量和H/scale回执字段清空；不能直接启动旧scale。

历史观测COMPLETE SHA：`0ff5568f3b4bc3db213cb19347a2e4807c375956293ff0cfd6bb2e2bfb1bf30f`；历史scale artifact SHA：`849eb3abc68592fc24c150b684254c0cead1aa61a9e4104844fa92384d1c32d9`。文件及原数值保留，不覆盖历史。ρ定义为初始target RMS匹配所观测T⁰的1%，不保证D1 state或reader loss扰动为1%；重算scale不授权第二ρ或标签驱动搜索。

## 3. 固定训练与 probe 家族

来自已封印 calibration24，每来源 4 个 train family，按 `selection_hash,group_id` 排序；来源顺序为 books/code/papers/web/news/news_event。每来源 rank0/1 用于训练及 scale 校准，rank2/3 为固定 train-side probe；rank0 单独用于初始扰动。共 12 训练、12 probe、6 初扰样本，不按新结果换样。

全部 24 个 selection hash、sample ID 已独立重算。完整 ID、group、正文/文章 hash、token hash、角色保存在 JSON `data.frozen_samples`，角色绑定 hash 为 `e9266b93767f47cd732b1d5b1bf31268b6ec4d85cfae62a3070d021e3b226842`。selection SHA `2290d5741531c6cf7ada6f4867be9abdf549b3b7fd56c05f230fe56917328a3e`，派生 manifest SHA `de184d29c53b80bd840b56d548f131550fae577a9999d7f3d4bb259f97f3ed1a`。

本地紧凑回执不包含全部 cache/indices。实际 runner 仍须核对 seal→manifest→payload 的 bytes/SHA，并写入所有 history/query input、position、valid/write/loss mask 的 frame hash。长度不足、split/identity 错误即停，不替换或静默丢弃。probe 不参与梯度、scale 或 ρ 选择。

## 4. 精确 frame 与因果顺序

所有区间均为左闭右开。

- **普通文档**：history `[0,512)`；query prefix `[384,512)` 加 targets `[512,640)`，输入长 256。已左移 labels 位于预测位置 `[127,255)`，其余 `-100`。至少 640 原生 tokens。
- **事件**：完整写第一篇，每块最多 512 新 tokens，包括末尾短块；后续块带同篇最多 128 prefix。首块自然开始；文章边界重置局部 prefix/hash 上下文，保留 event memory。之后只查询 `BOS=248044` 加第二篇前 128 tokens，真实输入 129，labels 位置 `[0,128)`。不截断第一篇、不写第二篇或 query。
- **Mask**：encoder valid 包含真实 prefix+new；write、H/T⁰ 统计、pool/right 只含 new。evidence adapter capture mask=write mask，但 native base embedding/hash 仍看到完整 input IDs；query adapter valid 包含真实 prefix。右 padding ID248044，仅补到 TP2 的偶数长度，不读、不写、不计 loss。
- **状态**：每 sample、step、probe 重新空 memory，同一样本 history 多块之间保持全部 writer 图；α 时钟只在非空新写块推进，prefix/pad/query 不推进。query只读，最后 history 块也写，因为随后仍有 query。
- **数值路径**：evidence H为base-only、labels=None，需按所选runtime重算。主评测登记为grad-enabled/no-backward，并与实际训练在backward前的前向输出比较。所有self/empty/probe条件使用同一所选路径；no-grad模式差异作为A调查继续，不能单独阻止训练。

本短测不替代正式 8/16-block 或完整事件 pilot。普通来源仅一个从空状态开始的写块，不能独立区分衰减效应；事件的完整多块历史保留衰减和 BPTT 链。

## 5. 八步优化与缩减预算

AdamW：lr=.0003，weight_decay=0，betas=(.9,.999)，eps=1e-8，amsgrad=false，无 scheduler；L2 clip=1、nonfinite即停。草案固定 foreach/fused/capturable/differentiable=false，runner 必须一致或在更新前登记变更。

每 α 独立新优化器；每步按来源顺序、每来源 rank0再rank1，顺序处理 12 个 B1 microbatch，每样本128 targets。每次 loss 按整个累积组分母 **1,536** 归一化；保持参数不变完成全部12个样本，再clip/step一次，共8步。每组同数据、同初始参数、同scale。

| 项目 | 计算 | 目标访问 |
|---|---:|---:|
| 两 α 优化 | 2×12×128×8 | 24,576 |
| probe self：step0/4/8 | 2×12×128×3 | 9,216 |
| probe empty一次 | 12×128 | 1,536 |
| 初扰 self | 2×6×128 | 1,536 |
| 初扰 empty一次 | 6×128 | 768 |
| **初始有界阶段** | | **37,632** |
| 新准备检查 | 8×128，详见§7 | **1,024** |
| **本次最大新增访问** | 37,632+1,024 | **38,656** |
| 已完成诊断 | 含旧norm FD 440，详见下文 | **9,444** |
| **全部完成时累计上限** | 9,444+1,024+37,632 | **48,100** |

初扰合计2,304，probe合计10,752。旧1,408-target全BF16 writer loss FD未执行，其预留已取消。此次父账本保守将全部9,444既有诊断同时计入75,000阶段上限和100,000全局诊断上限；全部完成后分别余26,900和51,900，余额不授权重试、扩步或新增候选。

已完成诊断 **9,444 = reader preflight33 + norm FD440 + TP结构11 + 原模式审计4,608 + HC eager books768 + 模块定位768 + sorted-router A/B审计2,816**；这些回执的optimizer更新均为0。48,100是完整计划的累计上限，尚未全部消费。PLE CPU参考不新增CE；24,576优化targets还计入原生writer累计100M。所有调用在dispatch前入账，失败或重复访问照记，重启不归零。

## 6. 初扰、稳定性观测与结果

初扰仍使用每来源rank0，共6家族、2,304目标，报告每α/来源及总体Δ_empty。历史 `abs(Δ_empty)<.001 nat/token` 不再作为新有界训练的硬门槛；采用下节最新稳定性停止规则。初扰小不证明量化方向保留，不自动触发第二ρ或用probe调整scale。

固定probe只在step0/4/8运行，每次按writer checkpoint重建全部history。主量为每家族 `gain=L_self(0)−L_self(8)`，分来源/总体报告两α的学习变化、Δ_empty、梯度/clip/nonfinite、β、target/Δ/T⁰尺度、参数变化与实际成本；α间配对差异只作描述，不选择norm胜者。

配对分层family bootstrap使用10,000次、seed104729、95%percentile区间；每来源两个probe family内重采样，所有α/时点共享抽样。只重算已有CPU结果，不增加模型评分。只有12个训练侧probe family，不能据此声称独立DEV64/FINAL效果。

## 7. 所选数值路径、必需检查与停止规则

A的来源已定位到原生router按grad模式改变top-k排序；固定 `sorted=True` 保留训练模式的顺序。books审计中，native/split两条PLE路径的初始writer与零Δ四种情况，grad/no-grad、grad-repeat的logits/loss全部逐位一致；两次实际backward的训练前向也与评估前向一致。此结论限于该固定样本与状态，不替代新准备流程的实际anchor比较。HC eager与compiled各自模式parity通过，但两者数值不相同；最终仍采用预先选定的HC eager，不按loss优劣选路径。

B使用native base BF16 TE key/value投影加独立FP32 Δ投影，norm/gate/conv以FP32执行，最后转BF16；所有条件共用此路径，`PLE_FUSED_KERNEL=0`。books实际writer观测中，投影相加后约99.99%的非零请求仍可见，cosine>.99999997；最终BF16转换仍有明显误差：有效/请求作用cosine约.36357、沿请求方向幅度比.9925、能量比7.45、relative-L2约2.543，约15.2%的非零请求坐标产生变化。沿方向幅度接近1不代表完整方向保留；不宣称量化问题全部解决或端到端loss光滑，也不新增100%转换保真门槛。

ρ保持.01，.1是已允许但未选择的fallback，不自动切换。上述审计复用历史H/scale，仅用于dispatch与量化观察；新准备流程必须在最终runtime下重新采集相同12个训练家族的base-only H，计算raw-target统计和共同scale，不能复用历史向量。新empty baseline也需重算并与native baseline量化比较。

优化前仍需完整writer固定真实cotangent adjoint、已通过的PLE FP64参考、一个预登记端到端方向检查及主评测/实际训练前向比较。旧SP结构证明仅沿不变的mapping源码和拓扑复用；现有anchor同时采集实际FP32 dense-Δ cotangent，以独立index-add核对稀疏lookup adjoint及两TP原始梯度逐位相等，不新增CE调用。reader边界使用materialized有效Δ，完整history写入与lazy衰减图保留。

新增1,024目标的精确构成为：native empty比较128、实际reader backward anchor128、匹配的grad-enabled/no-backward前向128、两个步长各正负一次共512、逐位baseline重复128。split empty复用初扰额度；固定cotangent writer检查、H/scale校准及局部量化统计不新增CE。端到端方向固定为seed20261020的Rademacher，覆盖有效materialized memory坐标、RMS=1，绝对步长为2⁻¹⁴与2⁻¹³；两条中心差分与解析方向导数须满足rtol=.20、atol=.001，差分互相一致到20%，|解析值|<.01判未定。全部前向保持相同grad-enabled/leaf边界并重放RNG，baseline须逐位一致；失败或未定即在STEP001前停止，不搜索新步长或容差。

[运行器](../../../scripts/native_qwen38/qwen38_native_paired_norm_probe.py)提供零CE、零optimizer的 `--validate-preparation`，随后 `--execute` 只加载一次模型。代码复核、44项相关CPU测试及materialized两块history VJP检查已通过；这些不代表新的GPU准备检查或训练完成。

逐α登记停止规则：

- 非有限loss、gradient或memory state，reader参数变化，任何预算越界：立即停。
- 每optimizer step的未裁剪writer梯度L2范数，大于此前5个完整step中位数的10倍，连续2步：停；前5步尚无足够历史，非有限检查始终有效。
- 每step所有有效写事件中β<.001或β>.999的比例>.01，连续2步：停。
- step4/8固定12家族probe的每臂总体平均NLL较step0恶化>.1 nat/token：停该臂并诊断。
- step8审阅平坦或改善的loss曲线后再决定是否扩展；不要求每batch单调下降，不自动进入2M pilot。

沿用两α×8×12×128=24,576优化目标，固定probe10,752、初扰2,304，阶段上限37,632；新增准备检查1,024，合计最多新增38,656。任何失败或停止保留已dispatch消费，不自动重启、改ρ或延长。

## 8. 当前证据与限制

原runtime的TP结构检查接受`replica`倍率，仅覆盖SP/lookup adjoint。PLE CPU参考回执SHA `f34c36c20e274c35df32f40f77f66305a2185533880b1db89d8c6cbb1355f2b6`已独立核验：12个stage与3个整链方向全部通过两个最小预登记FD步长。native BF16/FP64输出relative-L2=.0042922354、cosine=.999990889，embedding VJP relative-L2=.0069462134、cosine=.999975930。结构快照的合成Δ=`sin(…)×.01`不是实际writer；`reference_fd_passed`仅说明参考自洽，native局部kernel差异另列，不宣称已全部验收。

原六来源审计及单独HC eager检查的模式差异仍保留为历史证据；修复后的books审计回执为 `ab_repair_books_01_receipts/COMPLETE.json`，本地字节SHA已核验：`8d18a0c8f8fc9bf554aa955d1e3a9d8ab4f3089831ae37249683dd4c419d6c16`。其2,816目标、两次backward、零optimizer，只覆盖§7所述固定books样本，不能推广为全部数据上的梯度或训练验收。

split的两TP量化统计已记录，最终BF16转换的方向误差仍须随实际运行报告。历史H下split self比empty高约.04006 nat/token，只是该诊断样本的观测，不能作为新尺度训练的结论。当前已确认诊断消费9,444、optimizer0；新准备与有界训练的实际通过状态，以后续独立回执为准。完整记录见[启动登记§9](qwen38_native_launch_registration_20261020.md#9-基础设施验收检查点)。
