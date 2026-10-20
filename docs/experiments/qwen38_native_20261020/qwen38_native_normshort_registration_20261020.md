# Qwen3.8 固定 RMSNorm 主训练决策与准备记录

## 最新状态：main07 已启动，首个节点评测待完成

总B12的四阶段H12/Q4调度已在 `main05` 完成α=1的 **834组、19,981,748个模型目标**，到达20M以内的最后完整组；α=.99仍为0。`main05` 和 `main06` 的旧grad-enabled DEV64均在4/64家族后卡住并停止，当前没有完整DEV64收益结论。

本次只修复评测为 `torch.no_grad` 并增加batch/round显存日志；eval SHA256为 `46a3cc47d20bd8f4dce103b7aee5654824c95da5375bac08e074f3997b3aead9`，训练runner仍为 `eaf6fd988a77328fc658f426f0145ad291cb29e324b4294c3049118382cca44e`。CPU迁移仅接受这一项eval code hash变化，派生checkpoint只改code/runtime两个顶层hash；训练状态、`phase=evaluation_pending, attempt=2` 游标和预算保持，源main06完整hash与序列化往返核对已通过。

`main07` 于 **2026-10-20T15:26:49.711816Z** 启动，PID **86316**；提交快照已完成36/64家族，峰值allocated为71.693 GiB/卡，9轮结束后实占66.286–66.482 GiB/卡。评测完成后自动切换α=.99的第一个20M阶段；完整评测尚未结束。启动时累计训练消费 **21,450,663**、诊断 **59,620**、此前失败评测 **286,682**；100M共享训练上限不变，失败评测另记且不退款。详见[no_grad修复记录](qwen38_native_eval_no_grad_repair_20261020.md)、[机器回执](qwen38_native_eval_no_grad_repair_20261020.json)、[最新运行记录](qwen38_native_launch_registration_20261020.md)及[四阶段核对](qwen38_native_staged_schedule_20261020.md)。

两支最初均从同seed17原始writer、同固定scale独立初始化。main04仅加载后撤下、零新增消费；main05早期前3组的61,440目标是历史启动快照，已被834组进度替代。既有数值门槛失败保留，不宣称等价或噪声无害。

## 此前准备和接续历史

以下阶段状态、预算及待办均为当时记录；当前运行以上述main07状态及修复回执为准。

2026-10-20。**当时用户指令：“不用检查了，开跑把，已经很久了”。据此取消准备04及剩余准备检查，直接进入主训练；独立两α八步小样本短测也已取消。**最终路径保持sorted router、HC eager及split PLE，RMSNorm eps=1e-6、无log(RMS)、ρ=.01。完整writer固定cotangent FD、端到端方向检查尚未完成，不能记作通过；主训练依据这次明确授权推进，并保留数值、reader冻结与预算监控。

**此前批处理接续：**用户要求修复batch=1和PP2单微批吞吐。运行02已在第12组完成时停止，运行 native_main_training_03 当时启动（PID40692），恢复263,433个已训练target的writer、AdamW、sampler、四rank RNG及稳定性状态。新版每波将独立轨迹收拢为B4微批，12条轨迹最多提供3个PP微批；逐轨迹因果读写、完整历史梯度和整组实际target分母保持。本次切换没有丢弃在途target；切换时累计writer预算318,217，诊断14,564。18:46北京时间已完成第13–17组；五组平均536.70 target/s，去掉含首次编译的第一组后为629.60 target/s，旧版12组平均266.92 target/s。数据组不同，属于早期吞吐观察。用户随后要求只保留批处理，其余提速提议不实施，详见[启动登记](qwen38_native_launch_registration_20261020.md)和独立吞吐回执。

**此前重启决策：**用户要求“重启训练吧，这两个太严格了”“可以重训”。首轮（PID32914）完成2次更新、34,816目标，实际dispatch计费54,784；19,968个停止时在途目标保留消费。运行02于10:19 UTC启动（PID36991），从seed17重训并取消梯度突增和β饱和的硬停止。该监控设置及每组latest checkpoint在运行03继续保留，其他非有限数值、reader冻结和预算保护不变。

首轮历史入口SHA为 `ec5361e8187937798cfd70d269f55a7fe0a5bfbe3729ef6e0e16e9467d4f03b8`，注册SHA为 `360236895ee7b77418f9ad463aff08ee06c04fe8a628ffb46cc729be4bc47034`；它们不代表重启配置。新代码、注册、进程和实际更新另以新运行回执记录。重启不退还旧消费，也不从第2步checkpoint继续。

仓库[JSON](qwen38_native_normshort_registration_20261020.json)及下文准备合同保留历史，不是新的启动门槛；本次说明更新不修改该JSON。主训练由[独立入口](../../../scripts/native_qwen38/qwen38_native_train.py)及其实际数据、代码、预算和用户授权注册管理，不等待 `PREPARATION_COMPLETE`。旧37,632阶段、1,024检查及48,100累计计划已逐字归档于[取消阶段快照](archive/bounded_eight_step_canceled_20261020/qwen38_native_normshort_registration_20261020.md)，[manifest](archive/bounded_eight_step_canceled_20261020/archive_manifest.json)保存原始bytes/SHA；更早的[四臂草案](archive/four_arm_draft_20261020/qwen38_native_normshort_registration_20261020.md)也保留。旧八步阶段未执行优化。

## 1. 固定模型、writer 与输入处理

模型为 `Qwen/Qwen3.8-Flash-Next@de4b8e4d43b917e7706784d8bb445c9af86a3540`，reader 和原生 BF16 T⁰ 冻结。H 取真实 final HC mixer 后 LM-head 输入，2560 维、base-only、detach 后以 FP32 计算：

`H_norm = H / sqrt(mean(H², dim=-1, keepdim=True) + 1e-6)`

逐 token、沿最后一维、无可训练参数，在 writer pooling/right-window 之前执行。reader 自身 norm 不变，log(RMS) 幅度支路本轮不加入。固定 RMSNorm 是用户在优化前的决定，不是试验胜出或梯度验收结果。

两个组为 `alpha100_rmsnorm` 与 `alpha099_rmsnorm`。Writer 16 头、行宽 160、隐层 256、右窗 4、D1；载入同一 seed-17 `writer_initial_state.pt` 文件，其字节 SHA 为 `aa54ccfcdc800d46a145e7dab9787918fbbc96c0a4f662328e7d88dba6d32f11`。该文件含 unit target_scale；最终runtime的scale重新验收后，两组加载相同新向量。

## 2. 真实观测与唯一 ρ

旧compiled路径的12个训练家族无标签观测已完成：18个history frames、8,403个新token，前五来源各1,024，news_event 3,283；两事件完整首篇为2,163/1,120 tokens，分别5/3块。H RMS=3.729683，median=3.757632；T⁰各头RMS约.008008–.008489，RMSNorm条件raw target RMS约.003768，β接近初始.05。这些H和scale当前仅作历史证据；历史none条件不训练。

唯一共同 **ρ=.01** 保持。所选split-precision运行路径固定后，在相同训练家族上重新生成base-only H、raw-target统计及scale，每头仍按 `s_h=.01×RMS(T⁰_h)/RMS(raw_target_h)`；长事件按真实token数贡献，不按来源或family再等权，不重复计TP副本。FP64公式后一次转FP32，两α共用，训练中不再校准。旧16-head数值约.021256–.022531已移入JSON的 `scale_calibration.historical_compiled_runtime_calibration`，active向量和H/scale回执字段清空；不能直接启动旧scale。

历史观测COMPLETE SHA：`0ff5568f3b4bc3db213cb19347a2e4807c375956293ff0cfd6bb2e2bfb1bf30f`；历史scale artifact SHA：`849eb3abc68592fc24c150b684254c0cead1aa61a9e4104844fa92384d1c32d9`。文件及原数值保留，不覆盖历史。ρ定义为初始target RMS匹配所观测T⁰的1%，不保证D1 state或reader loss扰动为1%；重算scale不授权第二ρ或标签驱动搜索。

## 3. 准备阶段的固定家族

来自已封印calibration24，每来源4个train family，按 `selection_hash,group_id` 排序；来源顺序为books/code/papers/web/news/news_event。准备阶段只使用每来源rank0/1共12家族采集H及校准scale，rank0共6家族做初扰；保留rank2/3的既有角色身份，但不运行其旧短测probe。不按新结果换样，也不将这24家族小集合当作主训练数据池。

全部 24 个 selection hash、sample ID 已独立重算。完整 ID、group、正文/文章 hash、token hash、角色保存在 JSON `data.frozen_samples`，角色绑定 hash 为 `e9266b93767f47cd732b1d5b1bf31268b6ec4d85cfae62a3070d021e3b226842`。selection SHA `2290d5741531c6cf7ada6f4867be9abdf549b3b7fd56c05f230fe56917328a3e`，派生 manifest SHA `de184d29c53b80bd840b56d548f131550fae577a9999d7f3d4bb259f97f3ed1a`。

本地紧凑回执不包含全部 cache/indices。实际 runner 仍须核对 seal→manifest→payload 的 bytes/SHA，并写入所有 history/query input、position、valid/write/loss mask 的 frame hash。长度不足、split/identity 错误即停，不替换或静默丢弃。probe 不参与梯度、scale 或 ρ 选择。

## 4. 精确 frame 与因果顺序

所有区间均为左闭右开。

- **普通文档**：history `[0,512)`；query prefix `[384,512)` 加 targets `[512,640)`，输入长 256。已左移 labels 位于预测位置 `[127,255)`，其余 `-100`。至少 640 原生 tokens。
- **事件**：完整写第一篇，每块最多 512 新 tokens，包括末尾短块；后续块带同篇最多 128 prefix。首块自然开始；文章边界重置局部 prefix/hash 上下文，保留 event memory。之后只查询 `BOS=248044` 加第二篇前 128 tokens，真实输入 129，labels 位置 `[0,128)`。不截断第一篇、不写第二篇或 query。
- **Mask**：encoder valid 包含真实 prefix+new；write、H/T⁰ 统计、pool/right 只含 new。evidence adapter capture mask=write mask，但 native base embedding/hash 仍看到完整 input IDs；query adapter valid 包含真实 prefix。右 padding ID248044，仅补到 TP2 的偶数长度，不读、不写、不计 loss。
- **状态**：每 sample、step、probe 重新空 memory，同一样本 history 多块之间保持全部 writer 图；α 时钟只在非空新写块推进，prefix/pad/query 不推进。query只读，最后 history 块也写，因为随后仍有 query。
- **数值路径**：evidence H为base-only、labels=None，需按所选runtime重算。主评测登记为grad-enabled/no-backward，并与实际训练在backward前的前向输出比较。所有self/empty/probe条件使用同一所选路径；no-grad模式差异作为A调查继续，不能单独阻止训练。

这些frame只用于准备和梯度验收。普通来源仅一个从空状态开始的写块，不能独立区分衰减效应；事件完整多块历史保留衰减和BPTT链。主训练的8/16-block及完整事件采样由独立主训练注册规定。

## 5. 已结束准备尝试与实际预算

以下是已取消继续执行的准备计划，供审计追溯：`--prepare-only`原拟重算H/scale，做初扰、实际reader anchor、writer adjoint及端到端方向检查，完整一次最多3,328目标。最新指令下不再运行准备04，也不将该模式作为主训练前置。主训练不沿用“12家族×128 targets×8步”的短测调度。

| 项目 | 计算 | 目标访问 |
|---|---:|---:|
| 初扰 self | 2×6×128 | 1,536 |
| 初扰 empty一次 | 6×128 | 768 |
| 新准备检查 | 8×128，详见§7 | **1,024** |
| **单次完整准备上限** | 2,304+1,024 | **3,328** |
| 准备02实际消费 | 2,304+128+128 | **2,560** |
| 准备03实际消费 | 2,304+128+128 | **2,560** |
| **主训练前既有诊断累计** | 9,444+2,560+2,560 | **14,564** |
| 准备04 | 最新用户指令取消，未执行 | **0** |

取消八步阶段后的第一次完整准备原上限为 **9,444+3,328=12,772**。准备02和03各消费2,560后在split lookup adjoint gate停止，均未进入匹配评估、writer FD及端到端FD，optimizer更新均为0。准备01在CE前失败，消费0。父账本现为14,564；准备03原累计上限15,332及准备04拟上限17,892均未全部消费，取消后不算作实际访问。每次失败与重复目标保留消费，不按成功次数或unique token计数。

较早的9,444构成为reader preflight33、norm FD440、TP结构11、原模式审计4,608、HC eager books768、模块定位768及sorted-router A/B审计2,816。新父账本保守把全部既有诊断同时计入75,000阶段上限及100,000全局诊断上限。H/scale和CPU参考无CE；准备阶段writer optimizer targets为0。全部原生主训练仍受累计100M writer loss-token上限约束，具体分配另行冻结。

## 6. 初扰与主训练边界

初扰仍使用每来源rank0，共6家族、2,304目标，报告每α/来源及总体Δ_empty。历史 `abs(Δ_empty)<.001 nat/token` 不再作为新有界训练的硬门槛；采用下节最新稳定性停止规则。初扰小不证明量化方向保留，不自动触发第二ρ或用probe调整scale。

旧step0/4/8 probe、八步gain与对应bootstrap均随独立短测一起取消，不进入本次准备调度。主训练继续保留loss/梯度/β/状态/reader冻结与预算的稳定性监控；主数据采样、优化步数、检查点与probe cadence须由主训练注册明确，不能把旧八步字段静默当作主训练合同。当前不声称独立DEV64/FINAL收益。

## 7. 所选数值路径、未完成检查与运行监控

A的来源已定位到原生router按grad模式改变top-k排序；固定 `sorted=True` 保留训练模式的顺序。books审计中，native/split两条PLE路径的初始writer与零Δ四种情况，grad/no-grad、grad-repeat的logits/loss全部逐位一致；两次实际backward的训练前向也与评估前向一致。此结论限于该固定样本与状态，不替代新准备流程的实际anchor比较。HC eager与compiled各自模式parity通过，但两者数值不相同；最终仍采用预先选定的HC eager，不按loss优劣选路径。

B使用native base BF16 TE key/value投影加独立FP32 Δ投影，norm/gate/conv以FP32执行，最后转BF16；所有条件共用此路径，`PLE_FUSED_KERNEL=0`。books实际writer观测中，投影相加后约99.99%的非零请求仍可见，cosine>.99999997；最终BF16转换仍有明显误差：有效/请求作用cosine约.36357、沿请求方向幅度比.9925、能量比7.45、relative-L2约2.543，约15.2%的非零请求坐标产生变化。沿方向幅度接近1不代表完整方向保留；不宣称量化问题全部解决或端到端loss光滑，也不新增100%转换保真门槛。

ρ保持.01，.1是已允许但未选择的fallback，不自动切换。上述books审计复用的历史H/scale仅用于dispatch与量化观察；主训练使用所选runtime下采集的base-only H和共同scale，以实际注册绑定其身份，不能沿用旧compiled向量。准备尝试中新empty与native baseline的比较保留，不因取消后续检查而抹去。

此前准备合同要求完整writer固定真实cotangent adjoint及一个端到端方向检查；这两项现按用户明确指令跳过，状态保持未完成。已通过的PLE FP64参考、books前向模式审计与本次实际lookup adjoint仍保留各自范围，不替代未完成项。旧SP结构证明仅沿不变的mapping源码和拓扑复用。reader边界使用materialized有效Δ，完整history写入与lazy衰减图保留。

未执行完的1,024检查额度原构成为：native empty128、实际backward anchor128、匹配评估128、两步长各正负一次共512、baseline重复128。原端到端合同为seed20261020的Rademacher、有效materialized memory坐标RMS=1、h=2⁻¹⁴/2⁻¹³、rtol=.20/atol=.001、两斜率互差≤20%、|解析值|<.01判未定、同RNG及逐位baseline。这里保存的是被跳过检查的原合同，不是通过证据，也不授权继续运行或调参重试。

[历史准备入口](../../../scripts/native_qwen38/qwen38_native_paired_norm_probe.py)与44项相关CPU测试、materialized两块history VJP结果保留。准备入口不再调度；CPU通过不代表被跳过的GPU检查或主训练完成。实际主训练状态以其新回执为准。

主训练监控按最新用户要求调整如下；旧短测step编号不再执行：

- 非有限loss、gradient或memory state，reader参数变化，任何预算越界：立即停。
- 每optimizer step的未裁剪writer梯度L2范数，以及超过此前5个完整step中位数10倍的连续次数：只记录和告警，不因该阈值停止。前5步历史不足时照常记录；非有限检查仍有效。
- 每step有效写事件中β<.001或β>.999的比例及其超过.01的连续次数：只记录和告警，不因该阈值停止。
- 固定probe总体平均NLL较初始基线恶化>.1 nat/token时停止并诊断；主训练单独登记样本和cadence，旧step4/8不再自动执行。
- 不要求每batch单调下降；主训练额度和延长依据必须在其注册内明确。
- 每个完整optimizer组保存latest checkpoint，减少中断时已完成更新的丢失；此次用户明确选择从seed17重新训练。

剩余准备调用已取消；重启继承14,564诊断父账本及54,784既有writer目标，按实际dispatch继续计数，不归零。主训练注册明确重启授权、仅监控的两个指标和未完成检查，不把“跳过”改写成“通过”。

## 8. 当前证据与限制

原runtime的TP结构检查接受`replica`倍率，仅覆盖SP/lookup adjoint。PLE CPU参考回执SHA `f34c36c20e274c35df32f40f77f66305a2185533880b1db89d8c6cbb1355f2b6`已独立核验：12个stage与3个整链方向全部通过两个最小预登记FD步长。native BF16/FP64输出relative-L2=.0042922354、cosine=.999990889，embedding VJP relative-L2=.0069462134、cosine=.999975930。结构快照的合成Δ=`sin(…)×.01`不是实际writer；`reference_fd_passed`仅说明参考自洽，native局部kernel差异另列，不宣称已全部验收。

原六来源审计及单独HC eager检查的模式差异仍保留为历史证据；修复后的books审计回执为 `ab_repair_books_01_receipts/COMPLETE.json`，本地字节SHA已核验：`8d18a0c8f8fc9bf554aa955d1e3a9d8ab4f3089831ae37249683dd4c419d6c16`。其2,816目标、两次backward、零optimizer，只覆盖§7所述固定books样本，不能推广为全部数据上的梯度或训练验收。

split的两TP量化统计已记录，最终BF16转换的方向误差仍须随实际运行报告。历史H下split self比empty高约.04006 nat/token，只是该诊断样本的观测，不能作为新尺度训练的结论。

准备01在CE前因回执字段读取错误停止：runner把历史observation的identity当作preflight schema，读取不存在的 `adapter_sha256`；其实际字段为 `adapter_source_sha256`。历史 `writer_source_sha256`也已不同于接受TP证明时的native源码，不能仅改字段名后沿用。修复改从接受TP结构检查的外层session回执读取adapter/native pins，核对注册SHA、外层result与内层结构回执逐项相等、模型验证身份及当前代码；该检查现在在CPU校验阶段完成。接受的TP外层SHA为 `b8513a1c58d43accce10d0bbdfef3e0a0b7a2fc4ae83e9db9020c600cbeb6a8d`。准备01新增CE和optimizer均为0，未绕过原证明。

准备02在实际split lookup adjoint gate停止，消耗2,560 CE、optimizer0；原实现先检查后保存，缺少失败细节。准备03在gate前保存了两TP统计：两个manual adjoint的最大误差都为0；TP原始梯度max-abs差4.656612873e-10、relative-L2差2.036203e-10、cosine=1，235个坐标不逐位相等。因此触发的是额外的TP逐位一致条件，不能据此称lookup伴随断裂；微小差异的具体kernel来源尚未证明。主训练的副本处理沿既定rtol=1e-6/atol=1e-7比较并使用一份规范副本，不将TP副本求和。准备03同样消费2,560、optimizer0。

取消准备04、跳过剩余writer FD/E2E后，首轮执行2次更新；运行02随后按用户要求从seed17重训，并将梯度突增和β饱和改为仅监控。用户进一步要求提升批处理和流水线吞吐，运行03现从运行02第12组、263,433个已训练target的完整状态接续，累计writer预算318,217、诊断14,564。没有追加旧准备阶段检查。完整历史见[启动登记§9](qwen38_native_launch_registration_20261020.md#9-基础设施验收检查点)，新运行实际结果以独立回执记录。

## 本次重启冻结

主入口 SHA `f237a3045d74a1c28c2b46f27fa2fa88a0966ed0543bd83a2020febc8a9d05f8`；监控 helper SHA `d2cdee8f4daed3690174d24528f5bb4298d7b5bba6178298bc64f751271d8d58`。实际[重启登记](qwen38_native_main_training_restart_20261020.json) SHA `25579bcbe5498ff869c1736f162a303cd1b341f8ccb4f4f36b0fb8ebbfeb7c58`。旧消费计入每臂2M额度，因此本次 α=1 最多1,945,216、α=.99最多2,000,000目标；实际完整轨迹端点可能略低。启动进程不等于已经完成更新，进度以运行中的 progress.json 为准。
