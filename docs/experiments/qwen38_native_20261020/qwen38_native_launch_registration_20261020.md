# Qwen3.8 原生主训练启动与历史准备登记

2026-10-20。**用户要求修复 batch=1 和 PP2 单微批吞吐后，运行02在第12组完成时停止；运行 native_main_training_03 已启动（PID40692），从263,433个已训练target的检查点继续。**新版按独立轨迹组成B4、每波最多3个微批送入真实PP流水线，随后收拢仍活跃的轨迹。writer、AdamW、sampler、四rank RNG及稳定性历史全部接续；没有重新初始化训练。本次切换没有检查点之后的在途target消费，累计writer预算为318,217，诊断仍为14,564。实际吞吐和显存以新运行的完整训练组回执为准。

上一次运行02于10:19 UTC启动（PID36991），根据用户要求从seed17重训，并将梯度突增与β饱和改为仅监控。该设置继续保留；sorted router、HC eager、split PLE、固定RMSNorm及ρ=.01保持。准备04与独立八步短测继续取消，完整writer FD/E2E仍未完成，不能记为通过。详见[决策与准备记录](qwen38_native_normshort_registration_20261020.md)；历史JSON保留，批处理接续使用独立登记。

18:46北京时间实测：新版已完成第13–17组，127,698 target / 237.93秒 = 536.70 target/s；去掉含首次B4编译的第一组，后四组为83,666 / 132.89秒 = 629.60 target/s。旧版12组平均266.92 target/s。这是相邻不同采样组的早期运行吞吐，尚非同组配对基准。每组实际B4、最多3微批；五组峰值allocated显存最高85.22 GiB。CPU17项测试及独立审阅通过。用户随后明确只保留批处理：没有启用EP4、激活重算、证据预缓存或TP抽查。见[接续登记](qwen38_native_main_batched_continuation_20261020.json)及[吞吐回执](qwen38_native_batched_throughput_20261020.json)。

依据：[训练计划 §9](../../training_plan_20261019.md#9-原生诊断阶段与预算)、[五源执行计划](../../multidomain_execution_plan_20261019.md)、[原生架构](../../design/qwen38_native_20261020/architecture.md)。用户已决定推进原生模型，原始文本及冻结划分从 B2 恢复，使用原生 tokenizer 重新编码；不重新抓取数据源。资源、adapter 与 runner 的实际状态由各自验收回执记录。

## 1. 确定项与启动前必须落盘的项

| 项目 | 当前登记 |
|---|---|
| 模型 | `Qwen/Qwen3.8-Flash-Next@de4b8e4d43b917e7706784d8bb445c9af86a3540`；实际权重、配置、代码及 tokenizer 文件另核 SHA256 |
| 优化对象 | 冻结 reader 和原生基表 T⁰，仅优化独立原生 writer；不加载小模型 writer 作为同形兼容 checkpoint |
| α | 仅固定 `1.0` 与 `0.99`；相同初始化、数据顺序、H 处理、ρ、每头尺度及优化配方 |
| Writer | 输入 2560、行宽 160、隐层 256、16 个头、右窗 4、D1 块内平均；原生地址含有效行 0，稀疏状态用 `native_id + 1` |
| 预算 | 接口/容量诊断合计 ≤100,000 目标 token 访问；所有原生 writer 优化合计 ≤100,000,000 loss tokens |
| 当前状态 | 首轮完成2次更新后按用户要求停止；新运行从seed17重启待执行，梯度突增/β饱和改为仅监控；独立八步阶段取消 |
| 第一更新前 | 将上述选择、全部父文件 hash、校准回执、预定检查点与停止条件写入不可覆盖的 run registration；配置 hash 纳入恢复 identity |

本文件保留完整原生分支及历史准备边界。ρ=.01保持唯一已选尺度；用户最新明确授权在剩余检查未完成的状态下直接主训练。主训练注册记录这项授权及未完成项，保留数值、reader冻结和预算停止条件；不再启动小样本短测或追加准备检查。

## 2. B2 恢复、原文身份与重新编码

1. 先验证 B2 对象大小和 SHA256，再验证包内成员 manifest、冻结数据划分、文档/家族/事件 ID、正文 hash、donor 和干扰文档配对。训练、DEV64、封存 FINAL 保留原集合，不重新抓源，不因新 tokenizer 丢弃难例或重新选择 donor。
2. 恢复原始 UTF-8 正文与已冻结的文档集合及 split。使用固定原生 tokenizer 将完整文档编码一次，再切 token IDs；不把旧 token IDs decode/re-encode 当作原文恢复。保存正文 hash、新 token IDs hash、tokenizer vocab/pipeline/special-token 指纹。
3. 复用五源原配比：code/papers/books/web/news/news_event 的 loss-token 权重为 `0.20/0.25/0.20/0.15/0.175/0.025`。前四源 8/16 块各半，news 2/4 块为 90%/10%；news_event 原有长度权重与文章边界从原 manifest 继承。512 新 token/块、最多 128 局部 prefix。新的可用窗口数、尾部、目标容量和重复消费率必须重算，旧 tokenizer 的容量不是原生容量。
4. 各报道之间重置局部文本 prefix 与 n-gram 边界，保留同流记忆；独立流之间清空记忆。prefix 不重复写入。原生特殊 token、EOS 和边界行为需由真实 adapter 验证。
5. 若原冻结来源在新 tokenizer 下不足以支持某个长度/份额，记录缺口并停止受影响的配置冻结；不静默拼文档、改 split、换 donor 或改变比例。新配方另立有父 hash 的修订。

可核对的旧身份（仅作为 B2 恢复的父身份，不能当作新 tokenizer 产物）：

| 父产物 | SHA256 / 已记录规模 |
|---|---|
| M1 `train_validation.jsonl` | `01748d406b7915e7b6a9347d703118fdbcafb4d64d3f02d71f29aad4b119f046` |
| M1 `train_validation_event_streams.jsonl` | `b0bad66e169f7c0394cdab88843f0df8c3c6015bb0c9446ee44aa30a0f6e6aaa` |
| 旧 data manifest | 41,635 documents、937 event documents；旧计数 187,925,853 raw tokens 与 2,486,100 event raw tokens，不等于新训练 token 预算 |
| DEV64 registry | `9ea663ad70131e272522cbad8f6c0fa20bc03f41b324fe7b71d0fb8f4fbfccc2` |
| DEV64 prefix annotations | `7164e9f8b6f23ebfeaaccc2687a6a8b6ad222803f705bacd1a203a2596fdbcce` |

FINAL 仅恢复、核对外层完整性与隔离身份；本阶段不打开其文章、问题或答案，不用其内容校准或选 checkpoint。

## 3. DEV64 的原生 tokenizer 派生登记

### 3.1 已核实的原始标注

旧 DEV64 有 64 个事件家族、128 篇配对正文、18 篇背景正文。标注有 **107 条 key facts**；每条保存早期原文的 `source_id / char_start / char_end_exclusive / text_sha256`。原复述阳性项另保存 `later_prefix_span`；每组保存旧前缀结束字符位置、前缀 token IDs hash 和 text hash。旧 `34 restated / 30 not_restated` 绑定 Qwen3-0.6B tokenizer，不能直接当作原生分组。[现有校验代码](../../../src/delta_ngram/recent_news_evaluation.py)逐项验证这些身份与字符 span。

### 3.2 在任何原生模型评分前重判

- 使用同一冻结正文与同一已记录命题，生成 Qwen3.8 的前 128 token、精确字符边界及 hash，形成新的派生 registry/annotation，保留旧版本和父 SHA。
- 逐组重新核对命题是否出现在新前缀。旧阳性 span 仍完整落在新前缀内是可复用的正证据；新前缀截断旧 span 时重新判断。新前缀比旧前缀长时，旧阴性不能证明新增区域没有复述，必须检查新增文字。
- 字符 span 指 Python Unicode 字符切片，不是 UTF-8 字节或 UTF-16 code units。若 token 边界无法解码为原文的精确字符前缀，标 `unknown`，不猜测。
- 每组保留旧/新标签、边界变化、理由、核对者和时间；新 restated/not_restated/unknown 数量按实际结果报告，不能按模型效果分组。donor、干扰文章顺序、事件集合不改变。
- 重新计算目标区、损失 token、write-block 数及地址；同一原生比较内各条件使用完全相同目标。不同 tokenizer 的 nat/token 不直接作跨模型等量比较，另报相同原文范围的损失总和和原文字符/字节量。

### 3.3 时间与预训练污染边界

DEV64 所有配对正文登记的发布日期范围为 **2025-08-01 至 2025-12-25**；原 registry 已写明 `training_cutoff="unknown"`、`base_pretraining_novelty_certified=false`。时间审计仍需核对具体事实首次发生/公布时间、文章修改时间及 `available_at`；文章发布时间不自动等于事实首次出现时间。

2026-10-20 对固定 3.8 revision 的 [Hugging Face 元数据](https://huggingface.co/api/models/Qwen/Qwen3.8-Flash-Next/revision/de4b8e4d43b917e7706784d8bb445c9af86a3540)读取结果为仓库 `createdAt=2026-08-24T08:24:59Z`、revision `lastModified=2026-08-27T05:03:36Z`。[官方模型卡](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)把发布文章和技术报告列为 2026 年 8 月；本次已查材料未提供可用的预训练截止日期。仓库创建/修改时间不是已认证的权重首次公开时间，更不是预训练截止日期。

因此保留 `pretraining_cutoff=unknown`，不断言具体事件已见或未见。既有冻结事件可用于 self/empty/donor 对照和历史信息利用研究；“新信息”结论还需独立的污染证据。若无法排除污染，另行登记有时间依据的新事件集或沿用原计划的训练后随机绑定测试，不能静默替换原 DEV64/FINAL，不能把 empty 的错答当作未见证明。本稿不授权新增采集或扩大评测预算。

## 4. 固定 RMSNorm、校准与已取消的短测

用户已经敲定无参数RMSNorm，epsilon=1e-6，不加入log(RMS)。原两α各8步、24,576优化目标及step0/4/8 probe已取消；没有以该协议执行优化。旧37,632阶段及48,100累计计划完整保存在[取消阶段快照](archive/bounded_eight_step_canceled_20261020/qwen38_native_launch_registration_20261020.md)及其[hash manifest](archive/bounded_eight_step_canceled_20261020/archive_manifest.json)。主训练使用独立数据与预算注册，不把24个校准家族当作主数据池。

从calibration24每来源按selection_hash/group_id排序取前两个家族训练和校准，后两个固定train-side probe。旧compiled路径观测包含12家族、18个history frames、8,403新tokens；H RMS=3.729683，RMSNorm raw-target RMS约.003768，T⁰约.00801–.00849。所选split-precision运行路径固定后重新生成H及每头scale；旧数值仅保留历史，当前无有效scale向量。唯一rho=.01保持，两α共用新向量；用户允许的.1仅是未选择fallback。

取消短测后的准备计划为初扰2,304加检查1,024，完整一次3,328，初始累计上限12,772。准备02和03各实际消费2,560后停止，准备01为0，故当前诊断父账本为 **9,444+2,560+2,560=14,564**。准备04已取消；旧1,408-target writer loss FD也未执行。未用预留不算实际消费，失败或重复目标不归零。

## 5. 真实原生 preflight 验收矩阵

已有真实 reader preflight 在固定 TP2/PP2 的合成输入上通过 empty logits parity 和 reader-to-delta 非零梯度；其回执明确不验收 writer BPTT、TP writer reduction 或 resume。以下是完整验收范围，未由实际回执覆盖的项仍待验收；CPU 组件通过、模型下载成功或 CUDA 版本匹配不能替代。

| 项目 | 必须保存的证据 |
|---|---|
| Δ=0 前向 | 同权重/输入/dtype，无 overlay 与空 overlay 的 logits、loss 差异；覆盖 fused 与可诊断 reference 路径、实际 CPU-offload 路径 |
| 原生寻址 | 16 个头区间、有效原生行 0、`+1` 映射、padding 排除；native hash 与 overlay 一致 |
| prefix/EOS/文章边界 | 同一完整文档切块，128-token prefix 不重复写，跨块 n-gram 与原文一致；EOS/局部 BOS、文章重置和 masked padding |
| 真实梯度 | 固定证据允许行上的 FP32 Δ，经真实 key/value/gate/conv 回到 Δ；小规模方向导数/有限差分；保存 ε 扫描、dtype、误差和预设容差 |
| Fused/offload 梯度 | 实际训练路径直接验收，不能只在 reference 路径有梯度；T⁰ 冻结，Δ 的设备传输/加法不断图；若 fused 不支持反传，只能明确选用已验收路径 |
| 因果与隔离 | 后续 loss 到早期 writer 有梯度；改变未来块不改过去 logits；同批 A 写入不影响 B；当前块预测完成后才写 |
| 重算与多卡 | 需要哪种配置就验收哪种：同快照 loss/梯度 parity、rank 间状态归属、全局 loss-token 归一化、通信路径；不把单卡结果推广到多卡 |
| 恢复与预算 | 独立进程恢复的 writer/optimizer/RNG/sampler/counters/identity；继续训练不清空预算；完整优化边界不越封顶 |

低精度量化会使极小 ε 的有限差分失去分辨率；先在可诊断 FP32 部件与真实路径的较大可分辨 ε 上查方向导数，再报告低精度误差。不能通过任意放宽容差把断梯度判为通过。无 reader 参数梯度不代表无 Δ 梯度；预测 forward 不能被整体 `no_grad` 包住。

## 6. token 账本与停止规则

### 6.1 硬预算（继承已有计划）

- **100,000 接口/容量诊断目标 token 访问**：原计划建议接口调通 ≤20,000、容量诊断 ≤80,000。候选尺度、重启、重复目标优化、有限差分及仅证据梯度基线的实际目标访问各自重复计数，不按 unique token 隐去；本轮保守将校准探针重复评分也计入接口子账本。没有损失目标的 H/T⁰ 观测另记 encoder/input 成本。
- **100,000,000 累计原生 writer loss tokens**：两臂、pilot、延长、重启、对比及被放弃训练都计入同一个总账本。初始短试训每臂 ≤2,000,000，两臂合计最多 4,000,000，从总额扣除。**后续两臂分配尚未冻结；没有每臂各 100M 的授权。**
- 旧计划 `P=max(2M,m_0.6B)`、pilot 最多 30M 是预登记延长依据，所有消费仍在总额内。`m_0.6B` 必须从原预定检查点规则核实，不能用任意有利 M1 指标冒充；开始延长前登记两臂累计额度及检查点。不因初始 2M 无显著收益就判方法失败。
- P6 仅可用 `100,000,000 − 已消费全部原生 writer token`；扩展需本次原生证据。P5 raw/对话收尾不是本次自动启动项。
- 独立评测的输入/目标/重算 token、次数、GPU 时长另列；不从 100M 中隐去训练，也不把“另记”解释为无限评测。FINAL 保持封存。

### 6.2 预设检查与停止

**Proposed** 初始两臂检查点为 0、0.5M、1M、2M loss tokens，在不超预算的完整 optimizer 边界执行，记录实际值与未消费尾额；相同检查点、样本和 bootstrap seed。训练侧每个更新记录 loss、裁剪前梯度范数、clip 状态、β、target/Δ/T⁰ RMS；H 分布按预定探针报告。不得因看到曲线后加密有利时点。

任一身份/hash、split 隔离、预算、因果/梯度/数值验收失败即停止受影响阶段，保存回执及最近可恢复点；非有限 loss/梯度/状态、实际更新无梯度或断图同样停止定位。OOM 只允许在保持全 BPTT、有效全局目标 token 归一化和配方身份的前提下降低 microbatch/增加累积，改变后重做相关 parity；不静默 detach、缩短轨迹或每块更新。

P6 仍执行原计划 §6.4：拟扩大的原生 reader/writer 在冻结开发协议上有 Δ_empty 和资源依据，Δ_donor 的文档家族配对 CI 下界为正，训练后独立随机绑定相对 empty 与 wrong-state 的预设主量都为正。具体阈值沿原登记，不用 M1 的效应替代原生 gate。样本或时长不足则报告未定，预算耗尽则停止；不循环重启到通过，不打开 FINAL 寻找通过条件。

## 7. 实际产物与运行入口

本次采用实际原生 runner 的 `registration.json`、`frames.json`、`COMPLETE.json`、逐块 artifact 和独立 FD receipts，统一绑定模型、代码、样本、输入和预算。无另一个 JSONL 校准输入协议或独立候选选择器。

- 数据派生：[prepare_qwen38_native_data.py](../../../scripts/prepare_qwen38_native_data.py)。它不加载模型、不运行优化，恢复原始正文并编码完整文档一次。
- 原生 loader/preflight、无标签观测、TP/完整 writer FD 与 paired 短测入口集中在 [scripts/native_qwen38](../../../scripts/native_qwen38)。运行行为以各入口已保存的代码 hash 为准。
- 数值合同：[短测 JSON](qwen38_native_normshort_registration_20261020.json)；阅读版：[短测说明](qwen38_native_normshort_registration_20261020.md)。最终可运行注册由固定诊断通过后独占新建，草案不能直接授权 optimizer。

本轮本地回执位于 `delta-ngram-runs/qwen38_native_launch_20261020`：`initial_receipts` 保留 reader preflight 与 calibration24 身份；`native_full_receipts` 保留全量派生和权重 header 审计；`norm_observation_receipts` 保留 H/T⁰ 与首轮未定 TP FD，`tp_fd_02_receipts` 保留第二轮未定回执；`norm_scale_calibration_20261020.json` 保存真实观测推导的单候选 scale。完整 tensor/cache 在运行主机，紧凑本地 manifest 不等于本地核验了所有 payload bytes。

核心模块位于 `src/delta_ngram`：`qwen38_native`/`qwen38_adapter` 管理地址与原生PLE接口，`qwen38_training` 管理轨迹梯度及lookup adjoint，`qwen38_precision` 实现split PLE，`qwen38_train_data` 提供主数据采样接口。主训练入口为 [qwen38_native_train.py](../../../scripts/native_qwen38/qwen38_native_train.py)；旧paired入口只保留历史准备实现，不再调度。HC在训练scope使用两个固定源码函数的eager实现，router固定top-k排序；各实际运行的代码及注册SHA独立保存。CPU检查不代替未完成GPU检查。

## 8. 全量原生数据容量复核

全量 manifest SHA 为 `236014f3f18286138d582e97c7454ee2e5780b3ce38e1e03488ba83bf33f7592`，seal SHA 为 `2be8e2ddaad69a0c48be38cad195e90882100a04a3e052983de8ea37165f8caa`。已核对 seal→manifest→selection 的本地 bytes/hash；共同父 manifest 与 calibration24 一致，原文/划分集合未变。

- 普通文档：38,566 train / 3,069 validation，共 41,635。
- 事件：713 train / 224 validation，共 937 流、2,002 篇文章。
- 新原生原文 token 总量 193,958,754（train 182,180,345；validation 11,778,409）；uint32 cache 为 775,835,016 bytes。旧 tokenizer 合计 190,411,953，不能沿用旧 token 计数。
- 事件训练后续文章目标为 1,007,625；所有注册长度单元非空，violations=[]，原来源和长度权重保留。

按固定来源/长度 weights 将**累计 100M**分配到 14 个训练单元，最大预计重复消费为 event 4-block 桶 2.691865 遍，低于原定 3 遍；event 32/8/16 桶为 2.506777/2.467605/2.429230，code 16-block 为 1.724614，其余更低。这里的 100M 是所有原生 writer 分支总额，短测与 pilot 消费还须扣除，不能变成每臂各 100M。8/16 两套窗口会复用原文，不能相加当作 unique targets。

全池默认 calibration12 的 sample_id 与 calibration24 各来源前两个完全一致；本短测仍只使用固定小样本反复访问 128 targets，不能称为已经训练整个 193.96M 原文池。ready_for_training 表示数据派生约束通过，不代替 native runtime、固定 donor 可用性、DEV64 prefix 重分类或预训练污染验收。

## 9. 基础设施验收检查点

以下记录当前证据；writer、RMSNorm及所选rho=.01不变。旧H/scale仅保留历史，所选split-precision运行路径固定后按原训练家族规则重算。

| 检查 | 已有证据与范围 |
|---|---|
| TP 结构倍率 | 原 runtime 的 SP/lookup adjoint 最大误差为0，FP64线性检查通过，接受 `replica`：一份逻辑CE梯度，不把TP副本相加。回执SHA `e8ddb1d127d9f8ae28e94512645e4cf419d9a070ecb73980d666b3671e11d0bb`；只验倍率/通信边界，不验原生局部kernel导数或writer BPTT。 |
| PLE FP64参考 | 实际权重快照的12个stage、3个整链方向，全部通过预登记的两个最小FD步长。native BF16相对FP64的输出relative-L2=.0042922354、cosine=.999990889；embedding VJP relative-L2=.0069462134、cosine=.999975930。已独立读取并核验回执SHA `f34c36c20e274c35df32f40f77f66305a2185533880b1db89d8c6cbb1355f2b6`。 |
| 参考检查边界 | 输入来自结构快照，Δ=`sin(…)×.01`，不是实际writer状态；本次CPU参考新增CE目标为0。`reference_fd_passed`只证明参考函数的导数自洽；native VJP/输出误差是另列观测，尚未形成原生导数验收。 |
| 原始模式审计 | 六来源×两状态×三模式，共4,608目标；12组grad-repeat logits/loss均逐位一致，grad/no-grad却全部不一致。该结果确认可重复的模式差异，不能称训推等价。回执SHA `8ea1203e0aa8542c98ab0033833f0b772946b89fb49b4e0ab86c328805a28d7a`。 |
| 实际writer量化可见性 | 非零Δ坐标中10.7–14.4%改变BF16 lookup；52.7–79.2%的请求Δ能量位于未改变坐标，有效扰动能量/请求能量约33.8–70.9%，二者不是互补量，也不证明方向保留。请求Δ与有效Δ的直接cosine仍待补；PLE变化非零目前只覆盖TP0局部位置，须扩展TP覆盖。 |
| 单独HC eager历史检查 | 消费768 targets，两组grad-repeat逐位一致但grad/no-grad仍不同；仅解包HC不足以消除A。此为后续router定位前的历史结果。 |
| Sorted-router A/B检查 | 固定books、native/split两条PLE路径×初始writer/零Δ四种情况，grad/no-grad与repeat的logits/loss全部逐位一致；两次实际backward的训练前向也与评估前向一致。消耗2,816目标、optimizer0。回执SHA `8d18a0c8f8fc9bf554aa955d1e3a9d8ab4f3089831ae37249683dd4c419d6c16`已本地核验。 |

较早诊断为 **9,444 = 33 reader preflight + 440旧norm FD + 11 TP结构 + 4,608原模式审计 + 768 HC eager检查 + 768模块定位 + 2,816 sorted-router A/B检查**；加准备02和03各2,560后，当前父账本为 **14,564**。这些历史回执optimizer更新均为0，失败检查消费保留；CPU参考新增CE为0。

**最终runtime固定为sorted router + HC eager + split PLE。**A定位到router原来的 `sorted=torch.is_grad_enabled()`；固定 `sorted=True` 保留训练top-k顺序。修复后的books检查通过，但不推广为所有样本已验收。HC compiled/eager各自模式parity都通过，两者数值仍不同；采用预先选定的eager，未按loss择优。主评测保持grad-enabled/no-backward，准备流程仍直接核对真实训练anchor的logits/loss；其计算图、激活与资源成本照实记录。A的no-grad parity不新增为训练硬门槛。

B的FP32投影支路保留了约99.99%的非零请求，books观测cosine>.99999997；末尾BF16转换仍有明显方向误差：cosine约.36357、沿请求方向幅度比.9925、能量比7.45、relative-L2约2.543、变化坐标约15.2%。这些是两TP上的实际writer诊断，不证明完整方向保留、端到端loss光滑或B已全部解决；不新增100%保真门槛。native/split empty也不同，必须重新记录baseline。上述审计使用历史H/scale，不能替代最终runtime下的重算；ρ=.01保持，.1不自动启用，`PLE_FUSED_KERNEL=0`保持。

准备01在CE前因runner读取错误schema字段而停止：历史observation使用 `adapter_source_sha256`，并没有preflight的 `adapter_sha256`；其native相关源码hash也已经过时。修复从接受TP检查的外层session回执读取adapter/native pins，校验注册SHA、外层result与内层结构回执一致、模型身份及当前源码。外层SHA为 `b8513a1c58d43accce10d0bbdfef3e0a0b7a2fc4ae83e9db9020c600cbeb6a8d`。准备01消费0 CE/0 optimizer，未绕过证明。

准备02/03均在实际split lookup adjoint gate停止，各2,560 CE、0 optimizer。03提前保存了失败证据：两个manual adjoint最大误差均为0；TP原始梯度max-abs差4.656612873e-10、relative-L2差2.036203e-10、cosine=1，235个坐标不逐位相等。触发的是额外逐位条件，不能据此称伴随断裂；具体kernel来源尚未证明。主训练副本处理使用既定rtol=1e-6/atol=1e-7和一份规范副本，不把TP副本相加。剩余匹配评估、writer FD/E2E没有完成；原方向和容差只保留历史，不追加运行。

**首轮主训练与用户重启授权：**用户要求停止检查后，主训练于2026-10-20 09:48 UTC启动，远程PID32914；历史入口代码SHA `ec5361e8187937798cfd70d269f55a7fe0a5bfbe3729ef6e0e16e9467d4f03b8`，历史注册SHA `360236895ee7b77418f9ad463aff08ee06c04fe8a628ffb46cc729be4bc47034`。该进程已按后续用户要求停止：2次完成更新对应34,816目标，另有19,968在途目标，实际dispatch累计54,784全部保留计费；诊断14,564分账保留。

用户明确允许从相同seed17初始状态重训，不从第2步checkpoint继续；新进程及新代码/注册身份待其回执，旧身份不冒充重启配置。梯度突增与β饱和两项由硬停止改为只记录和告警；每个完整optimizer组保存latest checkpoint。非有限loss/grad/state、reader参数变化和预算越界仍保留保护，其他具体额度与probe cadence以主训练注册为准，旧八步step4/8不执行。不追加准备检查；完整writer FD/E2E继续标未完成，不把跳过写成通过。
