# Delta N-gram 最终训练方案

2026-10-19｜最终执行版｜数据、训练方法、评测门槛与实施顺序

## 1 已确定的主线

**冻结部署 reader，用真实文本流的后续 next-token loss 训练 D1 writer；块间按顺序更新记忆，先完成正确的批量训练，再依据实测瓶颈优化。**本轮不以推导通用时间并行算子为前置条件；若保留 D1 的并行推导和实现成本过高，留到后续论文。

机制训练、原生 n-gram 表的训练、前向记忆评测分别讨论。本轮只优化 writer，reader 与原生 n-gram 基表冻结。训练文本可以是基模型可能见过的内容；前向评测固定全部参数，只改变本次在线增量状态，分别报告细节恢复、新信息、冲突修订和随机绑定。

先准备约 **20M 合格代码原文 token**，做 0.6B 的 **5M 目标 token** 小试；信号成立后，数据池扩到约 **200M**，四源混合主 run 最多 **100M 目标 token**。原生接口提前检查，原生最大 run 单列预算。只有最贵的 **P6 原生扩大训练**增加“文档相关收益”和“随机绑定正收益”两项门槛；P1 仍按可训练性与 `Δ_empty` 推进。

本文的配比、阈值、检查点与预算是执行前预设，**不是实验结论或效果承诺**。实施状态截至本版：本地 CPU 全套 343 项测试通过；新 GPU 环境已完成 512-token 块、128-token prefix 的重算、恢复和短时吞吐工程验收。迁移验收时的 318 项基线中，新环境为 317 项通过、1 项历史 bitwise fixture 失败，未放宽断言。此前 GPU 验收使用高多样性合成输入，不能据此宣称正式语料训练有效。启动准备已补齐有界代码数据构建、固定开发集的组级配对 bootstrap CI 和中途保存；原生接口及完整 P1/P6 阶段结论仍待相应实验。

本文件为仓库内的后续执行依据。原稿与讨论保留为本地归档文件名（未随仓库分发）：`delta_ngram_training_plan_final_20261019.md`、`delta_ngram_raw_text_data_plan_20261019.md`、`delta_ngram_raw_text_data_plan_v2_20261019.md`、`delta_ngram_raw_text_data_plan_review_20261019.md`、`delta_ngram_raw_text_data_plan_review_reply_20261019.md`。

## 2 数据来源与可用量

首版为英文长文和程序源码。混合比例按**实际参与损失的目标 token**控制；原文池配额是采集目标，另需验收各来源能提供的长窗口数量。

| 来源 | 数据集 | 目标 token 比例 | 200M 合格原文池起始配额 | 连续单位 |
|---|---|---:|---:|---|
| 代码 | [Common Pile Stack v2 Edu filtered](https://huggingface.co/datasets/common-pile/stackv2_edu_filtered) | 30% | 60M | 单个源文件 |
| 论文 | [Common Pile arXiv papers filtered](https://huggingface.co/datasets/common-pile/arxiv_papers_filtered) | 30% | 60M | 一篇论文 |
| 书籍 | [Common Pile Project Gutenberg filtered](https://huggingface.co/datasets/common-pile/project_gutenberg_filtered) | 25% | 50M | 一本书 |
| 长网页 | [FineWeb-Edu `sample-10BT`](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) | 15% | 30M | 一篇网页 |

沿用已锁定的数据版本，下载清单仍须记录具体 shard 和校验和：

| 数据集 ID | revision | 正文路径模式 |
|---|---|---|
| `common-pile/stackv2_edu_filtered` | `c354dbe88469a1153e97c6a63ac50591849654de` | `stack-edu-*.json.gz` |
| `common-pile/arxiv_papers_filtered` | `033cf7f53f9b348deec868c1a5a48484f3ee9e52` | `arxiv-papers-*.json.gz` |
| `common-pile/project_gutenberg_filtered` | `3cdf6879c807f4e4e063f2ceb23bc268d8c29ab7` | `project_gutenberg-dolma-*.json.gz` |
| `HuggingFaceFW/fineweb-edu` | `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9` | `sample/10BT/*.parquet` |

首批约 20M 清洗、去重后的代码原文计入后续 60M 代码配额。跨 shard 固定 seed 打乱取样，按程序语言分层；Markdown 不占代码配额，单一编程语言原则上不超过代码目标 token 的 40%。不把整个仓库随意拼成一条文本流。

### 清洗、分组与切块

1. 保留自然段落、章节和源代码顺序；去除导航、重复发行说明、乱码、机械复制、生成代码和 vendor 副本。保留自然发生的 n-gram 复现。
2. 先做文档精确与近重复去重；作品版本、论文版本、代码仓库和近重复家族共同决定分组。
3. 按组划分约 **95% / 2.5% / 2.5%** 的训练、开发、保留集，报告实际 token 数。同组不得跨区。最终七任务及其支持、探针和派生材料独立留出。
4. 每个要作来源结论的开发分层，目标至少 64 个独立分组；不足时补采或标为证据不足。同一本书的许多窗口不能冒充独立文档。
5. 完整文档用目标 tokenizer 编码一次，然后按 token IDs 切块，禁止 decode/re-encode。8 块窗口至少 4,096 token，16 块至少 8,192 token。不同文档不拼接补长度。
6. 保留完整文档与长度索引，报告各来源 8/16 块可用目标 token、重复消费率和未用尾部。原文总量达标而长窗口不足，仍未完成准备。

原始层保存来源 ID、文档 ID、逐条许可、provenance、正文 hash，以及代码路径和 revision。代码保留 `metadata.license`、`detected_licenses`；论文保存原始 arXiv ID 与逐篇许可；书籍保存作品 ID 与来源权利声明；FineWeb-Edu 的数据集许可不能替代原网页的权利条件。不使用丢失来源字段的 text-only 合并视图。

## 3 Reader 身份与训练对象

| 对象 | 机制训练 | 前向记忆评测 |
|---|---|---|
| Reader 参数、原生 n-gram 基表 | 冻结 | 冻结，empty 对照也保留基表 |
| Writer 参数 `φ` | 由后文预测损失优化 | 冻结，不按评测样本更新参数 |
| 在线增量状态 `M` | 随文本块更新，轨迹内保留计算图 | 仅前向更新，独立文档或事件重置 |

小模型主线直接使用后训练版 `Qwen/Qwen3-0.6B`，不是先在 Base 上训练再默认可迁移。原生分支沿用项目指定的 `Qwen/Qwen3.8-Flash-Next`，项目记录 revision 为 `de4b8e4d43b917e7706784d8bb445c9af86a3540`。启动前核对实际权重指纹、tokenizer、模板、特殊 token、原生读写接口与注入位置并写入 manifest；本方案不把模型名称相同视为接口已验证。

原文阶段向冻结的后训练 reader 输入原始文本。后续对话适配才使用部署 chat template、`enable_thinking=False` 和 assistant-only loss。原文 next-token 训练不代表输入分布已经等同于对话部署。

外置 0.6B 原型与原生模型的状态维度、基表和读出接口不同，writer checkpoint 不直接互换。正常加载必须匹配 `base_model / tokenizer / addressing / writer / operator / trajectory` 六类 identity；state snapshot 也不是包含优化器、数据游标和随机数状态的完整训练恢复点。

P2 只做一项有预算的 Base 诊断：复用 P1 在后训练 reader 上 5M 的 writer；另在 `Qwen/Qwen3-0.6B-Base` 用相同数据顺序、初始化与优化设置训练 5M。主比较把两者放到**同一个后训练 reader**和同一评测协议中。仅在 tokenizer、地址和参数形状兼容时进行研究性 `B→I` 迁移；不兼容就记录原因，不能把各训各测当作迁移证据，也不放宽正常 checkpoint 的身份校验。

## 4 训练方法

### 4.1 先预测，再写入

| 项目 | 主线设置 |
|---|---|
| 每块新文本 | 512 token |
| Reader 可直接看到的旧原文 | 上一块末尾最多 128 token |
| 轨迹长度 | 8 块和 16 块混合，各占 50% 有损失目标 token |
| 首块 | 只写入，不计 loss；写入计算图保留 |
| 损失 | 当前块全部合法目标 token 等权 causal cross-entropy |
| 梯度 | 轨迹内完整 BPTT；块间不 detach 状态 |
| 参数更新 | Writer 在轨迹和梯度累积期间固定，optimizer step 在批次边界执行 |
| 在线状态 | 每条独立轨迹从空状态开始，并行样本隔离 |

```text
for each independent trajectory s:
    M[s, 0] = empty_delta_state
    M[s, 1] = Write_phi(M[s, 0], B[s, 1])
    for t = 2 ... T[s]:
        loss_sum += CausalLossSum_theta(prefix128 + B[s, t], M[s, t-1])
        loss_tokens += number_of_valid_target_tokens(B[s, t])
        if t < T[s]:
            M[s, t] = Write_phi(M[s, t-1], B[s, t])
L = loss_sum / loss_tokens
backward(L)
optimizer.step() at the configured accumulation boundary
```

一块内 reader 始终读取预测开始前的状态，使用 causal mask 和正确的 labels shift。只给当前新块的合法目标打标签；当前块首 token 由历史 prefix 最后位置预测。块内 token 在 causal forward 中并行计算，仍须避免提前把整块聚合结果写进状态。

**时序不变量：预测 `B_t` 只读 `M_(t-1)`；`B_t` 的全文池化只能在本块预测完成后参与写入；下一块才读取 `M_t`。**块内事件必须全部用同一个旧快照生成 gate，再按行求 mean 并一次更新；不能把事件依次写到新状态，再让后续事件读取已经改变的行，否则会悄悄把当前 D1 改成另一种算法。修改未来块只能影响未来状态及其后续预测，不得改变过去块的 logits。

Writer 只写终点位于新块的事件。跨边界 n-gram 使用最多 `max_order-1` 个旧 token 寻址，旧 token 不产生新写入；reader 的 128-token prefix 不重复写。Writer 的证据编码不读在线增量状态，保持当前 D1 定义。末块写入没有后续训练目标时省略。

冻结 reader 权重不等于把预测 forward 包在 `no_grad` 中：预测损失仍须通过 reader 对增量状态求导，再传回早期 writer。独立的冻结证据编码可不保留基模型参数梯度；writer 对这些特征的变换以及 D1 对旧状态的依赖必须保留梯度。

### 4.2 损失归一化与梯度累积

无论轨迹为 8 块还是 16 块，训练目标始终是**同一 optimizer step 内有效目标 token 的损失总和除以总 token 数**，不把每条轨迹或每个 microbatch 的均值再等权平均。实现可先知道整个累积组的 token 数，逐 microbatch 对 `loss_sum / group_loss_tokens` 反传；参数在组内保持不变。

显存不足时先降低独立轨迹的并行批量，使用 `raw.gradient_accumulation_steps` 保持优化批量。当前已实现可选的整段评分 non-reentrant checkpoint 包装，显式传入状态 tensor，并在每次初算/重算内部注册和移除读出 hook；CPU 测试与新 GPU 的确定性配置下，开关重算的 loss 与 writer 梯度均逐位一致，**不启用 Hugging Face 内置 layer checkpointing**。GPU 验收配置为 BF16 reader、FP32 writer、512-token 块、128-token prefix、8/16 块轨迹；混合批量与串行路径另按容差比较，不称为逐位一致。禁止为通过显存限制悄悄 detach、改成 truncated BPTT，或每块更新一次 writer。以后若引入截断，应作为单独训练方法报告梯度跨度。

重算函数不能从可变的“当前 memory”变量或全局 hook 获取状态。即使反传时在线流已推进到 `M_t`，此前评分的重算也只能使用它原本绑定的 `M_(t-1)`；该快照的 tensor 仍须连接早期 writer 的计算图。是否使用整段评分函数的 checkpoint 包装或其他方案，由这一不变量和 loss/梯度 parity 验证决定。

历史长度与有梯度跨度分别记录。无梯度长历史预热需要标记生成状态的 writer 版本；训练 16 块不能解释为已验证数千次更新。

### 4.3 Token 账本

分别保存 `loss_tokens / unique_raw_tokens / writer_new_tokens / encoder_input_tokens / reader_input_tokens`；评测与重算成本另记。首块、prefix、padding 不计入目标预算。8 块轨迹最多 3,584 个目标 token，16 块最多 7,680 个。

当前实现的输入 token 计数表示逻辑前向消费，**尚不累计 checkpoint 重放的 token**；完整 step 计时已包含重算。因此不能把这些逻辑计数直接当作所有实际计算的 token 总量，生产成本报告须补上重算统计或使用实测时间。

按目标 token 各半混合，100M 目标 token 对应理想原文读取量约为 `50M×8/7 + 50M×16/15 = 110.48M`；5M 小试约为 5.52M。这里不含清洗损耗、留出集、未用尾部和窗口重复。不同对比可复用训练原文，计算消费仍分别计费。

## 5 为什么像 online RNN，以及本轮怎样并行

### 5.1 递归发生在记忆状态，不是每块做一次 SGD

一条流的状态满足 `M_t = Write_phi(M_(t-1), B_t)`，下一块 loss 读取前一状态，因此训练结构类似按块展开的 RNN。训练时通过整个 8/16 块轨迹反传；部署时 writer 固定，只有 `M` 前向更新。这与部署时每块对模型参数执行 SGD 是不同的操作。

当前 D1 在同一块的同一物理行上，先对所有事件求均值：

```text
A[t,j] = mean_a(beta[a] * target[a])
w[t,j] = mean_a(beta[a])
delta[t,j] = delta[t-1,j] + A[t,j] - w[t,j] * delta[t-1,j]
```

分母是该行全部有效事件数，含零 gate 事件，不是 gate 之和；一块内所有事件读取同一个旧状态快照。触及的行计数每块加一，未触及行不变。D1 gate 还依赖旧行、`target-old_row` 和 `log(1+count)`，这些定义本轮保持不变。

### 5.2 DeltaNet 的启发与适用边界

[DeltaNet 的并行训练论文](https://arxiv.org/abs/2406.06484)研究了 delta-rule 递归难以沿序列维高效训练的问题，利用结构化矩阵乘积形成适合硬件的算法。[Gated DeltaNet](https://arxiv.org/abs/2412.06464)进一步结合遗忘门控与 delta 更新，并给出并行训练方法。这与本项目遇到的“状态递归限制吞吐”有关，但不意味着原论文算子可以直接替换当前 D1。

本项目的关键区别是 **D1 gate 由旧状态非线性决定**。虽然上式看起来像 `delta_t = a_t * delta_(t-1) + b_t`，`a_t`、`b_t` 本身依赖 `delta_(t-1)`，不能在计算旧状态之前全部独立求出。因此，当前 D1 不能直接套用固定系数 affine scan；这不是证明所有精确并行算法都不存在。

现有 D0 的 gate 只依赖证据，系数可先算，才有直接的 affine scan 路线：两个更新 `(a1,b1)`、`(a2,b2)` 依次复合为 `(a2*a1, a2*b1+b2)`，未触及行对应恒等变换。但把 D1 改成 D0 是算法变更，需要独立训练和效果比较，不能作为“无损加速”混入本轮结果。任意非线性函数在形式上可以复合，也不等于存在固定大小、低开销的 scan 表示。

### 5.3 本轮实施顺序

1. **先并行独立文本流。**同一设备在一个时间步处理多条轨迹，每个样本拥有独立状态；物理行号相同也不得串写。批量 loss 与逐条执行在数值误差范围内一致。
2. **保留块间顺序。**每条流按块更新，块内 reader 仍做 causal token 并行；写事件按行聚合可向量化。先得到可审计的完整 BPTT 基线。
3. **再优化不依赖旧状态的工作。**候选包括证据编码、地址计算、目标特征和数据加载；是否预计算由内存与 I/O 实测决定，不默认缓存全量高维 hidden。
4. **随后验证真实语料吞吐并评估多卡。**整段评分重算已通过 CPU 与 GPU 确定性配置下的快照 loss/梯度 parity，并测得合成输入上的短时显存与吞吐；正式语料和完整训练的成本仍须实测。DDP 必须保证独立流归属、分片数据和按全局有效 token 数归一化，目前尚未实现。
5. **最后再做时间并行研究。**只有推导出保留 D1 语义且成本合理的算子，并通过状态、loss 和梯度 parity 后，才接到主线；否则下一篇优化，不阻塞当前实验。

## 6 第一轮评测与推进门槛

### 6.1 固定目标上的四种条件

| 条件 | 定义 |
|---|---|
| `L_empty` | 只清空在线增量状态；reader、原生基表、writer、局部上下文和评分位置保持相同 |
| `L_self` | 状态仅由当前文档评分位置之前的历史写成 |
| `L_donor` | 用同域、同语言、历史长度相近的其他文档写成；至少两个固定 donor，排除同家族和近重复 |
| `L_context` | 清空在线增量状态，保留模型和基表，把同一段真实历史放回上下文，在相同目标 token 上评分 |

Donor 使用同一 writer checkpoint 和写入步数，固定匹配清单；先对两个 donor 的目标损失等权平均，再计算配对差，另报各 donor 结果。记录实际地址命中、已写行数、计数与状态范数。另做保持地址和计数、仅置换 value 的诊断；明确其为人工状态，不称为自然 donor。

主量为 `Δ_empty = L_empty - L_self`，文档相关量为 `Δ_donor = L_donor - L_self`。Self 只优于受干扰的 donor、却不优于 empty，不表示正收益；self 和 donor 同样优于 empty，可以表示通用适应，但不足以支持当前文档内容记忆的主张。Self 优于 donor 本身仍不能单独证明记住了全新信息。

`L_context` 是参照，不是理论上限。辅助比例 `R = Δ_empty / (L_empty-L_context)` 用分层汇总损失计算；仅在分母大于预设 `0.01 nat/token` 且文档分组 CI 下界为正时报告，否则为 `N/A`，不截断到 `[0,1]`。绝对损失差始终报告。

### 6.2 统计单位与监控

开发集按文档家族做配对 bootstrap；每次抽样后按有效目标 token 加权汇总，不能把 token 或同书窗口当作独立样本。固定开发集、donor 清单、bootstrap seed、指标和检查点；保留集不参与配比或 checkpoint 选择。

监控来源、此前已写块数、128-token prefix 以外的复现距离、实际地址命中、`β`、状态改变量与范数、有效写入行数、梯度范数，以及后续损失到早期写入的梯度。地址碰撞和字面复现不作为语义记忆真值。

After-write 4/8/16 的收益曲线严格在**下一块**评分；写 16 块后使用第 17 块，不能评分已经写入的第 16 块。

### 6.3 P1 只检验可训练性

因果梯度、数值与状态隔离正常；固定开发集 `Δ_empty ≥ 0.001 nat/token`，且 95% 文档分组配对 CI 下界大于零。主 run 在 2.5M、5M 两次预定检查都通过，第二 seed 在 5M 复核方向。留出针对 writer 训练，不要求基模型从未见过文本。

P1 不以 donor、随机绑定、新近新闻或 full-context 指标共同作为推进门槛。若信号弱，先定位梯度、读地址覆盖、状态破坏和原文可用收益；允许**一次最多 0.25M 目标 token**的短轨迹诊断，使用 2–4 块和更小块长，再回到 8/16 块主配方。块长、seed 和诊断目标事先记录；这 0.25M 计入 P1 主 run 的 10M 总封顶，不额外加预算，也不自动获得延长主 run 的资格。常规训练的 2.5M/5M 检查点不变，最终常规训练上限相应扣除诊断消费。

短轨迹有效而长轨迹无效，支持优先排查多次写入干扰和梯度跨度；不能直接当成长轨迹方案已通过。代码失败不等于自然语言必然失败，但也不无限换配方直到显著。

### 6.4 只有进入 P6 前增加两项最低证据

P3 原生 pilot 与 P4 小模型混合训练先产生正面信号；原生的因果正确性、吞吐、显存和 `Δ_empty` 也须成立。P3 的初始 2M 是诊断起点，不能在明显少于已观察到的小模型学习时长时直接宣布门槛失败；先按第 9 节预登记足够且有封顶的 pilot，再判断。在此基础上，增加：

1. **文档相关收益：**在预先固定的留出目标和 donor 清单上，`Δ_donor` 的 95% 文档家族配对 CI 下界大于零。不另加很高的效应量阈值。
2. **随机绑定正收益：**reader 与 writer 冻结后独立产生随机绑定，正确状态相对空状态和交换事实的错误状态，预设主指标都取得正收益。此项要求方向成立，CI 一并报告，不额外追加“其 CI 下界也须为正”的门槛。

两项在 0.6B 和原生小规模阶段都测；**P6 决策以拟扩大训练的原生 reader/writer 在原生 pilot 中的结果为准**，0.6B 结果不能代替原生证据。开发样本量不足时结论为未定，先补齐预定评测；不能靠反复选 checkpoint 制造通过结果。没有通过时停在已授权的小规模诊断范围，不直接开启 100M 原生 run。

## 7 前向记忆评测

训练学通用读写机制；评测分开回答以下四类问题。模型参数是压缩知识与统计先验，也可能保留部分具体文本；见过一段文本不等于能够恢复本次需要的所有细节。

| 能力 | 主要证据 | 结论范围 |
|---|---|---|
| 熟悉内容的细节恢复 | 公共原文上的 self/empty/donor/context | 本次状态对预测和内容利用的作用 |
| 新近事实 | 晚于已固定权重时间界限的新事件材料 | 在可核查的时间条件下吸收新事实 |
| 冲突修订 | 初值、更正与各自时间均保留 | 是否随后续证据更新，并减少旧值回答 |
| 随机绑定 | 训练后随机分配、交换值、状态干预 | 状态是否携带本次产生的具体绑定 |

### 随机绑定的固定小测试

先按源文档及所有变体整体留出；训练、开发、保留的映射与 seed 隔离。在 reader、writer 冻结后，给实体随机分配同一组候选值，优先实体间对调，控制词汇与长度分布。A/B 两个绑定版本在评分位置之前的**整个可见因果前缀**相同，包括 128-token prefix 和当前块已出现 token；答案只能从在线状态传入。

清空跨块 KV 和其他旁路。比较正确状态、交换事实状态、空/清除状态；要求预测随绑定改变。P6 使用的主量固定为**候选答案首个分歧 token 的 NLL**，A/B 两版本都以其对应正确 token 计分，按独立文档组汇总：

```text
Δ_bind_empty = NLL_empty - NLL_correct
Δ_bind_wrong = NLL_swapped_state - NLL_correct
P6 方向条件：Δ_bind_empty > 0 且 Δ_bind_wrong > 0
```

优先选首 token 即有分歧的值；若有共同前缀，只可使用对两候选相同、不给出绑定答案的公共部分，不先 teacher-force 正确值的特有开头。整值自由生成正确率和 A/B 反转一致率辅助报告。该小测试不替代全部语义记忆任务，也不把公共原文收益一概改称“新事实存储”。

### 新近材料与自然修订

为准确的模型 revision 记录权重指纹及其已存在的时间证据，再选择之后才成立或公布的具体事件。保存首次发布时间、版本时间、抓取时间、正文 hash、原 URL 和权利条件；“网页最近更新”不足以证明事实新颖，empty loss 也不是未见过证明。关闭搜索、检索和跨样本缓存等其他信息通路。

单篇新闻不足 4k 仍可用于独立前向评测；跨报道按事件和真实时间排序，转载家族整体留出，不同事件隔离状态。标题、摘要或局部前缀已给出答案的目标单独标记。来源许可和数量实际验收后再确定公开方式；不为主训练先建设庞大的新闻数据工程。

一致改名是补充诊断，不能自动证明结构陌生。代码按作用域改内部标识符并检查语法，避免外部 API、反射和不确定字符串；变体重新 tokenize，并报告地址覆盖与长度变化。

## 8 对话格式适配

从 P4 checkpoint 分出两支，各最多 1M 有损失目标 token：一支继续 raw text，一支按部署模板仅对 assistant 正文计 loss。两支使用相同部署 reader 和对话评测，记录实际输入、写入 token 与 GPU 时长；目标 token 相同不等于成本相同。

候选用 [OpenAssistant/oasst2](https://huggingface.co/datasets/OpenAssistant/oasst2)，revision `179dd21fc55192153d94adb0e0ce8f69e222bf75`，读取 `2023-11-05_oasst2_ready.messages.jsonl.gz` 或相应 ready trees。保留 Apache-2.0 许可和树 ID；逐条筛英语、`synthetic=false`、未删除、审核与质量合格节点。按 `parent_id` 重建根到叶路径，不拼兄弟回答；先按整树划分，再取至少两次 assistant 回复的路径，去重共享祖先消费。

此来源用于多轮格式适配，不假称已验证的跨会话新事实数据。预算为 `min(1M, 合格且去重的 assistant token 数)`，不足则两支等量缩小，不复制补数。

**本方案的默认对话部署只写 user 消息正文。**系统、模板控制 token 和 assistant 输出不进入在线写入；P5 的训练也按此 profile，仅用 assistant loss 学此前 user 内容应怎样写。Teacher forcing 允许在当前预测的因果前缀中看到此前正确 token，但不将整段 gold assistant 答案预先或事后写成默认记忆。原始文本主线的每块先预测、后写入协议保持不变。该 profile 为后续 P5/部署实施要求，本轮原文训练改动尚未修改聊天入口的写入默认值。

写入 assistant 输出是独立可选 profile：只能在回答生成完成后写入实际生成文本，检查错误被记住、后续自我强化及修订能力。若训练此选项，须包含当前模型实际生成的 on-policy rollouts 并单列其生成与训练预算，不能只用干净 gold 回答训练、再默认可承受部署生成错误。此分支使用 P5 已有预算，不自动加量。无论哪个 profile，回复先预测/生成，全文汇聚与写入在其完成后发生，不能提前写答案。

主要 checkpoint 做小规模换问法与颜色检查；对话支路还比较指令遵循、格式、拒答与恶意记忆干扰。未通过则保留 raw checkpoint。

## 9 原生诊断、阶段与预算

### 9.1 先检查原生表的可用容量

原生接口打通后、2M writer pilot 之前，先做少量**直接优化增量表 patch**的容量诊断：reader 和基表冻结，跳过 writer，仅对允许写入的 patch 求梯度。使用独立于训练的诊断文档后续块和随机绑定小样本，区分两条路径：

- **目标监督 oracle：**使用待预测块或绑定答案的标签直接优化 patch。这是给予额外信息的乐观诊断，不可部署，也不是任何 writer 的数学性能上界。结果只能说明该读出接口在这些约束下能否被找到的 patch 改善；不能冒充前向记忆收益。
- **仅证据梯度基线：**只用已到达证据的 loss 优化 patch，再预测后文；禁止借用后文标签。它符合在线信息边界，但不同于无在线梯度的 D1 writer，计算成本另报。

允许行仅从既往证据地址确定，不按目标答案新增行。各条件使用相同的 reader、读出、可写行集合、patch 范数和存储预算；比较 self、empty、donor/交换事实状态，逐项记录地址命中和状态规模。Oracle 的目标拟合结果与任何未参与其优化的后续查询结果分别报告。

先检查梯度是否连通、目标是否下降、尺度限制是否过紧，并在至少 3 个预设初始化/重启及预设学习率或步长下查看收敛。少数未收敛运行接近零，只表示诊断不足；若充分排查后仍近零，暂停 writer 扩展，重查地址容量与读出接口，**不据此证明任何 writer 都不可能有效**。

接口检查与此诊断共享原有 **0.1M 目标 token 访问上限**：建议接口调通最多 0.02M，容量诊断最多 0.08M。重复优化同一标签、不同重启和仅证据基线都重复计入消费，不按 unique token 隐去计算。全部验证评分另记评测成本；上限内不足以辨别时报告未定。原生接口目前尚未实现，此项为计划。

### 9.2 用小模型里程碑决定有界 pilot

P4 除 25M/50M/100M 主检查外，在 **1M、2M、5M、10M、20M、30M** 增加预定的轻量 donor 检查，固定同一配对开发集。记录 0.6B 的 `Δ_donor` 首次稳定为正所需 token：定义为连续两次预定检查的 95% CI 下界均大于零，里程碑 `m_0.6B` 取第二次检查的累计 token。额外评测费用入账，不事后挑检查时点。

原生 writer 先做 2M 初始诊断；在开始延长前，将 pilot 检查点与封顶写入 run budget。起始规则为 `P = max(2M, m_0.6B)`，**pilot 最高 30M**；阈值时刻只为安排预算提供依据，不保证不同 reader、tokenizer 和原生接口按 token 等比例收敛。P6 门槛在足够且预登记的原生 pilot 上评估，初始 2M 未通过不直接判定方法失败。

若 0.6B 尚无稳定里程碑，或里程碑超过 30M，现有预算不足以支持对应时长判断：报告未定，停止昂贵扩展，不能把“未给足学习时长”写成原生方法失效。原生 pilot 的延长从原定原生训练总额中分配，不另加预算，也不在没有上限的反复试验中寻找通过结果。

### 9.3 总预算

| 阶段 | 本阶段上限 | 推进条件 |
|---|---|---|
| P0 数据与协议 | 首批约 20M 代码原文；少量调通批次单列 | 分组、token 切块、寻址、因果 labels、早期写入梯度、恢复与批量隔离可复现 |
| P1 0.6B 代码小试 | 主 run 先 5M，通过后最多 10M；第二 seed 另 5M | 第 6.3 节可训练性与 `Δ_empty` 门槛；短轨迹诊断计入主 run 10M 封顶 |
| P2 Base 迁移诊断 | Base 另 5M；后训练版复用 P1 的 5M checkpoint | 在同一部署 reader 上完成兼容且公平的比较 |
| P3 原生提前验证 | 接口与容量诊断共不超过 0.1M；writer 初始 2M，预登记延长最多至 30M | 先检查容量与因果路径；pilot 时长参考 `m_0.6B`，再看原生留出收益、显存与吞吐 |
| P4 四源混合 | 约 200M 原文池；一个主 run 最多 100M 目标 token | 25M/50M/100M 检查，总体与逐源报告，不隐藏未通过来源 |
| P5 对话适配 | raw/对话两支各最多 1M | 相同部署 reader 上比较 assistant loss、自由生成与部署回归；不通过保留 raw checkpoint |
| P6 原生扩大训练 | 剩余额最多 `100M − 已消费全部原生 writer pilot token` | 充分且预登记的 P3/P4 有信号，原生 `Δ_empty` 与资源依据成立，并通过第 6.4 节 `Δ_donor` CI 和随机绑定方向门槛 |

P3 的接口检查在 P1 数据流调通后即可并行开展，不等 P4 完成；训练 pilot 在 P1 有信号后进行。P3 如果尚未启用重算或多卡，不以未测配置宣称这些能力已验证；目标原生部署确实需要它们时，须补齐对应验证后再扩大。

P4 使用已验证配方，从固定初始化重新开始四源主 run，使 100M 内 30/30/25/15 的来源比例明确。P3 与 P6 属于同一原生 writer 的连续训练预算；pilot 消费不因改名为扩大训练而归零。另行重启、对比或被放弃的原生 writer 优化也计入同一封顶，不隐去消费。

按封顶计：0.6B 的 P1/P2 共 **20M**，P4 **100M**，P5 **2M**，合计 **122M 目标 token**，另加显式登记的 P0 调试与评测成本。原生现在统一为 **0.1M 接口/容量诊断 + 100M 累计 writer 训练 = 100.1M**：初始与延长 pilot 均包含在 100M 中，P6 只用余额。此账本替换 v2 的 `0.1M + 2M + 100M = 102.1M`，没有增加硬预算。1B 原文池、中文、跨仓库序列和超长历史后置。

## 10 工程交付与验收

实施位置为[本仓库](../README.md)，本轮从 `948774d` 起在 `codex/raw-text-training` 推进。保留合成任务与既有 D1 行为，新增原始文本入口；本轮不改核心 D1 数学。

| 交付 | 本轮目标 | 当前状态 |
|---|---|---|
| [原始文本数据接口](../src/delta_ngram/data/raw_text.py) | 文档元数据、一次 tokenize、连续窗口、按目标 token 采样和恢复 | CPU 验证通过；内存 JSONL pilot，支持 train/validation；[有界代码构建脚本](../scripts/prepare_code_pilot.py) 先分组再切分，保留集另存，非全库去重生产管线 |
| [Batch memory state](../src/delta_ngram/batching.py) | 独立流隔离，按 batch 行读取和更新，保留快照计算图 | CPU 串行/批量及梯度验证通过 |
| [Raw LM trainer](../src/delta_ngram/raw_lm.py) 与 [smoke 配置](../configs/raw_smoke.yaml) | 先预测后写入、完整 BPTT、单设备批处理、梯度累积、恢复 | CPU 公开 CLI smoke 通过；GPU 独立进程恢复和 512-token 块验收通过；均为工程验收 |
| 回归与最小调通 | 串行/批量 parity、因果标签、首块早期梯度、跨边界事件、预算账本 | 本地 CPU 343 项通过；迁移基线在新 GPU 环境为 317 项通过、1 项历史 bitwise fixture 失败，断言未放宽 |
| 评测与数据生产 | 四条件配对、固定 donor、随机绑定、bootstrap CI、阶段 gate；四源采集清洗 | self/empty 的组级配对 CI 与单点 P1 检查已接通；固定样本/协议 hash 纳入恢复身份。整体 P1 仍要求两里程碑与第二 seed；donor、随机绑定和四源完整构建待实现，`p6_gate_evaluated=false` |
| 原生接口 | 原生 n-gram 基表、在线增量状态与梯度路径 | 尚未实现，P3 单列 |
| 整段评分重算 | non-reentrant 包装、显式快照 tensor、每次调用独立 hook 生命周期 | CPU 与 GPU 确定性设置下 loss/梯度逐位一致；HF layer checkpointing 关闭；合成输入短时性能已测 |
| DDP | 全局 token 归一化、分布式状态归属 | 尚未实现，按实测需要推进 |
| D1 时间并行算子 | 保持算法语义的推导、成本与梯度验证 | 不作为本轮前置交付 |

验收要求为：同一 batch 中 A 的写入不改变 B 的预测；未来块不影响此前评分；reader prefix 不重复写；跨边界寻址保留 token 身份；后续 loss 能传到首块 writer；8/16 混合和梯度累积的目标等价于按 token 汇总；旧合成路径及 D1 状态更新无回归。

本地 CPU 证据为 [raw LM 19 项测试](../tests/test_raw_lm.py)、[batch 30 项测试](../tests/test_batching.py)、[data 23 项测试](../tests/test_raw_text_data.py)与 246 项原有测试，共 318 项通过。混合 8/16 块的 CPU 测试使用小块长（8 token）；重算 loss/梯度逐位一致，混合 batch 对逐条计算在数值容差内一致。累积步数 1 和 2 的同配置运行，各自在 optimizer step 2 中断后恢复至相同的 128 目标 token 终点（累积 1 为 6 steps，累积 2 为 3 steps），与各自连续运行的 writer、Adam、sampler、计数和 RNG 逐位一致；按 token 加权的 microbatch 与不足整批的尾部也有覆盖。

启动准备在上述基线上增加 11 项 raw LM 评测/保存/恢复测试及 14 项代码数据构建测试，当前全套 **343 项通过**。代码数据脚本只在有界候选池内检测精确和近重复，不宣称识别全库所有 fork。P1 配置为 [p1_code_seed17.yaml](../configs/p1_code_seed17.yaml)：5M 名义目标 token，2.5M 与终点评测，128 个固定开发窗口、至少 64 个独立组；每 250k 目标 token 保存最近恢复点，评测里程碑另存不可覆盖的 writer 和训练快照。所有间隔在完整 optimizer 边界检查，不足一条完整轨迹的尾部显式留空。固定开发协议变更会拒绝恢复，包括缺少该身份字段的旧调通检查点。

### 新 GPU 环境的工程验收

设备报告为 RTX 4090、约 48GB 显存；此处记录该设备的报告值，不将其推广为所有 4090 的规格。新环境使用 `torch 2.14.1+cu132`，历史 fixture 环境为 `torch 2.8+cu129`。新环境全套测试为 **317 passed、1 failed**：失败为历史 bitwise fixture 的严格比较，保留原断言，不把跨版本不逐位一致改记成全部通过。

| 验收 | 配置与结果 | 范围 |
|---|---|---|
| 重算一致性 | BF16 reader、FP32 writer；512-token 块、128-token prefix；T8/T16 在确定性设置下开关 checkpoint 的 loss/梯度逐位一致 | 同环境、同配置的前向/反向实现 |
| 独立进程恢复 | 2 个 optimizer steps、15,360 loss tokens；恢复与连续运行的 writer、Adam、sampler、counters、Python/Torch/CUDA RNG 逐位一致 | 同环境、同配置的训练恢复 |
| 混合批量数值 | BF16 混合批量与比较路径通过容差检查，梯度相对 L2 差为 **2.9781%** | 通过容差，非 exact parity |
| 短时吞吐与显存 | B4/T16、checkpoint 开启；3 次短计时的中位数 **7,718 loss tokens/s**，allocated **12.43 GiB**、reserved **13.95 GiB** | 高多样性合成输入的短时工程测量 |

这些验收使用高多样性合成输入，证明对应配置可以执行并支持上述一致性结论；不表示正式代码/论文/书籍/网页语料已经训练有效，也不表示原生接口或 P1/P6 统计门槛已通过。

证据仅保留为本地归档文件名（未随仓库分发）：`machine_acceptance_summary.json`、`cpu_fixture_audit.json`、`gpu_gradient_diagnostic.json`、`gpu_mixed_diagnostic.json`、`gpu_acceptance_final_20261019/report.json`、`gpu_cli_fresh_process_20261019/acceptance.json`。仓库不包含原始报告或训练 checkpoint。

发布包也已验证：在仓库外构建 sdist，独立目录解压后通过打包的公开 CLI、配置和样例 JSONL 完成 CPU smoke，6 steps、128 loss tokens，退出码 0。这项验证确认入口与样例随包可运行，不扩展为 GPU 或原生模型结果。

训练恢复点只在已完成的 optimizer 边界保存，至少记录 writer、优化器、scheduler（若有）、全局步、预算计数、数据顺序与游标、随机数状态以及模型/数据/config 身份。稀疏状态快照可用于前向恢复或调试，不能独自冒充完整训练恢复。实现缺项必须在入口明确拒绝或在产物中标明，不静默近似。

## 11 产物与成本记录

```text
source_manifest.json         来源 revision、shard、许可、清洗去重版本和 hash
documents.parquet            文档/家族 ID、split、语言、来源、许可、长度和 hash
tokens/                      连续 uint32 token IDs 与文档 offsets
windows.parquet              原文位置、块数、来源、重复消费标记
reader_manifest.json         权重指纹、tokenizer、模板、地址与读写接口
sampling_manifest.json       比例、seed、消费顺序和恢复位置
eval_pairs.parquet           self/donor/empty/context 匹配与评分位置
novelty_manifest.json        时间证据、事件组、权利条件、绑定映射与 seed
run_budget.json              支路预算、预定检查点、门槛、实际 token 与 GPU 时长
data_stats.json              长窗口产量、地址覆盖、家族数与去重统计
```

200M uint32 token IDs 约 0.8GB，另加原文与元数据；不默认缓存全量高维 hidden。以上为生产数据产物的目标格式，首版调通接口可使用可追溯的小型本地样本，不把接口完成写成全量语料已准备。

时间报价最终依据真实语料 pilot 的完整 optimizer step：在目标 GPU 上，预热后分别测 8/16 块，覆盖证据编码、读取、写入、反传、优化器、数据加载及实际使用的重算/通信，报告稳定 loss tokens/s、峰值显存、GPU 数与评测时长。目前已有第 10 节合成输入上的 7,718 loss tokens/s 短时中位数，不能直接作为正式语料整场实验的时长承诺。按 `目标 token / 真实语料实测 loss tokens/s` 估算训练，再加数据准备、保存与评测；原生 MoE 另测，更换硬件或配置重新测量。

**执行结论：先以保留 D1 的单设备、多流、块间顺序完整 BPTT 建立原始文本训练基线；按既定预算验证收益，再决定原生扩展。并行训练研究有清楚的接口和比较基线，但不阻塞当前论文。**
