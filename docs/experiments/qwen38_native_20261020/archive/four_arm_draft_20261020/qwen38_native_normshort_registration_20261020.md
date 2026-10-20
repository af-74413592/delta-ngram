# Qwen3.8 writer 入口归一化机制短测登记草案

2026-10-20。**DRAFT_NOT_FROZEN；尚未执行本短测的优化。**机器可读合同为[同名 JSON](qwen38_native_normshort_registration_20261020.json)。真实无标签 H/T⁰ 观测已完成，共同 ρ 与两组 scale 已固定；leaf-Δ TP 有限差分两轮均未取得通过回执，禁止 optimizer 更新。

## 1. 问题与四组设置

在固定训练家族上比较 writer 入口 `none` 与无参数 RMSNorm，观察初始尺度匹配后短程学习和数值稳定性。四组为 `α=1.0/0.99 × none/rmsnorm`。主干、原生 T⁰、原生 reader 和其 norm 均冻结。

RMSNorm 只作用于 base-only、final HC mixer 后实际 LM-head 输入的每个 2560 维 H：先 detach、转 FP32，再计算 `H / sqrt(mean(H²) + 1e-6)`。每 token 独立，沿最后一维，无可训练参数；先处理 H 再计算 writer 的 right-window 与 pooled features。`none` 同样使用 detached FP32 H。

Writer 沿现有原生定义：16 头、行宽 160、隐层 256、右窗 4、D1。四组载入完全相同的 seed-17 初始参数，各自独立新建优化器。两 norm 条件各有一组经真实观测校准的 16 个固定 scale；同一 norm 的两个 α 组共用 scale。

优化器明确登记为 AdamW：`lr=0.0003, weight_decay=0, betas=(0.9,0.999), eps=1e-8, amsgrad=false`，无 LR scheduler，L2 梯度裁剪 1.0，非有限值即停。前三个核心数值及 seed 继承 M1；betas/eps 显式固定其既有默认值。草案将 `foreach/fused/capturable/differentiable` 设为 false，实际 runner 必须一致或在首个更新前修订登记。

每组 8 个 optimizer steps，每步按相同顺序处理 12 个样本、每样本 128 targets。整个累积组的 loss 分母为 **1,536**；先累积完整梯度，再裁剪、step 一次。真实 TP 梯度缩放必须有有限差分回执，不能凭单卡经验补乘或除。

## 2. 固定 24 个训练家族

`data/calibration24` 的 calibration-only 派生已保存 `ready_for_calibration=true` 回执，每来源 4 个 family。下表及 JSON 已绑定取回的真实 selection；实际远端绝对路径仍由 runner 绑定。六个来源按 `books, code, papers, web, news, news_event` 排序。

每来源按已保存的 `selection_hash, group_id` 排序：

- 前两个 family：训练及 H/T⁰/scale 校准，共 12 样本。
- 后两个 family：固定 train-side probe，共 12 样本；不参与梯度、scale 或 ρ 选择。
- 第一个 family：初始扰动检查子集，共 6 样本。

每 family 保留派生器原选定的文档或事件，不因长度或结果换样。验证 `COMPLETE.json → data_manifest.json → files` 的大小和 SHA 链、train split、24 个独立家族及两组不相交。缺失、长度不足或 token/hash 错误即停止；不得缩短目标、替换家族或静默丢弃。

最终 role/frame manifest 必须逐样本保存 `sample_id/source/group_id/document_or_event_id`，正文及完整文档/文章 token hash，cache offset/count、所用区间、输入 IDs、valid/write/loss mask、position IDs 的 hash，实际新 token/prefix/padding/target 数和文章边界。所有区间均为左闭右开。

已独立重算全部 24 个 `selection_hash=SHA256(canonical[17,source,group_id])` 和 `sample_id=calibration:SHA256(canonical[17,source,group_id,id])`，按 hash 排序分组，不依赖 JSON 文件行序。完整 ID、family、正文/文章 hash、token hash 和角色保存在 JSON 的 `data.frozen_samples`；下表只用文档 ID 摘要便于查看，每格从左到右按来源内 rank 排列。

| 来源 | 训练及校准 rank 0 / 1 | 固定 probe rank 2 / 3 |
|---|---|---|
| books | `2707fce844dc` / `f3440f10e051` | `edf4a2b7faec` / `9470ce099e91` |
| code | `d098bb57690b` / `b13581af6def` | `380a8d76afb7` / `06b919f06e3d` |
| papers | `b60e4f621e86` / `365a89585e6d` | `becf4f18d0e5` / `fc8d4d6a2a61` |
| web | `52c0e0c0167c` / `0ba780034e73` | `ecafc515c11b` / `c820e7801f8d` |
| news | `c0e34f4d372e` / `91e028e670fc` | `22becee3a7fa` / `2ca830f2add4` |
| news_event | `7f85a7883c4e` / `5b1a2e8785f8` | `9d58c933a5df` / `4d5cf6633efe` |

已核对的本地文件 SHA256：

- selection：`2290d5741531c6cf7ada6f4867be9abdf549b3b7fd56c05f230fe56917328a3e`
- 派生 manifest：`de184d29c53b80bd840b56d548f131550fae577a9999d7f3d4bb259f97f3ed1a`
- 派生 seal：`65352a4a0c796daaa85b22820cc62a5c72ab0451164754f728823fa2131a5b07`
- 24 家族角色绑定 canonical hash：`e9266b93767f47cd732b1d5b1bf31268b6ec4d85cfae62a3070d021e3b226842`

本地目前只有 seal、manifest、selection，尚未取得 token cache 和 indices；JSON 中缓存 SHA 是已验证 manifest 的期望值，不能视为本地实际 bytes 校验。真实 runner 在生成 frames 前须核对所有 payload，exact frame hashes 仍待填。

## 3. 精确写入与查询合同

### 3.1 普通文档

写入原生 token `[0,512)`，只有一个新证据块。查询 token `[512,640)`，输入为同文档 prefix `[384,512)` 加 128 个 targets，总长 256。查询标签已经左移到预测位置：input 位置 `[127,255)` 的 labels 分别为这 128 个 targets，余处为 `-100`。

需要至少 640 个原生 token。短测只使用这些固定区间；它不以派生器普通样本的完整 8/16-block window 冒充本次短测长度。

### 3.2 新闻事件

按冻结时间顺序，**完整写入第一篇文章**，每块最多 512 个新 token，包括末尾不足 512 的块；不能把新闻截成一半作为 history。之后只查询第二篇文章前 128 个 token，输入为本地 `BOS=248044` 加这 128 targets，共 129 个真实 token。

查询 labels 位于预测位置 `[0,128)`；本地 BOS 预测第二篇的第一个 token。第二篇及后续文章不写入本次 memory。该事件必须至少两篇，第二篇至少 128 token。

第一篇的首个写块自然开始，无外部 prefix；后续写块带同篇之前最多 128 token 的 base-only encoder prefix。因此 native PLE 实际调用同时给出正确的跨块地址和 base H。文章切换时重置局部 prefix/n-gram 上下文，保留这个 event 的 owner memory。

### 3.3 Mask、边界与状态

- `encoder_valid_mask` 包含真实 prefix 和新内容，排除右 padding。查询的 adapter `valid_mask` 使用它；evidence 的 adapter capture `valid_mask` 使用 `write_mask`，使 physical rows 与 T⁰ RMS 只统计新 token。evidence 模式仍原样返回所有 token 的 base embedding；native hasher 实际看到的完整 input IDs 不变。
- `write_mask` 只含当前新证据 token；prefix 不写，不进入 pool/right，也不进入 H、T⁰、target 的统计。
- 写入调用为 `mode=evidence, state=None`，没有语言 labels/loss；最后 H 的 TP sequence-parallel shards 按 token 顺序拼接，不重复计数。
- 查询只读固定快照，128 个 shifted targets 以外的 labels 全为 `-100`，不会写 query tokens。
- 为 TP=2 把序列长度右补到 2 的倍数，padding ID 为 248044；padding 不读、不写、不计 loss，native causal mask 阻止真实 token 看未来右 padding。129-token event query 因而物理长 130。
- 每样本、每训练 step 及每次 probe 从独立空 memory 开始；同一样本 history 多块之间保留完整 writer 计算图。α 只在有新写入 token 的块推进一次，prefix/padding/query 不推进时钟。

普通样本只有一个从空状态开始的写块，因此该短测的普通来源不能单独区分衰减效应；多块事件 history 才包含跨块衰减。完整正式 pilot 仍沿原 8/16-block 和完整事件 trajectory，本短测不修改其协议。

## 4. 先无标签观测，再固定一个 ρ

对训练侧 12 个样本的全部登记 history frames，零更新运行真实 base H/native IDs/反量化 T⁰ 观测。H forward 不求语言 loss，不计目标访问；另记所有新 token、prefix、padding、重复 forward 和 GPU 时间。

按同一 seed-17 writer、unit target scale，分别计算 `none/rmsnorm` 的 raw targets。每头均按实际新 token 写事件的平方和/元素数聚合，重复地址按事件频率计数。实际下载权重及当前 host T⁰ 为 BF16，统计其有效浮点值；只有明确登记的量化 backend 才需要反量化 scale。不能假设当前权重有 FP8 `weight_scale`，也不能以全表均匀 RMS、头均值或块均值替代。

具体地，`T0_RMS_h=sqrt(sum_block(n_new×block_RMS_h²)/sum_block(n_new))`；每事件的 160 维 raw target 用相同 token 权重。长事件全文会比 512-token 普通 history 贡献更多观测，这里不再按来源或 family 等权重归一化。TP 重复副本不增加分母。

实测已完成。在任何候选语言评分之前，根代理已固定**唯一共同 ρ=0.01**及两组 scale：

`s[mode,h] = ρ × RMS(T⁰_h) / RMS(raw_target[mode,h])`

零/非有限分母直接停止。两组各 16 个 scale 已写入 JSON 的 `scale_calibration.target_scales_by_mode`，由真实回执按 FP64 公式计算后一次转为 FP32。`none` 范围约 0.004621–0.004898，`rmsnorm` 约 0.021256–0.022531；同一 norm 两 α 共用且不随优化再校准。这是初始 target RMS 为 T⁰ 的 1%，不保证 D1 state 或 reader loss 扰动为 1%。

真实观测包含 12 个训练家族、18 个 history frames、8,403 个新 token：前五来源各 1,024，news_event 3,283；两事件首篇分别 2,163/1,120 token，完整写入分别 5/3 块。H RMS 为 3.729683、median 3.757632、p05/p95 为 2.477737/4.580167；T⁰ 各头 RMS 为约 0.008008–0.008489。原始 target RMS 在 none 约 0.01733、rmsnorm 约 0.003768，β 接近初始 0.05；这些观测尚不说明哪种 norm 学习更好。

观测 `COMPLETE.json` SHA 为 `0ff5568f3b4bc3db213cb19347a2e4807c375956293ff0cfd6bb2e2bfb1bf30f`。独立 canonical scale JSON 的 SHA 为 `849eb3abc68592fc24c150b684254c0cead1aa61a9e4104844fa92384d1c32d9`，路径见机器合同；它明确保留初扰与梯度诊断尚未通过的状态。

随后只在每来源第一个训练 family 上做初始扰动：6×128 targets 的 empty 一次，四组 self 各一次，共 **3,840** 次目标 token 使用。逐组计算 `Δ_empty=(L_empty−L_self)/128`，要求总体及每来源 `abs(Δ_empty)<0.001 nat/token`。此处每来源恰一个样本，须保留逐样本 loss。候选失败先停止诊断，不自动尝试第二个 ρ，不动 probe 家族。

## 5. 硬预算与记录

| 项目 | 计算 | 目标 token 使用 |
|---|---:|---:|
| 四组优化 | 4×12×128×8 | 49,152 |
| 固定 probe 的 self，step 0/4/8 | 4×12×128×3 | 18,432 |
| 固定 probe 的 empty 一次 | 12×128 | 1,536 |
| 初始扰动 self | 4×6×128 | 3,072 |
| 初始扰动 empty 一次 | 6×128 | 768 |
| **预定合计** | | **72,960** |
| 原始诊断预留 | 75,000−72,960 | **2,040** |

probe 合计 19,968。2,040 是尚未计入梯度诊断时的原始预留，不是当前剩余。现在登记如下：

| 预留内项目 | 状态 | 目标 token |
|---|---|---:|
| 首轮 leaf-Δ TP FD | 已消耗，结果 inconclusive | 220 |
| 同 grad-mode 的 TP FD 修复 | 已消耗；baseline span=0，梯度缩放检查仍 inconclusive | 220 |
| 完整 writer BPTT FD | 固定一次，11×128 | 1,408 |
| **总计划、已消费与剩余预留上限** | 72,960+220+220+1,408 | **74,808** |
| **未分配余额** | 75,000−74,808 | **192** |

完整第二个 ρ 候选仅四组 self 就要 3,072，无法放入余额；余额本身也不授权自动新增调用。当前不启动 paired 优化。用户随后明确追加的同批 grad/no-grad logits+loss audit 由根代理在全局 100k 剩余额度中单列 interface audit，不占用这里的 192，也不能重分类此前已计入 norm 的两次失败 FD。

所有重复评分、候选、失败/中断调用和有限差分的 targets 均据实际调用计数，不能按 unique token 去重；分母除一次不改变访问次数。empty 只有在模型、输入、mask、运行时身份完全一致时才复用。无法确定中断消耗时，保守记入全部已派发 targets。

49,152 优化 targets 同时计入本次 ≤75,000、既有诊断 ≤100,000 和所有原生 writer 累计 ≤100M 账本。其余评分计诊断账本；无标签 H 观察另记输入成本。启动前补齐此前已消费值，并预留全部剩余计划；任何调用预计超出任一上限都不得派发，重启不归零。额外真实 TP/BPTT 验证如使用语言 targets，也必须纳入对应账本。

## 6. 固定观测与解释

固定 train-side probe 只在 optimizer step **0、4、8** 运行。每次从空 memory 用对应 writer checkpoint 重建全部 history，所有组目标完全相同。empty 一次保存并复用，不跨 runtime/config 变更。

主量为每家族 `gain = L_self(step0) − L_self(step8)`；分别在 α=1 和 α=.99 内报告配对 `gain_rmsnorm − gain_none`。同时报告每来源及总体的 step0/4/8 loss、Δ_empty、训练 loss、裁剪前梯度范数、clip、非有限值、门控分布、target/Δ/T⁰ RMS、参数变化与实际耗时。β 饱和份额固定统计 `<0.01` 和 `>0.99`。

描述性区间用 seed 104729 的 10,000 次配对 family bootstrap：每来源各 2 个 probe family 内有放回重采样，组间/时点保持配对，给出 95% percentile 区间；只在 CPU 上重算已有结果，不增加模型评分。12 个固定训练家族不足以证明独立评测集效果，不能把拟合改善外推为 DEV64/FINAL 收益。

**结果出现前的默认偏好已由用户明确为无参数 RMSNorm**：对齐小模型 post-final-norm 的输入条件，代价低且不增加训练参数。四臂短测保留用于检查副作用；这不是短测已胜出或验收通过。`log(RMS)` 幅度特征本轮不加入，避免同时改变输入信息设计。

此稿不自动宣布胜者。根代理按默认偏好及完整诊断登记后续主分支共用的 norm；若两 α 的比较冲突、差异不清或数值失败，报告未定或失败，不临时延长、重选样本、扩 ρ 网格或用 DEV64/FINAL 决定。

## 7. 条件冻结与单进程验收

根代理已授权单进程完成固定诊断到执行注册的转换，无需在已授权的通过路径上重复请求批准。授权不跳过以下状态：

1. `DRAFT_NOT_FROZEN`：字段未齐，只能准备；本文件当前为此状态。
2. `CANDIDATE_FROZEN_CONDITIONAL`：样本、frames、代码、运行时、ρ、scale、数值设置与预算全部固定，且绑定**已接受的真实 leaf-Δ TP policy**。`conditional_execution_authorized=true`、`diagnostic_execution_allowed=true`，但 `optimizer_execution_allowed=false`、`execution_allowed=false`。只能执行已登记初扰与 writer FD。
3. `FROZEN_EXECUTABLE`：两个固定诊断均通过，保存其 receipt hash 和父候选 hash，先独占写入不可变最终注册，再将两个 optimizer/execution 标志设为 true 并开始 8-step 短测。
4. `FAILED_DIAGNOSTIC`：保存失败/未定与消费记录，所有 execution 标志为 false；不自动调参数、步长、阈值或换样再试。

最终文件的字节 SHA 存在外部回执或 sidecar，不能在同一文件内递归填自己的字节 SHA。

### 7.1 候选及执行前清单

1. 校验 24 家族派生 seal、全文件 SHA、原始 split/model/tokenizer 身份；写入实际绝对路径与 24 个角色/frames。
2. 保存真实 H/T⁰/raw-target 观测与固定的单个 ρ、两组 scale；完成 6-family 初始扰动回执。
3. 先绑定 observation/修复产生的已接受 leaf-Δ TP policy；再按下节执行 writer 全 history BPTT 检查。reader→Δ preflight 通过并不自动验收 writer 参数梯度和 TP reduction。
4. 写入实际 runtime、代码、writer 初始参数、优化器 flags、预算父账本及各回执 SHA。
5. 将补齐的候选 JSON 以新不可覆盖文件保存；固定诊断全部通过后再写最终执行注册，保留 UTC 时间及完整 hash 链。任何未通过状态都保持 optimizer 禁止；本草案不能作为可运行配置。

已有真实 reader preflight 的 `COMPLETE.json` 已核 SHA：`d77b318d66e22e464c60ea67f259e3569b6498113f7a590edba7244d2d238001`。它记录 empty logits 逐位一致、两个 PLE TP rank 的非零 Δ 梯度，以及 TP2/PP2/EP2/ETP1/CP1/DP1；其自身明确未验收 writer BPTT、writer TP reduction 或 resume。本地只把这部分证据绑定到草案，不扩大其通过范围。

该已完成 preflight 为 3 次各 11-target 的 forward，计 **33**；两轮 norm 所附 TP FD 各 20×11=**220**，已知全局消费为 **473**，其中 **440** 同时占 norm 75k。第二轮 receipt SHA 为 `90d4fb84fd7512b7b460eab93c2a597c2f3e7819b43bcb72fcc17cc5104f8ac9`。backward 的标签数属于对应 forward，不能重复加一次。启动候选前绑定根代理的权威总账本。

### 7.2 一次完整 writer BPTT 有限差分

固定 α=.99、none、训练 rank0 的 news_event `7f85a7883c4e6b57134f9b02af9d93523c24702498befcfe7c7f7247a936641b`。使用已观测首篇全部 2,163 token、5 个 writer blocks，查询第二篇前 128 targets。不可截断为两个块；若实际不足两块则停止而不替换。

首次完整 BPTT 取得全部可训练 writer 参数梯度，按固定参数名顺序拼接，在 TP0 归一化为单位 L2 方向并广播；无 optimizer 更新。方向构造本身已预登记，不能换方向追求通过。对初始全部可训练参数的 L2 范数，固定相对步长 `{0.005,0.01,0.02,0.04}`，每档做正负中心差分。一次 analytic、两次相同中心的 forward、八次差分，共 11×128=**1,408** 目标访问。

所有 query 包括 empty、初扰、probe 和 FD 均使用与训练相同的 grad-enabled 原生数值路径，必要时创建 requires-grad leaf；仅是否 backward 不同。evidence H 仍为 base-only no-grad。analytic 与前置中心 loss 必须完全相等，否则最多两次 writer FD 调用后停止。

固定 `rtol=.20, atol=.001`。噪声下界为三个中心 mean loss 的 `max(span, max_abs×2e-7, 1e-7)`；每档要求正负 loss 差的绝对值大于 20 倍噪声，且中心导数绝对值大于 `10×atol`。按 h 递增取首对满足 `abs(D_right−D_left)≤atol+rtol×abs(D_left)` 的相邻合格档；选档时不看 autograd。随后两档都须满足 `abs(dot(g,v)−D)≤atol+rtol×abs(D)`。没有合格档或不符即未定/失败，不放宽阈值。

诊断后恢复并校验初始 writer 参数 hash，FD 状态不进入正式四组。单方向、单条件检查只验收这条共享计算链，不等于四组的穷尽梯度证明或收益证据。

本次只新增本 Markdown 与 JSON；不启动 GPU、修改既有 M1 路径或打开 DEV64/FINAL。
