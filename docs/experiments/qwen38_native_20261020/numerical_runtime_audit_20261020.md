# Qwen3.8 数值运行路径审计

2026-10-20。结论：固定原生 MoE top-k 排序后，本次 books 样本的 grad/no-grad 差异消失；split PLE 的 FP32 投影支路保留了 Δ 信号，但最终 BF16 转换仍产生明显量化误差。以下都是优化前诊断，optimizer 更新数为 **0**。

## 1. 范围与消费

模型为 `Qwen/Qwen3.8-Flash-Next@de4b8e4d43b917e7706784d8bb445c9af86a3540`，TP2×PP2，reader eval、权重冻结、`PLE_FUSED_KERNEL=0`。修复审计在同一次模型加载中使用固定 books 家族、256 个有效输入位置、128 个 loss targets，比较初始 writer 和相同行的零 Δ；每次比较完整 248,320 维词表 logits。

**H 和 target_scale 仍来自历史 compiled 路径，仅用于诊断。** 本结果没有完成最终训练路径的 H/scale 校准，也没有证明后续训练效果。最终路径需按既定 ρ=.01、相同训练家族重新生成 base-only H 和每头 scale，两 α 共用。

本次 22 次查询共 **2,816** target accesses，其中两次包含实际 reader backward；重复、compiled 对照和 backward 均已计数。全局诊断累计为 **9,444 = 6,628 + 2,816**。快照和 PLE 局部 observer 重算增加 CE targets 为 0。

## 2. A：定位与修复 grad/no-grad 分叉

原 compiled 路径 books 的 `loss(no_grad)-loss(grad)` 为初始 writer **+0.01718354**、零 Δ **+0.04118180 nat/token**。仅展开两个 HC compiled helper 后，仍分别为 **+0.00123596**、**−0.00342488**，最大 logit 差为 2.53125、3.1875。

两个第一阶段 TP rank 的逐模块记录都在第 43 个事件首次分叉：第一层 `mlp.router`。该调用输入相同，输出 FP32 路由概率不同，bool 专家选择 mask 相同。安装版本 `moe_utils.py` 的非分组 top-k 使用：

```python
torch.topk(scores, k=topk, dim=1, sorted=torch.is_grad_enabled())
```

后续在选出的 scores 上归一化，因此顺序会影响浮点归约。修复通过现有 `router_replay.get_replay_topk` 回调固定 `sorted=True`；回调无历史重放，保留训练排序，不修改全局 torch 函数或路由得分公式。每个 PP shard 的 24 个 router 在全部 22 次查询中各调用一次，四个 rank 均确认恢复原绑定。

| PLE / Δ 状态 | eager grad NLL | eager no-grad − grad | compiled − eager NLL | compiled/eager top-1 改变 |
|---|---:|---:|---:|---:|
| native / 初始 writer | 2.56784153 | 0 | +0.01127625 | 9/128 |
| native / 零 Δ | 2.58673835 | 0 | −0.01961541 | 14/128 |
| split / 初始 writer | 2.61079979 | 0 | −0.03364706 | 8/128 |
| split / 零 Δ | 2.57073951 | 0 | +0.02011800 | 7/128 |

四组 eager grad/no-grad 和 grad 重复的 **logits 全部逐位相同**；compiled 自身的 grad/no-grad 也全部逐位相同。两组初始 writer 的实际 backward 所用前向，与对应 eager grad/no-grad 前向逐位相同。native eager grad NLL 与修复前完全相同，修复后的 no-grad 与其对齐，支持所定位排序分支是这次分叉的原因。

compiled 与 eager 仍是不同数值路径。表中 NLL 仅描述差异，不用于挑选更低 loss 的实现。本轮有界训练按固定的 **sorted router + eager HC + split PLE** 路径继续；self、empty、evidence、probe 共用该路径。

## 3. B：Δ 可见性与剩余量化

split PLE 保留 native BF16 T⁰ key/value 投影，单独以 IEEE FP32 计算 Δ 投影与相加，norm/gate/conv/residual 使用 FP32，最后转回原生 BF16。以下为真实初始 writer 状态、eager grad、TP rank0；rank1 独立记录得到相同结论。两份记录是复制观测，不能相加计样本。

| 信号边界 | 有效/请求能量 | 余弦相似度 | 相对 L2 误差 | 非零请求中可见比例 |
|---|---:|---:|---:|---:|
| native embedding 的 BF16(T⁰+Δ) | 0.359706 | 0.449677 | 0.905712 | 11.0001% |
| split key 投影相加 | 0.9999986 | 0.9999999791 | 0.0002043 | 99.9851% |
| split value 投影相加 | 1.0000004 | 0.9999999973 | 0.0000736 | 99.9908% |
| split 最终 BF16 输出转换 | 7.452022 | 0.363575 | 2.543034 | 15.2015% |

native embedding 中完全被吞没坐标所对应的请求 Δ 能量占 **78.1357%**。最终转换的参考信号是“FP32 PLE(Δ)−FP32 PLE(0)”，有效信号是对应两个 BF16 输出之差；能量放大到 7.45 倍不代表信号保留更好。其沿请求方向的投影比例为 0.992501，但低余弦和高相对误差说明有较大的舍入扰动。

split 初始 writer 对零 Δ 的完整 PLE 输出 RMS 差为 `4.60979e-6`，128 个 loss 位置为 `1.06086e-6`；两个 TP sequence shard 已拼接后统计。新旧路径的零 Δ **完整模型输出也不同**：split−native NLL 为 **−0.01599884**，最大 logit 差 4.51953125，top-1 改变 8/128。局部 observer 的 native-zero 比较只复用相同 base 投影重算 PLE body，不能替代这个完整模型比较。

## 4. 梯度证据的准确边界

- **既有 TP 结构证据接受 replica 倍率。** 实际 SP 算子的 FP64 线性 FD 在 h=.001/.002 为 12.81355932203354 / 12.813559322034429，两 TP 的独立解析值均为 12.8135593220339；SP 拼接 adjoint 和原 native lookup 手工 index-add 的最大误差均为 0，原 raw Δ 梯度 SHA 相同。这只接受该原路径的 TP 缩放，复制梯度不能再求和。
- **FP64 PLE 参考通过。** 使用真实 PLE 权重、输入和输出 cotangent；12 个阶段与 3 个整链方向均通过两个最小预登记相对步长 `3e-5,1e-5`，容差 `rtol=1e-5, atol=1e-8`。native/FP64 输出相对 L2 为 0.00429224、余弦 0.999990889；embedding VJP 相对 L2 为 0.00694621、余弦 0.999975930。
- 该参考快照中的 Δ 是结构探针合成值，不是本次校准 writer。参考 FD 自洽不能单独接受全部原生 GPU kernel，也不把此前完整 BF16 CE FD 的 inconclusive 改成通过。
- 本次 native/split raw reader cotangent 的 L2 分别为 2.27300954 / 2.37192607，两 TP 相同；回执明确 `tp_policy_applied=false`、未执行 writer BPTT。相同范数不等于逐位相同梯度。最终训练锚点仍需绑定所选运行路径，核对两 TP 梯度及 split dense-Δ→leaf 手工 adjoint，再完成真实固定 cotangent 的完整 writer 历史验证与已登记的端到端方向检查。

## 5. 可追溯回执与标识修正

证据位于工作区相邻目录 `delta-ngram-runs/qwen38_native_launch_20261020/`。以下 SHA 均为实读文件或回执中的明确源码标识：

| 证据 | SHA-256 |
|---|---|
| `ab_repair_books_01_receipts/COMPLETE.json` | `8d18a0c8f8fc9bf554aa955d1e3a9d8ab4f3089831ae37249683dd4c419d6c16` |
| `mode_trace_books_01_receipts/COMPLETE.json` | `a3d88857c451ec0a24e2171e2aa087a46a0f638810d85bb82074bdf5caad19e5` |
| `tp_structure_01_receipts/tp_structure/COMPLETE.json` | `e8ddb1d127d9f8ae28e94512645e4cf419d9a070ecb73980d666b3671e11d0bb` |
| `ple_reference_01_receipts/COMPLETE.json` | `f34c36c20e274c35df32f40f77f66305a2185533880b1db89d8c6cbb1355f2b6` |
| 已执行 mode audit 源码 | `792c054be7c41cc13b4106245b3cd39ea8404c43c0d9cb2db04f968ddc483fc5` |
| sorted router helper | `90086da11a565dab1157a6f8719db7a127f6297d930beed1ef5e05d0d0d5faea` |
| 已执行 split precision 源码 | `6a2a1a01dd23649fc15c31b9782f6bc5d7a9c8cd23608f675830595939c69c80` |
| 安装的 MCore `moe_utils.py` | `629c48538f2d69a2e110ec5f5b5925d20ccf541a7e960adff43ebb61d4528713` |
| 已执行 FP64 参考源码 | `7c5f5f1f0e16564f8466682dc161d39be776fb1d070daff6dcef0208d8c83ba0` |

原始 AB 回执保持不变。其 case 顶层 `runtime_identity_sha256` 存的是该 case **最后一次查询**：初始状态最后为 eager backward，零状态最后为 compiled no-grad。因此零状态顶层 SHA 不是主 eager 结果的标识；主结果来自 `grad/no_grad/grad_repeat` 字段，native 快照也明确绑定 eager 标识。

| 实际 dispatch | runtime identity SHA-256 |
|---|---|
| native + sorted + eager HC | `fb334126b4e7b8f60546fe26ca4c3373cc51026990333771a6418f753b1714b1` |
| native + sorted + compiled HC | `5756204d379b6f941c059155939da5dd543addfaa51608a2d0e0051af661ac23` |
| split + sorted + eager HC | `b01bf575369144e57cab200fd71534df782a0597984cdbb64611733e2e777e5a` |
| split + sorted + compiled HC | `50ff1f9e947957339f7d9efbe99e62742c3ce335fcb7cfcabd7890d689b7d224` |

后续源码已修正为保存 `runtime_identity_sha256_by_mode`，顶层固定引用主 `grad`。修正版 SHA 为 `ad4073152601d6d078da0862e0d7b1a74c70d99946953b7c1d6a212996128ca2`；7 项 audit CPU 测试通过，包含“compiled 最后执行”回归检查。该改动仅修正标注，没有改变数学或重跑 GPU。

以上完成数值路径诊断。后续按[固定 RMSNorm 有界训练登记](qwen38_native_normshort_registration_20261020.md)刷新 H/scale、完成既定锚点检查并执行两 α 各 8 步；不据本诊断增添 ρ、norm、样本或训练步数。
