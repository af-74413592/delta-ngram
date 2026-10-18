"""前向调试：一条证据怎样写进记忆，一个问题怎样从记忆里读出来。

只跑前向，不更新参数。把正式代码的一次写入和一次查询展开成顺序步骤，
方便在 VS Code 中下断点，查看 hidden、rows、target、beta 和记忆状态。
外层训练还需要答案损失、反向传播和优化器更新；这个脚本不执行这些步骤。

用法：
    python scripts/debug_forward.py --checkpoint /path/to/writer.pt

模型、读者和写入器都按 checkpoint 的配置构造，目前只支持 D1。
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import torch

from delta_ngram.checkpoints import load_checkpoint, restore_writer
from delta_ngram.config import ExperimentConfig
from delta_ngram.runtime import assert_read_only, state_guard, token_ids
from delta_ngram.train import create_runtime


def load(checkpoint: str, device: str):
    """使用正式加载入口，保留模型版本和 checkpoint 身份检查。"""
    payload = load_checkpoint(checkpoint)
    values = deepcopy(payload["config"]["package_config"])
    if values["method"]["name"] != "forward" or str(values["method"]["operator"]).upper() != "D1":
        raise ValueError("this debug script only covers the D1 forward writer")
    values["model"]["device"] = "cpu" if values["model"]["smoke"] else device
    config = ExperimentConfig(values, Path(checkpoint).resolve())

    method, _ = create_runtime(config)
    restore_writer(method, payload, config)
    method.writer.requires_grad_(False)
    method.eval()
    print(f"checkpoint step {payload['step']}  |  row width {method.reader.width}"
          f"  |  orders {method.reader.orders}")
    print(f"注入位置：decoder block index {method.reader.layer_index}"
          f"（从 1 开始数是第 {method.reader.layer_index + 1} 个 block）入口")
    return method


def encode(method, text):
    """文本 → token IDs、有效位置 mask、供打印用的单 token 文本。"""
    ids = token_ids(method.tokenizer, text, method.device)
    mask = torch.ones_like(ids)
    tokens = [method.tokenizer.decode([token]) for token in ids[0].tolist()]
    return ids, mask, tokens


@torch.no_grad()
def write(method, state, text):
    """① 编码证据 → ② 产生每个事件的目标和门控 → ③ 按行更新记忆。"""
    reader, writer = method.reader, method.writer
    ids, mask, _ = encode(method, text)
    if ids.shape[1] > method.max_evidence_tokens:
        raise ValueError(f"Evidence has {ids.shape[1]} tokens, limit is {method.max_evidence_tokens}; not truncating")

    # ① 冻结模型在不注入记忆的情况下编码证据，取 LM head 输入作为 hidden。
    # hidden: [1, T, width]；rows: [1, T, 阶数]。
    hidden, rows = reader.encode_evidence(ids, mask)

    # ② target 由证据特征生成；D1 的 beta 还读取旧行值、target - 旧值和写入次数。
    events = writer.events(hidden, rows, mask, state)

    # ③ 复用正式 D1 算子。同一物理行的事件按旧状态计算，再平均更新。
    new_state = writer.operator(state, events.rows, events.target, events.beta)

    # 以下只打印中间量，不参与读写计算。
    positions = (mask.bool()[:, :, None] & rows.ne(0)).nonzero().tolist()
    grams_by_row = {}
    print(f"\n[写入] {text!r}")
    print(f"  {'位置':>4} {'n-gram':>22} {'行号':>6} {'beta':>6} {'|target|':>8}")
    target_norms = events.target.norm(dim=-1).tolist()
    for (_, position, head), row, beta, norm in zip(
            positions, events.rows.tolist(), events.beta.tolist(), target_norms):
        order = reader.orders[head]
        gram_ids = ids[0, position - order + 1:position + 1].tolist()
        gram_text = method.tokenizer.decode(gram_ids)
        grams_by_row.setdefault(row, []).append(gram_text)
        print(f"  {position:>4} {gram_text!r:>22} {row:>6} {beta:>6.3f} {norm:>8.3f}")

    touched = torch.unique(events.rows)
    before, count_before = state.lookup(touched)
    after, count_after = new_state.lookup(touched)
    change = (after - before).norm(dim=-1)
    beta_mean = f"{events.beta.mean().item():.3f}" if events.beta.numel() else "N/A"
    print(f"  共 {len(positions)} 个事件，涉及 {touched.numel()} 行；beta 均值 {beta_mean}")
    print("  向量变化最大的 6 行（变化大小不代表因果贡献）：")
    for index in change.argsort(descending=True)[:6].tolist():
        row = int(touched[index])
        print(f"    行 {row:>6}  |变化| {change[index]:.3f}"
              f"  写入次数 {int(count_before[index])}→{int(count_after[index])}"
              f"  n-gram {grams_by_row[row]}")
    return new_state


@torch.no_grad()
def ask(method, state, question, k=5):
    """④ 查看问题会读到的残差 → ⑤ 查看最后位置预测的下一个 token。"""
    reader = method.reader
    before = state_guard(method, state)
    ids, mask, tokens = encode(method, question)

    # ④ 按问题的 n-gram 地址取行，各阶向量求和后除以 sqrt(阶数)。
    # 这里取出残差供观察；实际 block 入口注入仍由 reader 完成。
    rows = reader.adapter.addresses(ids, mask)
    injected = reader.adapter.read_residual(state, rows)[0]

    # ⑤ 一次只读前向：问题末尾的 logits 预测下一个 token。
    output = reader(ids, mask, state)
    probabilities = output.logits[0, -1].float().softmax(-1)
    assert_read_only(method, state, before)
    top = probabilities.topk(k)

    print(f"\n[提问] {question!r}")
    print("  每个位置注入的残差大小：" + "  ".join(
        f"{token!r}:{norm:.2f}" for token, norm in zip(tokens, injected.norm(dim=-1).tolist())))
    print(f"  下一 token 的前 {k} 个候选（不是完整答案评分）：" + "  ".join(
        f"{method.tokenizer.decode([token])!r} {probability:.2f}"
        for probability, token in zip(top.values.tolist(), top.indices.tolist())))
    return probabilities


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args(argv)
    method = load(args.checkpoint, args.device)
    empty = method.empty_state()

    evidence_a = "Profile\nName: Kalibo\nFavorite color: white\n"
    evidence_b = "Profile\nName: Kalake\nFavorite color: black\n"
    question_a = "Profile\nName: Kalibo\nFavorite color:"
    question_b = "Profile\nName: Kalake\nFavorite color:"
    paraphrase_a = "What is Kalibo's favorite color? Answer with one word:"

    # 建议从这里下断点，按顺序观察空表、第一次写入和第二次写入。
    ask(method, empty, question_a)
    state_after_a = write(method, empty, evidence_a)
    ask(method, state_after_a, question_a)
    state_after_b = write(method, state_after_a, evidence_b)
    ask(method, state_after_b, question_b)
    ask(method, state_after_b, question_a)
    ask(method, state_after_b, paraphrase_a)


if __name__ == "__main__":
    main()
