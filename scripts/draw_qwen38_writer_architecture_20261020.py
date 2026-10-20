"""Render the native writer design as editable SVG and readable PNG."""
from pathlib import Path
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def build(output: Path):
    plt.rcParams.update({
        "font.family": ["Microsoft YaHei", "DejaVu Sans"], "font.size": 13,
        "axes.unicode_minus": False, "svg.fonttype": "none",
    })
    fig, ax = plt.subplots(figsize=(20, 13))
    fig.patch.set_facecolor("#f7f9fc")
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    ink, muted = "#172f4a", "#53657a"
    blue, green, orange = "#dcecff", "#dff2e9", "#fff0d7"

    def box(x, y, w, h, title, body="", color="white", fontsize=13):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.012",
            linewidth=1.25, edgecolor="#8ca3bc", facecolor=color, zorder=3))
        if body:
            ax.text(x+w/2, y+h*.73, title, ha="center", va="center",
                    fontsize=fontsize+1, color=ink, weight="bold", zorder=4)
            ax.text(x+w/2, y+h*.34, body, ha="center", va="center",
                    fontsize=fontsize, color=ink, linespacing=1.65, zorder=4)
        else:
            ax.text(x+w/2, y+h/2, title, ha="center", va="center",
                    fontsize=fontsize, color=ink, linespacing=1.6, zorder=4)

    def arrow(start, end, color=muted, style="-", bend=0):
        ax.add_patch(FancyArrowPatch(
            start, end, arrowstyle="-|>", mutation_scale=18, linewidth=1.7,
            color=color, linestyle=style, connectionstyle=f"arc3,rad={bend}",
            zorder=2))

    ax.text(.035, .966, "Qwen3.8-Flash-Next · 原生 PLE + 前向 writer",
            fontsize=25, weight="bold", color=ink, va="center")
    ax.text(.035, .93, "接口微调：2560 维证据 → 160 维增量行；D1 块内平均保持；α 仅比较 1 / 0.99",
            fontsize=15, color=muted)
    for x, label in [(.18, "① 证据编码与 writer"), (.5, "② 稀疏记忆更新"), (.82, "③ 原生模型读取")]:
        ax.text(x, .884, label, fontsize=17, color=ink, weight="bold", ha="center")

    box(.04, .79, .28, .065, "已到达证据块 xᵦ", color=blue)
    box(.04, .655, .28, .10, "冻结原生模型 · Δ = 0",
        "T⁰ 保留 → HC mixer → H [2560]\nH detach；不携带问题或答案", blue)
    box(.04, .49, .28, .12, "事件特征 fₐ [7712]",
        "Hₚ + 右窗均值 Hₚ₊₁:ₚ₊₄\n+ 块级 Pool(H) + Emb(阶数, 头)", blue)
    box(.04, .345, .28, .10, "目标头：7712 → 256 → 160",
        "SiLU；输出 vₐ（每头固定尺度）", blue)
    box(.04, .18, .28, .115, "门控头：8033 → 256 → 1",
        "输入 fₐ、旧 δ̃、vₐ−δ̃、log(1+c)\nβₐ = sigmoid(·)", blue)
    for a, b in [((.18,.79),(.18,.765)), ((.18,.655),(.18,.61)),
                 ((.18,.49),(.18,.445)), ((.18,.345),(.18,.295))]:
        arrow(a, b)

    box(.37, .765, .26, .09, "旧稀疏状态 Δ",
        "只存触及的物理行；每行 160 维", green)
    box(.37, .615, .26, .105, "每个有效证据块衰减一次",
        "δ̃ = αΔ；α ∈ {1, 0.99}\n未命中行也按块龄惰性衰减", green)
    box(.37, .365, .26, .19, "同一物理行 · D1 块内平均",
        "β̄ⱼ = (1/nⱼ) Σₐ βₐ\nΔ⁺ⱼ = (1−β̄ⱼ) δ̃ⱼ + (1/nⱼ) Σₐ βₐvₐ\n所有事件读取同一份旧快照", green, 12)
    box(.37, .18, .26, .105, "新稀疏状态 Δ⁺",
        "FP32 累积；保留离线 BPTT 梯度\n查询只读，不推进衰减时钟", green)
    arrow((.5,.765),(.5,.72))
    arrow((.5,.615),(.5,.555))
    arrow((.5,.365),(.5,.285))
    arrow((.32,.4),(.37,.44), color="#2873ad")
    arrow((.32,.235),(.37,.395), color="#2873ad")
    # The gate's explicit state dependence is shown; targets do not read delta.
    arrow((.37,.65),(.32,.27), color="#aa7134", style="--", bend=.17)
    ax.text(.335, .575, "旧状态\n入门控", color="#956027", fontsize=10,
            ha="center", va="center", zorder=4)

    box(.68, .79, .28, .065, "下一块训练文本 / 在线查询", color=orange)
    box(.68, .655, .28, .10, "原生 n-gram 地址与查表",
        "2 阶 × 8 头 + 3 阶 × 8 头\n原生 hash / offsets / EOS 语义", orange)
    box(.68, .49, .28, .12, "拼接 16 × 160 = 2560",
        "每头 lookup(T⁰) + lookup(Δ⁺)\n在原生 key/value 投影之前相加", orange)
    box(.68, .325, .28, .12, "保持原生 PLE 读出",
        "key/value 投影 → HC 门控 / 归一化\n→ 因果卷积 → 原生 PLE 层（ID 2）", orange, 12)
    box(.68, .18, .28, .10, "冻结主干 → HC mixer → LM head",
        "离线：后续 token CE；在线：前向预测", orange, 12)
    for a, b in [((.82,.79),(.82,.765)), ((.82,.655),(.82,.61)),
                 ((.82,.49),(.82,.445)), ((.82,.325),(.82,.28))]:
        arrow(a, b)
    arrow((.63,.23),(.68,.53), color="#31856c", bend=-.08)
    ax.text(.653, .35, "Δ⁺", fontsize=13, color="#31856c", ha="center")

    ax.text(.035, .119, "冻结：主干、LM head、T⁰、原生 PLE 投影 / 门控 / 卷积。   训练：writer（约 408 万参数）；α 固定。",
            fontsize=13, color=ink)
    ax.text(.035, .081, "训练梯度：后续 CE → 冻结 reader 的计算图 → Δ → writer；证据编码 H 已 detach。在线无反向、无优化器。",
            fontsize=13, color=ink)
    ax.text(.035, .043, "代码隔离：M1 原实现保留；原生适配位于独立 qwen38_* 模块。CPU 组件已验证，全模型 hook 与 GPU 尚待验收。",
            fontsize=12, color=muted)
    output.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        fig.savefig(output / f"qwen38_delta_writer_architecture_20261020.{suffix}",
                    dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    build(parser.parse_args().output)
