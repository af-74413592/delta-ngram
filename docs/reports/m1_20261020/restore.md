# M1 / RTX 4090 归档恢复说明

本说明给出归档身份、恢复映射和校验命令。**B2 对象位置、上传与回读的最终状态，以[归档索引](archive_manifest.json)和独立校验收据为准。** 封存清单记录的是封存时点；其中 `cloud_upload_verified=false` 不会在上传后回写，后续成功状态由新的回读收据证明。

效果结论和统计范围见[正式报告](report.md)。恢复文件不会自动启动训练或 FINAL 评测；M1 的原 100M 预算已经到达完整轨迹允许的终点。

**Core 与 supplement 均已通过 B2 全量独立回读，合计 1,753 文件、25,120,671,034 bytes。** 新增的 DEV64 full-context 原始权重参照及 CPU 分析已完成，增量 B2 归档仍待验收，不包含在这两份封存对象集中；新增结果见[独立报告](context_reference.md)。

## 1. 归档分为两部分

### 1.1 Core：GPU 实例上的训练、数据与环境材料

Core 的原始封存目录为 `/workspace/delta-ngram-backup-stage-20261020`，目标 bucket 为 `clclcl`，目标前缀为 `cl-research/delta-ngram/20261020-rtx4090/`。实际对象键以回读收据为准；下载后应直接看到 `data/`、`runs/`、`operations/`、`source_snapshots/`、`model_dependencies/`、`environment/`、`metadata/`。

| 计数口径 | 已封存身份 |
|---|---:|
| `member_manifest.jsonl` 列出的成员 | 1,730 文件，23,884,626,562 bytes |
| 另加的封口文件 | `COMPLETE.json`、`member_manifest.jsonl`、`member_sha256.txt` |
| 完整 Core 对象集 | **1,733 文件，23,885,567,983 bytes** |
| 封存完成时间 | 2026-10-20 05:23:38.335976 UTC |

Core 已在独立目录 `/workspace/delta-ngram-backup-restorecheck-20261020` 完成一次全量回读检查：1,730 个成员及 3 个封口文件的 SHA256 全部一致，恢复文件 inode 全部与原 stage 独立。[主体回读收据](backup_receipts/core_restore_verification.json)时间为 2026-10-20 05:37:47.711632 UTC，SHA256 为 `bb243094e0ea3052ab9264a3e4e37b13ea4f98fd3489b09156e9e96af97c9cc0`。它只证明 Core；supplement 的独立回读结果见下一节。

封口文件的固定 SHA256：

| 文件 | SHA256 |
|---|---|
| `metadata/COMPLETE.json` | `ae0efc8a826d4173664cff3ee8e046a8a55ee38b9aa6d6c53123c6fce9507e95` |
| `metadata/member_manifest.jsonl` | `be4a8b39dd2f5c3e50fe9aea472a1f2d06c9a28e3717db674d1762b821502611` |
| `metadata/member_sha256.txt` | `72dd24744e0c0551851e0edbb917189b860e88c63c88cc4a8443a5f9d4f34b43` |
| `metadata/backup_summary.json` | `b064f40003bfc61f48d64c1efe592bf6e0036cd46b5ea0d380f989e3738d0d47` |

### 1.2 Supplement：正式分析、Git 历史与 P2 Base 依赖

补充材料使用独立封存目录 `/workspace/delta-ngram-closure-supplement-20261020`；B2 实际前缀为 `cl-research/delta-ngram/20261020-rtx4090/supplement/`，位于同一 bucket `clclcl` 的 Core 前缀之下。补充包共 **20 个对象、1,235,103,051 bytes**，单独计数与核验，不改变 Core 的 1,733 文件身份。下载 Core 时必须排除这个嵌套前缀，再将 supplement 单独下载到另一空目录，详见第 3 节。

Supplement 已恢复到独立目录 `/workspace/delta-ngram-supplement-restorecheck-20261020`，在 2026-10-20 05:56:17.804734 UTC 完成全量核验：20 个对象的 SHA256、大小和精确文件集合全部匹配，恢复文件与 stage 的 inode 独立，无 symlink 或未完成文件。[补充回读收据](backup_receipts/supplement_restore_verification.json)的 SHA256 为 `6a62d724fc22910ca140d3361ab632ce1af228c9cf61ac1ec9a08b97e5da375e`。

封存时原件可在 Git 中复查：[COMPLETE](backup_receipts/supplement_COMPLETE.json)、[成员清单](backup_receipts/supplement_member_manifest.jsonl)、[SHA256 清单](backup_receipts/supplement_sha256.txt)及[外部封存回执](backup_receipts/supplement_finalization_receipt.json)。最终回读收据在封存后产生，保存在后续 Git 提交中，不属于本次 B2 补充包；封口中的 `b2_upload_verified=false`、`b2_restore_verified=false` 保留历史时点，不回写为新状态。

| 对象 | 固定身份或用途 |
|---|---|
| `analysis/m1_analysis_bundle_20261020.zip` | 27,767,060 bytes；SHA256 `e9cad1aea3cf5a7e90a6ec55da57026aa1cefedc9f7682e6cd1ca314be11cb40` |
| `code/delta_ngram_code_fcd0564_20261020.bundle` | 完整 Git 历史快照；3,617,902 bytes；SHA256 `bf5f2728b311938ec9830ebc980c3f36bb3db38fc5ac77827709dde1373ec550` |
| `code/code_bundle_restore_receipt_20261020.json` | 上述 bundle 的独立克隆验证记录；HEAD `fcd056414fd148425940d76c845b39e0ec14e381` |
| `model_dependencies/Qwen3-0.6B-Base__da87bfb608c14b7cf20ba1ce41287e8de496c0cd/` | P2 已缓存 Base 模型的 7 个文件，合计 1,203,625,970 bytes |
| `metadata/p2_base_dependency_manifest.json` | 上述 7 文件逐成员 SHA256；清单自身 SHA256 `905b0ff1f3fdc37d0000fa5ae3c1a228c6ff932d0f66b6a90fa449b276f59428` |

Base 依赖仅从原机器已有的指定 revision 缓存解析并硬链接，没有重新下载模型或执行 GPU 工作。P2 的 tokenizer 配置仍使用 Core 中的 `Qwen/Qwen3-0.6B` revision `c1899de289a04d12100db370d81485cdf75e47ca`。

本说明和最终回读收据晚于上述 `fcd0564` 代码快照，由后续 Git 提交保存。B2 包内的 `RESTORE.md` 保留封存时版本，本仓库说明补充了独立下载、固定 SHA 预检和成功回读记录。运行本次 M1 需要的历史提交已在该 bundle 中核实存在。

## 2. 关键材料覆盖核对

以下检查基于 Core 已封清单，并与实际 M1 预检、训练记录及完整 DEV64 原始结果交叉核对：

| 必需材料 | 核对结果 |
|---|---|
| M1 0M、5M……100M writer | 全部 21 个登记路径、大小及 SHA256 匹配 |
| 中间训练恢复状态 | 20 对 `writer_step_*` / `training_step_*`；另有 `writer_init.pt`、终点 `writer.pt` / `training.pt`，共 43 个 `.pt` |
| 第零步恢复边界 | 原运行没有单独的 `training_init.pt`；保存了随机初始化 writer、配置、seed、数据与初始化来源，不能声称存在第零步 optimizer 快照 |
| 实际 M1 训练输入 | `merged_event_v5/train_validation.jsonl`、`train_validation_event_streams.jsonl`、`data_manifest.json` 的 SHA256 与启动预检一致 |
| 原始文本与构建记录 | `selected_texts.jsonl.gz`、`selected_event_texts.jsonl.gz`、候选数据库及构建/划分/容量清单均在 `data/` 中 |
| DEV64 输入 | 146 份提取文本、146 份原始响应/记录的 SHA256 与评分 manifest 匹配；registry 与 prefix 标注匹配 |
| DEV64 输出 | 21 份完整结果及 `runtime.json`、`completion.json` 与已验收本地副本逐字节 SHA256 一致 |
| α 对照 | 四个完整分支与 `fixed_099_probe20` 各有 writer/training/alpha 三件套；原前缀、配置、日志、注册记录均保留 |
| 历史源码 | 12 个冻结评测文件与正式源码清单一致；9 个 M1 训练核心文件与实际训练记录的源码 SHA256 一致 |
| M1 模型与 tokenizer | `Qwen/Qwen3-0.6B` 的指定 revision 共 7 个文件，合计 1,519,182,365 bytes；缓存 symlink 已解析为普通归档文件 |
| P2 中间恢复状态 | 两个 `training_step_*` 已包含在 Core，Base 模型另在 supplement |
| 环境证据 | Python/package 版本、GPU/driver、源码 revision、模型依赖说明和 staging 脚本共 6 文件，SHA256 已核对 |

Core 的 24 条排除项全部为源码目录下的 `.git` 或缓存目录，没有排除训练数据、结果或权重。Git 历史通过 supplement 的 bundle 单独保存。Python/CUDA 安装介质和其他无关 Hugging Face 缓存未归档；本地新生成的正式分析包、报告和代码快照由 supplement/Git 提供。

## 3. 分别下载并校验

以下命令适用于已安装 Python 3.11+ 和 rclone 的 Linux 恢复环境。将 `existingRemote` 替换为已配置、可访问该 B2 bucket 的 rclone remote 名称；不需要在命令中填写凭据。将 `CORE` 和 `SUPPLEMENT` 设置为两份下载结果的根目录，两个目录必须为空、互相独立且不相互嵌套。

```bash
set -euo pipefail
export CORE=/workspace/restored-core-20261020
export SUPPLEMENT=/workspace/restored-supplement-20261020

python - "$CORE" "$SUPPLEMENT" <<'PY'
import os, sys
from pathlib import Path

requested = [Path(value) for value in sys.argv[1:]]
if any(path.is_symlink() for path in requested):
    raise SystemExit("Restore destinations must not be symlinks.")
core, supplement = [path.resolve() for path in requested]
if core == supplement or core.is_relative_to(supplement) or supplement.is_relative_to(core):
    raise SystemExit("Core and supplement destinations must be separate, non-nested directories.")
for path in (core, supplement):
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise SystemExit(f"Restore destination must be empty: {path}")
for path in (core, supplement):
    path.mkdir(parents=True, exist_ok=True)
if os.path.samefile(core, supplement):
    raise SystemExit("Restore destinations resolve to the same directory.")
print("Two empty, independent restore destinations are ready.")
PY

rclone copy existingRemote:clclcl/cl-research/delta-ngram/20261020-rtx4090/ \
  "$CORE" --exclude '/supplement/**'
rclone copy existingRemote:clclcl/cl-research/delta-ngram/20261020-rtx4090/supplement/ \
  "$SUPPLEMENT"
```

`--exclude '/supplement/**'` 是本次目录布局所必需的过滤条件。如果直接下载整个父前缀，Core 目录会多出 supplement 的 20 个对象，下面的精确 1,733 文件检查会正确拒绝该目录。不要通过放宽校验来接受混合目录；应分别恢复两份对象集。

### 3.1 Core：封口身份、逐成员内容和完整文件集合

这一步只读文件，不启动模型：

```bash
python - "$CORE" <<'PY'
import hashlib, json, sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
pins = {
    "metadata/COMPLETE.json": "ae0efc8a826d4173664cff3ee8e046a8a55ee38b9aa6d6c53123c6fce9507e95",
    "metadata/member_manifest.jsonl": "be4a8b39dd2f5c3e50fe9aea472a1f2d06c9a28e3717db674d1762b821502611",
    "metadata/member_sha256.txt": "72dd24744e0c0551851e0edbb917189b860e88c63c88cc4a8443a5f9d4f34b43",
    "metadata/backup_summary.json": "b064f40003bfc61f48d64c1efe592bf6e0036cd46b5ea0d380f989e3738d0d47",
}
def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

for name, expected in pins.items():
    assert digest(root / name) == expected, name
complete = json.loads((root / "metadata/COMPLETE.json").read_bytes())
rows = [json.loads(line) for line in
        (root / "metadata/member_manifest.jsonl").read_bytes().splitlines()]
assert complete["complete"] is True
assert len(rows) == len({r["relative_path"] for r in rows}) == 1730
assert sum(r["size_bytes"] for r in rows) == 23884626562
for row in rows:
    path = root / row["relative_path"]
    assert path.resolve().is_relative_to(root) and not path.is_symlink(), path
    assert path.stat().st_size == row["size_bytes"], path
    assert digest(path) == row["sha256"], path

seals = {"metadata/COMPLETE.json", "metadata/member_manifest.jsonl",
         "metadata/member_sha256.txt"}
expected = {r["relative_path"] for r in rows} | seals
actual = set()
total = 0
for path in root.rglob("*"):
    assert not path.is_symlink(), path
    if path.is_file():
        actual.add(path.relative_to(root).as_posix())
        total += path.stat().st_size
assert actual == expected, (sorted(expected - actual), sorted(actual - expected))
assert len(actual) == 1733 and total == 23885567983
print("Core: 1733 files, all SHA256 and byte counts verified.")
PY
```

### 3.2 Supplement：完整补充包核验

补充包的封口与验证脚本已经固定；下列 SHA256 已与独立保存的封存回执及本地副本核对。这些是文件身份，不代替云端回读收据。

| 文件 | 固定 SHA256 |
|---|---|
| `metadata/COMPLETE.json` | `f1dbbb7c2795b4c480b7fdac01ba5bac2a8bc9a832a9c7bfedef9593ad3a148f` |
| `metadata/supplement_member_manifest.jsonl` | `762e62db16cfa7d30058e05cd897dcbbc31fc2d6af7f02926b44824e3b11600f` |
| `metadata/sha256.txt` | `48f97e76611d3c5b4018cc7106f9c88634d1c614634dc42675b752f1acc1f40c` |
| `metadata/verify_supplement_20261020.py` | `f8bbd451d9609a8963aa69493f47390874d4f8c2f0e045f4b2c028bcc384433c` |

**先用下面的独立代码核对这四个文件，再执行归档中的验证脚本：**

```bash
python - "$SUPPLEMENT" <<'PY'
import hashlib, sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
pins = {
    "metadata/COMPLETE.json": "f1dbbb7c2795b4c480b7fdac01ba5bac2a8bc9a832a9c7bfedef9593ad3a148f",
    "metadata/supplement_member_manifest.jsonl": "762e62db16cfa7d30058e05cd897dcbbc31fc2d6af7f02926b44824e3b11600f",
    "metadata/sha256.txt": "48f97e76611d3c5b4018cc7106f9c88634d1c614634dc42675b752f1acc1f40c",
    "metadata/verify_supplement_20261020.py": "f8bbd451d9609a8963aa69493f47390874d4f8c2f0e045f4b2c028bcc384433c",
}
for name, expected in pins.items():
    path = root / name
    if path.is_symlink() or not path.resolve().is_relative_to(root):
        raise SystemExit(f"Unexpected verification path: {name}")
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise SystemExit(f"SHA256 mismatch: {name}")
print("Supplement seals and verifier SHA256 verified.")
PY

python3 "$SUPPLEMENT/metadata/verify_supplement_20261020.py" \
  --root "$SUPPLEMENT" \
  --expected-complete-sha256 f1dbbb7c2795b4c480b7fdac01ba5bac2a8bc9a832a9c7bfedef9593ad3a148f \
  --receipt /workspace/supplement_restore_verification.json
```

回执应写到恢复目录之外。若原封存目录仍可访问，额外传入 `--source-stage /workspace/delta-ngram-closure-supplement-20261020`，核对恢复文件与原 stage 的 inode 不同。该脚本检查全部对象数量、大小和 SHA256，并拒绝 symlink 或未完成的临时文件。不要从待验证的 COMPLETE 文件自身计算“预期 SHA”来替代独立封存回执。

### 3.3 Supplement：独立核对 P2 Base 依赖

下面只验证 Base 模型及其清单。analysis/code/最终报告等其余补充对象须同时通过 supplement 总清单与回读收据：

```bash
python - "$SUPPLEMENT" <<'PY'
import hashlib, json, sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
manifest = root / "metadata/p2_base_dependency_manifest.json"
assert hashlib.sha256(manifest.read_bytes()).hexdigest() == \
    "905b0ff1f3fdc37d0000fa5ae3c1a228c6ff932d0f66b6a90fa449b276f59428"
data = json.loads(manifest.read_bytes())
assert data["revision"] == "da87bfb608c14b7cf20ba1ce41287e8de496c0cd"
assert len(data["members"]) == 7
assert sum(row["size_bytes"] for row in data["members"]) == 1203625970
for row in data["members"]:
    path = root / row["relative_path"]
    assert path.resolve().is_relative_to(root) and not path.is_symlink(), path
    assert path.stat().st_size == row["size_bytes"], path
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    assert h.hexdigest() == row["sha256"], path
print("P2 Base: 7 files and manifest SHA256 verified.")
PY
```

## 4. 恢复到原执行路径

保留原目录关系可复用已冻结配置与原文件身份：

| 归档位置 | 原执行位置 |
|---|---|
| Core `data/` | `/workspace/delta-ngram-data/` |
| Core `runs/` | `/workspace/delta-ngram-runs/` |
| Core `operations/` | `/workspace/delta-ngram-transfer-20261019/` |
| Core `source_snapshots/delta-ngram-m1/` | `/workspace/delta-ngram-m1/`，对应下表冻结 Git 提交 |
| Core `source_snapshots/delta-ngram-evaluation-20261019/` | `/workspace/delta-ngram-evaluation-20261019/`，对应下表冻结 Git 提交 |
| Core 其他 `source_snapshots/<name>/` | 原 `/workspace/<name>/`；无 `.git` 的副本用于源码比对 |
| Core `model_dependencies/Qwen--Qwen3-0.6B/c189…/` | `/workspace/.hf_home/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca/` |
| Supplement `model_dependencies/Qwen3-0.6B-Base__da87…/` | `/workspace/.hf_home/hub/models--Qwen--Qwen3-0.6B-Base/snapshots/da87bfb608c14b7cf20ba1ce41287e8de496c0cd/` |

模型文件可直接放入上述 snapshot 目录，不需要重新创建原机器的 blob symlink。配置继续使用原模型名称及完整 revision，避免因修改模型路径而改变严格恢复的配置身份。

以下复制命令面向**尚未存在目标项目目录的新环境**。任一目标已存在就停止，不合并或覆盖已有实验：

```bash
python - "$CORE" "$SUPPLEMENT" <<'PY'
import shutil, sys
from pathlib import Path

core, supplement = (Path(value).resolve() for value in sys.argv[1:])
workspace = Path("/workspace")
rev = "c1899de289a04d12100db370d81485cdf75e47ca"
base = "da87bfb608c14b7cf20ba1ce41287e8de496c0cd"
pairs = [
    (core / "data", workspace / "delta-ngram-data"),
    (core / "runs", workspace / "delta-ngram-runs"),
    (core / "operations", workspace / "delta-ngram-transfer-20261019"),
    (core / "source_snapshots/delta-ngram-sdist-before-git-d7b49ee8cff1",
     workspace / "delta-ngram-sdist-before-git-d7b49ee8cff1"),
    (core / f"model_dependencies/Qwen--Qwen3-0.6B/{rev}",
     workspace / f".hf_home/hub/models--Qwen--Qwen3-0.6B/snapshots/{rev}"),
    (supplement / f"model_dependencies/Qwen3-0.6B-Base__{base}",
     workspace / f".hf_home/hub/models--Qwen--Qwen3-0.6B-Base/snapshots/{base}"),
]
for src, dst in pairs:
    assert src.is_dir(), src
    assert not dst.exists() and not dst.is_symlink(), dst
for src, dst in pairs:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst, copy_function=shutil.copy2)
print("Data, run states, operations, source distribution and model caches copied.")
PY

export HF_HOME=/workspace/.hf_home
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

### 用 Git bundle 恢复冻结执行仓库

Core 源码副本排除了 `.git`。需要 Git 版本检查的历史入口，应使用已校验 bundle 创建仓库并 checkout 相应历史提交；不要直接在最新原生原型上运行 M1。

| 原仓库目录 | 冻结 HEAD |
|---|---|
| `delta-ngram` | `30a34215719e3575c8c4a3f15caaa544370c1546` |
| `delta-ngram-alpha` | `30a34215719e3575c8c4a3f15caaa544370c1546` |
| `delta-ngram-m1` | `d0dec9f1bc87b05e4d100dcd9b81863bfc5ce247` |
| `delta-ngram-evaluation-20261019` | `167d24783161be7842bbb9aec52f2c7804191ba9` |

例如恢复 M1 训练与 DEV64 评测仓库；要求两个目标目录尚不存在：

```bash
export CODE_BUNDLE="$SUPPLEMENT/code/delta_ngram_code_fcd0564_20261020.bundle"

python - "$CODE_BUNDLE" <<'PY'
import hashlib, sys
from pathlib import Path
p = Path(sys.argv[1])
assert hashlib.sha256(p.read_bytes()).hexdigest() == \
    "bf5f2728b311938ec9830ebc980c3f36bb3db38fc5ac77827709dde1373ec550"
print("Code bundle SHA256 verified.")
PY

git clone "$CODE_BUNDLE" /workspace/delta-ngram-m1
git -C /workspace/delta-ngram-m1 checkout --detach d0dec9f1bc87b05e4d100dcd9b81863bfc5ce247
git clone "$CODE_BUNDLE" /workspace/delta-ngram-evaluation-20261019
git -C /workspace/delta-ngram-evaluation-20261019 checkout --detach 167d24783161be7842bbb9aec52f2c7804191ba9
```

其他两份仓库同理使用表中提交。源码字节可继续与 Core 清单核对；运行登记和分析脚本的来源说明见[冻结登记](registration/README.md)与[归档脚本清单](../../../scripts/archive/m1_20261020/provenance_manifest.json)。

## 5. 环境与恢复边界

Core `environment/python_packages.json` 记录 Python **3.12.15**、PyTorch **2.14.1+cu132**、Transformers **5.5.0**、NumPy **2.5.4**、tokenizers **0.22.2**、safetensors **0.8.0**；GPU 记录为 RTX 4090、49,140 MiB、driver **615.71.09**。安装介质未包含在归档中，恢复环境需另行提供兼容的 Python/CUDA 运行时。

文件回读通过证明归档内容与封存内容一致，尚不等同于在新环境中再次通过训练恢复验收。完整续训需使用对应的 `training.pt` 或 `training_step_*`，其中包含 writer、optimizer、sampler、计数及 RNG；只加载 `writer.pt` 适用于固定权重评测，不构成完整续训。

主要数值统计可按[报告第 10 章](report.md)从独立分析 ZIP 在 CPU 上复算。重新推理或续训前应保持原数据、源码和模型 revision；FINAL 仍遵循 100M 唯一主检查点、最多一次的既有登记。
