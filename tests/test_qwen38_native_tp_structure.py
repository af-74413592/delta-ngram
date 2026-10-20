import copy
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "native_qwen38"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "delta-ngram" / "src"))

from delta_ngram.qwen38_training import GradientScaleProbe, NativeTPStructurePolicy
from qwen38_native_tp_structure import manual_lookup_adjoint, native_cutpoint_gradients


def policy_args():
    return dict(
        reduction="replica", world_size=2, runtime_identity_sha256="a" * 64,
        operator_source_sha256="b" * 64, proof_source_sha256="c" * 64,
        direction_id="d" * 64,
        operator_probes=(GradientScaleProbe(.001, 3., (3., 3.)),
                         GradientScaleProbe(.002, 3., (3., 3.))),
        sp_adjoint_max_abs_errors=(0., 0.), overlay_adjoint_max_abs_errors=(0., 0.),
        overlay_reference_max_abs=(.7, .7), replicated_gradient_sha256=("e" * 64,) * 2,
        duplicated_readout_sha256=("f" * 64,) * 2)


def test_structure_receipt_has_separate_scope():
    policy = NativeTPStructurePolicy(**policy_args())
    receipt = policy.receipt()
    assert receipt["protocol"] == "qwen38_native_tp_structure_v1"
    assert receipt["does_not_validate_bf16_ce_finite_differences"]
    assert policy.receipt() == receipt


@pytest.mark.parametrize("change", [
    {"reduction": "sum"}, {"world_size": 1},
    {"operator_probes": (GradientScaleProbe(.001, 3., (6., 6.)),
                         GradientScaleProbe(.002, 3., (6., 6.)))},
    {"sp_adjoint_max_abs_errors": (0., 1e-8)},
    {"overlay_adjoint_max_abs_errors": (.7, .7)},
    {"overlay_reference_max_abs": (0., 0.)},
    {"replicated_gradient_sha256": ("a" * 64, "b" * 64)},
    {"duplicated_readout_sha256": ("a" * 64, "b" * 64)},
    {"rtol": .1},
])
def test_structure_policy_rejects_unproven_scaling(change):
    values = policy_args()
    values.update(change)
    with pytest.raises(ValueError):
        NativeTPStructurePolicy(**values)


def test_manual_lookup_adjoint_handles_repeats_padding_and_mask():
    rows = torch.tensor([1, 3, 9])
    physical = torch.tensor([[[1, 3], [1, 9], [0, 9]]])
    mask = torch.tensor([[True, True, False]])
    gradient = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)
    actual = manual_lookup_adjoint(rows, physical, gradient, mask)
    expected = torch.tensor([[4., 6.], [2., 3.], [6., 7.]])
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_cutpoints_preserve_graph_and_restore_exact_binding():
    class PLE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.ple_embedding = torch.nn.Identity()

        def compute(self, x):
            return self.ple_embedding(x) * 3

        def forward(self, x):
            return self.compute(x).transpose(0, 1)

    class Adapter:
        ple_layer = PLE()
        embedding = ple_layer.ple_embedding

    adapter = Adapter()
    x = torch.arange(6, dtype=torch.float32).reshape(1, 2, 3).requires_grad_()
    original = adapter.ple_layer.compute
    with native_cutpoint_gradients(adapter) as captures:
        adapter.ple_layer(x).sum().backward()
    assert adapter.ple_layer.compute == original
    assert "compute" not in adapter.ple_layer.__dict__
    assert set(captures) == {"full_compute", "local_sp", "embedding"}
    torch.testing.assert_close(captures["embedding"]["gradient"], torch.full_like(x, 3.))
    assert not adapter.ple_layer._forward_hooks
    assert not adapter.embedding._forward_hooks
