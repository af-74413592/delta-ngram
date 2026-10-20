import copy
import json
import unittest

import torch

from delta_ngram.qwen38_native import make_qwen38_writer
from delta_ngram.qwen38_probe import (
    NativeProbeLedger, build_history_memory, policy_from_receipt, query_frame, tensor_state_hash,
)
from delta_ngram.qwen38_training import GradientScaleProbe, NativeTPStructurePolicy, TPGradientPolicy


class QueryContracts(unittest.TestCase):
    def test_ordinary_shift_and_prefix(self):
        ids = list(range(1000))
        result = query_frame({"kind": "ordinary", "sample_id": "x", "source": "books"}, ids)
        self.assertEqual(result["input_ids"], ids[384:640])
        self.assertEqual(result["labels"][127:255], ids[512:640])
        self.assertTrue(all(value == -100 for value in result["labels"][:127]))
        self.assertEqual(result["labels"][-1], -100)
        self.assertEqual(sum(value != -100 for value in result["labels"]), 128)
        self.assertFalse(any(result["write_mask"]))

    def test_event_local_bos_and_padding(self):
        ids = list(range(300))
        result = query_frame({"kind": "event", "sample_id": "e", "source": "news_event"}, ids)
        self.assertEqual(result["input_ids"], [248044] + ids[:128] + [248044])
        self.assertEqual(result["labels"], ids[:128] + [-100, -100])
        self.assertEqual(result["encoder_valid_mask"], [True] * 129 + [False])

    def test_short_sample_is_not_replaced_or_padded_into_eligibility(self):
        with self.assertRaises(ValueError):
            query_frame({"kind": "ordinary", "sample_id": "s", "source": "news"}, [1] * 639)
        with self.assertRaises(ValueError):
            query_frame({"kind": "event", "sample_id": "e", "source": "news_event"}, [1] * 127)


class BudgetContracts(unittest.TestCase):
    def test_prior_failed_diagnostics_count_and_dispatch_is_durable(self):
        saved = []
        ledger = NativeProbeLedger(prior_diagnostic=473, prior_optimizer=0,
                                   prior_normshort=440, save=lambda x: saved.append(copy.deepcopy(x)))
        ledger.reserve("initial", targets=128, inputs=256)
        ledger.reserve("train", targets=128, inputs=256, optimizer=True)
        self.assertEqual(saved[-1]["normshort_target_accesses"], 696)
        self.assertEqual(saved[-1]["global_diagnostic_target_accesses"], 729)
        self.assertEqual(saved[-1]["writer_optimizer_target_accesses"], 128)
        with self.assertRaises(ValueError):
            ledger.reserve("train", targets=128, inputs=256, optimizer=True)
        self.assertEqual(len(saved), 2)

    def test_cap_checked_before_dispatch(self):
        saved = []
        ledger = NativeProbeLedger(prior_diagnostic=74900, prior_optimizer=0,
                                   prior_normshort=74900, save=saved.append)
        with self.assertRaises(ValueError):
            ledger.reserve("forbidden", targets=128, inputs=256)
        self.assertEqual(saved, [])
        self.assertEqual(ledger.targets, 0)


class PolicyReceiptContracts(unittest.TestCase):
    def test_json_roundtrip_of_numerical_policy_and_tamper_rejected(self):
        probes = (GradientScaleProbe(.01, 2., (2., 2.)), GradientScaleProbe(.02, 2., (2., 2.)))
        policy = TPGradientPolicy("replica", 2, "a" * 64, "direction", probes, .01, .001)
        receipt = json.loads(json.dumps(policy.receipt()))
        self.assertEqual(policy_from_receipt(receipt).receipt(), policy.receipt())
        receipt["receipt_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            policy_from_receipt(receipt)

    def test_json_roundtrip_of_structure_policy(self):
        probes = (GradientScaleProbe(.01, 2., (2., 2.)), GradientScaleProbe(.02, 2., (2., 2.)))
        policy = NativeTPStructurePolicy(
            "replica", 2, "a" * 64, "b" * 64, "c" * 64, "d" * 64,
            probes, (0., 0.), (0., 0.), (1., 1.), ("e" * 64,) * 2, ("f" * 64,) * 2)
        receipt = json.loads(json.dumps(policy.receipt()))
        self.assertEqual(policy_from_receipt(receipt).receipt(), policy.receipt())


class HistoryContracts(unittest.TestCase):
    def test_two_native_writes_keep_writer_graph_and_detach_H(self):
        torch.manual_seed(17)
        writer, alpha = make_qwen38_writer(alpha=.99)
        hidden = [torch.randn(1, 3, 2560, requires_grad=True) for _ in range(2)]
        rows = torch.arange(1, 49).reshape(1, 3, 16)
        blocks = [{"H_new": value, "physical_rows_new": rows,
                   "T0_head_rms_new": torch.full((16,), .008)} for value in hidden]
        diagnostics = {}
        state = build_history_memory(writer, alpha, blocks, norm_mode="rmsnorm",
                                     row_capacity=100, device="cpu", diagnostics=diagnostics)
        self.assertEqual(int(state.block_clock[0]), 2)
        self.assertTrue(state.values.requires_grad)
        self.assertEqual(diagnostics["write_blocks"], 2)
        self.assertEqual(diagnostics["write_events"], 96)
        self.assertEqual(len(diagnostics["scaled_target_rms_by_head"]), 16)
        self.assertEqual(len(diagnostics["final_effective_delta_rms_by_head_on_history_events"]), 16)
        state.values.square().sum().backward()
        self.assertTrue(all(value.grad is None for value in hidden))
        self.assertTrue(any(value.grad is not None and bool(value.grad.ne(0).any())
                            for value in writer.parameters()))
        self.assertTrue(all(value.grad is None or bool(value.grad.isfinite().all())
                            for value in writer.parameters()))

    def test_initial_state_tensor_hash_changes_with_parameter_bytes(self):
        original = {"x": torch.ones(2, 3)}
        changed = {"x": original["x"].clone()}
        changed["x"][0, 0] += .01
        self.assertNotEqual(tensor_state_hash(original), tensor_state_hash(changed))
        self.assertEqual(tensor_state_hash(original), tensor_state_hash({"x": original["x"].clone()}))


if __name__ == "__main__":
    unittest.main()
