"""Method construction and the single forward-write entry point."""
from __future__ import annotations

import torch

from .interfaces import MemoryMethod, WriteResult
from .operators import WriteOperator
from .runtime import token_ids
from .state import SparseMemoryState
from .writer import ForwardWriter


class ForwardMemoryMethod(MemoryMethod):
    name = "forward"
    supports_outer_training = True

    def __init__(self, reader, tokenizer, device, writer, max_evidence_tokens=128):
        super().__init__(reader, tokenizer, device, max_evidence_tokens)
        self.writer = writer

    def empty_state(self):
        return SparseMemoryState.empty(self.reader.width, self.device)

    def write(self, state, evidence):
        ids = token_ids(self.tokenizer, evidence, self.device)
        if ids.shape[1] > self.max_evidence_tokens:
            raise ValueError(f"Evidence has {ids.shape[1]} tokens, limit is {self.max_evidence_tokens}; not truncating")
        mask = torch.ones_like(ids)
        hidden, rows = self.reader.encode_evidence(ids, mask)
        return WriteResult(self.writer(hidden, rows, mask, state))

    def parameters(self):
        return self.writer.parameters()

    def train(self, mode=True):
        self.writer.train(mode)
        self.reader.model.eval()
        return self


class EmptyMemoryMethod(MemoryMethod):
    name = "empty"

    def empty_state(self):
        return SparseMemoryState.empty(self.reader.width, self.device)

    def write(self, state, evidence):
        return WriteResult(state)


def make_method(config, reader, tokenizer, device):
    values = config.values
    spec = values["method"]
    common = {"reader": reader, "tokenizer": tokenizer, "device": device,
              "max_evidence_tokens": values["run"]["max_evidence_tokens"]}
    name = spec["name"]
    if name == "forward":
        frozen = set(spec["frozen_rows"])
        for fragment in spec["frozen_template_texts"]:
            ids = token_ids(tokenizer, fragment, device)
            _, rows = reader._inputs(ids, torch.ones_like(ids))
            frozen.update(int(row) for row in torch.unique(rows).tolist() if row > 0)
        operator_name = str(spec["operator"]).upper()
        if operator_name == "OVERWRITE":
            operator_name = "DIRECT"
        operator = WriteOperator(operator_name, kappa=spec["kappa"], precision_init=spec["precision_init"],
                                 frozen_rows=tuple(sorted(frozen)))
        writer = ForwardWriter(reader.width, hidden_dim=values["writer"]["hidden_dim"],
                               orders=reader.orders, right_window=values["writer"]["right_window"],
                               operator=operator).to(device)
        return ForwardMemoryMethod(**common, writer=writer)
    if name == "empty":
        return EmptyMemoryMethod(**common)
    from .baselines import OnlineGradientMethod, FullContextMethod, BM25RetrievalMethod, EvidenceRidgeMethod
    if name in ("adam", "sgd", "read_normalized_sgd"):
        return OnlineGradientMethod(**common, kind=name, lr=spec["lr"], online_steps=spec["online_steps"])
    if name == "context":
        return FullContextMethod(**common)
    if name == "retrieval":
        return BM25RetrievalMethod(**common, top_k=spec["top_k"], k1=spec["k1"], b=spec["b"])
    if name == "engram_edit_style":
        return EvidenceRidgeMethod(**common, **{key: spec[key] for key in
                                  ("target_steps", "target_lr", "target_l2", "ridge_lambda", "reuse_penalty")})
    raise ValueError(f"unknown method: {name}")
