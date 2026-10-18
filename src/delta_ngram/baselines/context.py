"""Full evidence history and deterministic lexical BM25 retrieval baselines."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import re

from ..interfaces import MemoryMethod, WriteResult
from ..runtime import token_ids
from ..state import SparseMemoryState


@dataclass(frozen=True)
class TextHistoryState:
    evidence: tuple[str, ...] = ()


def lexical_tokens(text: str) -> list[str]:
    return re.findall(r"\w+", text.casefold(), flags=re.UNICODE)


def bm25_indices(documents: tuple[str, ...], query: str, top_k: int,
                 k1: float = 1.5, b: float = 0.75) -> list[int]:
    """Rank by BM25, break ties by arrival index, then render chronologically."""
    if top_k < 1 or k1 <= 0 or not 0 <= b <= 1:
        raise ValueError("BM25 needs top_k>=1, k1>0, and b in [0,1]")
    if not documents:
        return []
    counts = [Counter(lexical_tokens(text)) for text in documents]
    lengths = [sum(tokens.values()) for tokens in counts]
    average = sum(lengths) / len(lengths)
    terms = set(lexical_tokens(query))
    frequencies = {term: sum(term in document for document in counts) for term in terms}
    scores = []
    for index, tokens in enumerate(counts):
        score = 0.0
        for term in sorted(terms):
            frequency = tokens[term]
            if not frequency:
                continue
            document_frequency = frequencies[term]
            inverse = math.log(1.0 + (len(documents) - document_frequency + 0.5)
                               / (document_frequency + 0.5))
            length_ratio = lengths[index] / average if average else 0.0
            score += inverse * frequency * (k1 + 1.0) / (frequency + k1 * (1.0 - b + b * length_ratio))
        scores.append(score)
    ranked = sorted(range(len(documents)), key=lambda index: (-scores[index], index))[:top_k]
    return sorted(ranked)


class FullContextMethod(MemoryMethod):
    name = "context"
    supports_row_diagnostics = False

    def empty_state(self):
        return TextHistoryState()

    def write(self, state, evidence: str):
        ids = token_ids(self.tokenizer, evidence, self.device)
        if ids.shape[1] > self.max_evidence_tokens:
            raise ValueError("Evidence exceeds max_evidence_tokens; no silent truncation")
        result = TextHistoryState((*state.evidence, evidence))
        return WriteResult(result, {"evidence_tokens": int(ids.numel()),
                                    "history_blocks": len(result.evidence), **self.size_report(result)})

    def memory(self, state):
        return SparseMemoryState.empty(self.reader.width, self.device)

    def selected_evidence(self, state, question):
        return state.evidence

    def render_prompt(self, state, question):
        documents = self.selected_evidence(state, question)
        return "\n".join(documents) + "\n" + question if documents else question

    def counterfactual(self, reference, alternative):
        return TextHistoryState(tuple(alternative.evidence))

    def replace_memory(self, state, memory):
        raise TypeError("Text-context methods do not have writable row memory")

    def size_report(self, state):
        size = sum(len(text.encode("utf-8")) for text in state.evidence)
        return {"state_rows": 0, "table_tensor_bytes": 0, "optimizer_tensor_bytes": 0,
                "text_utf8_bytes": size, "persistent_tensor_bytes": 0,
                "persistent_state_bytes": size}

    def state_bytes(self, state):
        return self.size_report(state)["persistent_state_bytes"]


class BM25RetrievalMethod(FullContextMethod):
    name = "retrieval"

    def __init__(self, reader, tokenizer, device, max_evidence_tokens=128, *,
                 top_k=2, k1=1.5, b=0.75):
        super().__init__(reader, tokenizer, device, max_evidence_tokens)
        if top_k < 1 or k1 <= 0 or not 0 <= b <= 1:
            raise ValueError("BM25 needs top_k>=1, k1>0, and b in [0,1]")
        self.top_k, self.k1, self.b = top_k, k1, b

    def selected_evidence(self, state, question):
        indices = bm25_indices(state.evidence, question, self.top_k, self.k1, self.b)
        return tuple(state.evidence[index] for index in indices)


__all__ = ["FullContextMethod", "BM25RetrievalMethod", "TextHistoryState", "bm25_indices"]
