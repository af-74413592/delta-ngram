"""One CPU-only behavioral smoke test for the minimal interactive chat."""
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import delta_ngram.chat as chat
from delta_ngram.interfaces import WriteResult
from delta_ngram.state import SparseMemoryState


def test_chat_write_modes_read_only_queries_reset_and_quit(monkeypatch):
    events, outputs, reads = [], [], []

    def snapshot(state):
        return tuple(tensor.clone() for tensor in (state.rows, state.values, state.counts))

    # Unchanged zero values with advancing counts also exercise a valid beta=0
    # write: statistics must count observed rows, not only changed value vectors.
    written_states = [
        SparseMemoryState(torch.tensor([2, 5]), torch.zeros(2, 2), torch.tensor([1., 1.])),
        SparseMemoryState(torch.tensor([2, 5, 9]), torch.zeros(3, 2), torch.tensor([1., 2., 1.])),
        SparseMemoryState(torch.tensor([2, 5, 9]), torch.zeros(3, 2), torch.tensor([1., 2., 2.])),
    ]

    class FakeMethod:
        def __init__(self):
            self.writes, self.empty_states = [], []

        def eval(self):
            return self

        def empty_state(self):
            state = SparseMemoryState.empty(2)
            self.empty_states.append(state)
            return state

        def memory(self, state):
            return state

        def write(self, state, evidence):
            assert not torch.is_grad_enabled()
            result = written_states[len(self.writes)]
            self.writes.append((state, evidence, result))
            events.append(("write", evidence))
            return WriteResult(result)

    method = FakeMethod()

    def generate(active_method, state, question, max_new_tokens):
        assert active_method is method and max_new_tokens == 7
        assert not torch.is_grad_enabled()
        reads.append((question, state, snapshot(state)))
        events.append(("read", question))
        return " answer<" + question + ">\ncontinuation"

    monkeypatch.setattr(chat, "greedy_generate", generate)
    commands = iter([
        "default-one", "default-two", "/write on", "stored fact",
        "/write off", "recall-one", "/write all", "second fact",
        "/write off", "recall-two", "/reset", "fresh", "/quit", "unreachable",
    ])
    chat.chat_loop(method, max_new_tokens=7, input_fn=lambda prompt: next(commands),
                   output_fn=outputs.append)

    assert next(commands) == "unreachable"
    assert [question for question, _, _ in reads] == [
        "default-one", "default-two", "stored fact", "recall-one",
        "second fact", "recall-two", "fresh",
    ]  # Every read receives just the current input, never a transcript.
    assistant_evidence = "Assistant said:  answer<second fact>\ncontinuation"
    assert events == [
        ("read", "default-one"), ("read", "default-two"),
        ("read", "stored fact"), ("write", "stored fact"),
        ("read", "recall-one"), ("read", "second fact"),
        ("write", "second fact"), ("write", assistant_evidence),
        ("read", "recall-two"), ("read", "fresh"),
    ]
    assert len(method.empty_states) == 2
    assert all(reads[index][1] is method.empty_states[0] for index in (0, 1, 2))
    assert reads[3][1] is reads[4][1] is written_states[0]
    assert reads[5][1] is written_states[2]
    assert reads[6][1] is method.empty_states[1]
    assert reads[6][1].rows.numel() == 0
    assert method.writes[0][0] is reads[2][1]
    assert method.writes[1][0] is written_states[0]
    assert method.writes[2][0] is written_states[1]
    for _, state, before in reads:
        assert all(torch.equal(actual, original) for actual, original in
                   zip((state.rows, state.values, state.counts), before))
    assert [line for line in outputs if line.startswith("[write]")] == [
        "[write] rows=2 new=2", "[write] rows=2 new=1", "[write] rows=1 new=0",
    ]
    assert "Assistant>  answer<second fact>\ncontinuation" in outputs
