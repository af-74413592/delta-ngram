"""Shared tokenization, generation, loss, timing, and model loading."""
from __future__ import annotations

from pathlib import Path
import json
import string
import time
from typing import Any

import torch
import torch.nn.functional as F


def state_guard(method, state):
    memory = method.memory(state)
    return tuple(value.clone() if value is not None else None
                 for value in (memory.rows, memory.values, memory.counts, memory.precision))


def assert_read_only(method, state, before):
    memory = method.memory(state)
    for old, new in zip(before, (memory.rows, memory.values, memory.counts, memory.precision)):
        if (old is None) != (new is None) or (old is not None and not torch.equal(old, new)):
            raise RuntimeError("A read-only query mutated explicit memory")


def answer_loss(method, state, question: str, answer: str):
    prompt = method.render_prompt(state, question)
    ids, mask, labels = answer_batch(method.tokenizer, prompt, answer, method.device)
    output = method.reader(ids, mask, method.memory(state))
    logits = output.logits[:, :-1, :].float()
    shifted_labels = labels[:, 1:]
    if not shifted_labels.ne(-100).any():
        raise ValueError("No answer tokens remain after the causal label shift")
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                           shifted_labels.reshape(-1), ignore_index=-100)


@torch.no_grad()
def greedy_generate(method, state, question: str, max_new_tokens: int):
    """One cache-free greedy decoder for all methods, including text baselines."""
    ids = token_ids(method.tokenizer, method.render_prompt(state, question), method.device)
    before = state_guard(method, state)
    generated = []
    eos_ids = getattr(method.tokenizer, "eos_token_id", None)
    eos_ids = set(eos_ids if isinstance(eos_ids, (tuple, list)) else [eos_ids])
    for _ in range(max_new_tokens):
        output = method.reader(ids, torch.ones_like(ids), method.memory(state))
        token = int(output.logits[0, -1].argmax().item())
        if token in eos_ids:
            break
        generated.append(token)
        ids = torch.cat((ids, ids.new_tensor([[token]])), dim=1)
    assert_read_only(method, state, before)
    return method.tokenizer.decode(generated, skip_special_tokens=True)

class CharacterTokenizer:
    """Offline tokenizer for the CPU-only random-model plumbing smoke test."""

    pad_token_id = 0
    eos_token_id = 1

    def __init__(self) -> None:
        self.characters = "\n" + string.printable.replace("\n", "")
        self.mapping = {character: index + 2 for index, character in enumerate(self.characters)}
        self.vocab_size = len(self.characters) + 2

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [self.mapping[character] for character in text]

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return "".join(self.characters[index - 2] for index in ids if index >= 2)


def token_ids(tokenizer: Any, text: str, device: torch.device) -> torch.Tensor:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids:
        raise ValueError("Empty token sequence")
    return torch.tensor([ids], dtype=torch.long, device=device)


def answer_batch(tokenizer: Any, question: str, answer: str, device: torch.device):
    """Require exact prefix token alignment; never guess the answer boundary."""
    prefix = tokenizer.encode(question, add_special_tokens=False)
    full = tokenizer.encode(question + answer, add_special_tokens=False)
    if not prefix or full[:len(prefix)] != prefix or len(full) <= len(prefix):
        raise ValueError("Question/answer tokenization is not an exact prefix extension")
    ids = torch.tensor([full], dtype=torch.long, device=device)
    labels = torch.full_like(ids, -100)
    labels[:, len(prefix):] = ids[:, len(prefix):]
    return ids, torch.ones_like(ids), labels


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed(call, device: torch.device):
    synchronize(device)
    start = time.perf_counter()
    result = call()
    synchronize(device)
    return result, time.perf_counter() - start


def normalized_answer(text: str) -> str:
    # Score the first answer word only, never search for a favorable substring.
    words = text.strip().split()
    return words[0].strip(".,!?;:\"'").casefold() if words else ""


def normalized_first_line(text: str) -> str:
    lines = text.strip().splitlines()
    return lines[0].strip().casefold() if lines else ""


def mean(values: list[float | bool]) -> float | None:
    return sum(values) / len(values) if values else None


def log_event(path: Path, value: dict) -> None:
    line = json.dumps(value, ensure_ascii=False, allow_nan=False)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
        handle.flush()
    print(line, flush=True)


def save_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def load_model_and_tokenizer(args, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if args.smoke:
        from transformers import LlamaConfig, LlamaForCausalLM
        tokenizer = CharacterTokenizer()
        model = LlamaForCausalLM(LlamaConfig(
            vocab_size=tokenizer.vocab_size, hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
            max_position_embeddings=512, pad_token_id=0, eos_token_id=1,
            attention_dropout=0.0, use_cache=False))
        model = model.to(device)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
        dtype = torch.float32 if device.type == "cpu" else (
            torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, revision=args.revision, torch_dtype=dtype, attn_implementation="sdpa").to(device)
    model.requires_grad_(False)
    model.eval()
    model.config.use_cache = False
    return model, tokenizer
