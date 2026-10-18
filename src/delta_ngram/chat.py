"""Minimal interactive queries and optional writes through the existing method."""
from __future__ import annotations

import torch

from .checkpoints import load_checkpoint, restore_writer
from .runtime import greedy_generate
from .train import create_runtime


COMMANDS = "Commands: /write off | /write on | /write all | /reset | /quit"


@torch.no_grad()
def chat_loop(method, *, max_new_tokens, input_fn=input, output_fn=print):
    """Answer from the current state, then optionally write this turn's text."""
    method.eval()
    state = method.empty_state()
    mode = "off"
    output_fn("Write mode: off. " + COMMANDS)

    def write(current, text):
        before = method.memory(current)
        result = method.write(current, text)
        after = method.memory(result.state)
        _, old_counts = before.lookup(after.rows)
        written = int((after.counts > old_counts).sum().item())
        new = int((~torch.isin(after.rows, before.rows)).sum().item())
        output_fn(f"[write] rows={written} new={new}")
        return result.state

    while True:
        try:
            text = input_fn("You> ")
        except (EOFError, KeyboardInterrupt):
            output_fn("")
            return
        command = text.strip()
        if not command:
            continue
        if command == "/quit":
            return
        if command == "/reset":
            state = method.empty_state()
            output_fn("Memory reset.")
            continue
        if command in ("/write off", "/write on", "/write all"):
            mode = command.split()[1]
            output_fn(f"Write mode: {mode}.")
            continue
        if command.startswith("/"):
            output_fn(COMMANDS)
            continue

        answer = greedy_generate(method, state, text, max_new_tokens)
        output_fn("Assistant> " + answer)
        if mode in ("on", "all"):
            state = write(state, text)
        if mode == "all":
            state = write(state, "Assistant said: " + answer)


def run_chat(config, checkpoint):
    payload = load_checkpoint(checkpoint)
    method, args = create_runtime(config)
    restore_writer(method, payload, config)
    method.writer.requires_grad_(False)
    chat_loop(method, max_new_tokens=args.max_new_tokens)
