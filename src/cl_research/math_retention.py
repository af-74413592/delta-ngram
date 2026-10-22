"""Locked MATH-500 retention data and final-answer scoring primitives.

No training or model imports belong here. Gold solutions never enter prompts.
Symbolic verification runs in a disposable CPU worker with an external deadline.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import importlib.metadata
import json
import multiprocessing
import re
import time

REVISION = "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"
DATA_URL = f"https://huggingface.co/datasets/HuggingFaceH4/MATH-500/resolve/{REVISION}/test.jsonl"
MATH_VERIFY_VERSION = "0.9.0"
PROBE_COUNT = 64
SEED = 17
MAX_NEW_TOKENS = 1024
PROMPT_TEMPLATE = (
    "Solve the following mathematics problem. Show your reasoning, then end with "
    "your final answer in \\boxed{{...}}.\n\nProblem:\n{problem}\n\nSolution:\n"
)


def digest(value):
    return hashlib.sha256(value).hexdigest()


def json_digest(value):
    return digest(json.dumps(value, sort_keys=True, ensure_ascii=False).encode())


def validate_records(rows, expected_count=500):
    if len(rows) != expected_count:
        raise ValueError(f"expected {expected_count} MATH rows, got {len(rows)}")
    required = ("unique_id", "problem", "answer", "subject", "level")
    ids = []
    for row in rows:
        if not all(key in row for key in required):
            raise ValueError("MATH row is missing required fields")
        if not all(isinstance(row[key], str) and row[key].strip() for key in required[:-1]):
            raise ValueError("MATH text fields must be nonempty strings")
        if type(row["level"]) is not int or row["level"] not in range(1, 6):
            raise ValueError("MATH level must be an integer from 1 to 5")
        ids.append(row["unique_id"])
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate MATH unique_id")
    return sorted(rows, key=lambda row: row["unique_id"])


def select_probe(rows, count=PROBE_COUNT, seed=SEED):
    """Hamilton proportional allocation across subject x level; hash within cells.

    Allocation uses only metadata and is independent of input ordering, answers,
    and all model results. Very small cells may receive zero slots.
    """
    if not 0 < count <= len(rows):
        raise ValueError("probe count must be positive and no larger than dataset")
    if len({row["unique_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate selection IDs")
    strata = defaultdict(list)
    for row in rows:
        strata[(row["subject"], row["level"])].append(row["unique_id"])
    total = len(rows)
    quotas = {key: count * len(ids) // total for key, ids in strata.items()}
    remainder_order = sorted(strata, key=lambda key: (-(count * len(strata[key]) % total), key))
    for key in remainder_order[:count - sum(quotas.values())]:
        quotas[key] += 1
    selected, allocation = [], []
    for key in sorted(strata):
        ids = sorted(strata[key], key=lambda uid: hashlib.sha256(f"math-probe-v1/{seed}/{uid}".encode()).digest())
        chosen = ids[:quotas[key]]
        selected.extend(chosen)
        allocation.append({"subject": key[0], "level": key[1], "population": len(ids), "quota": len(chosen)})
    return sorted(selected), allocation


def project_rows(rows, task_ids):
    ids = [row["task_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate task IDs")
    if not task_ids or len(set(task_ids)) != len(task_ids):
        raise ValueError("requested task IDs must be unique and nonempty")
    mapping = {row["task_id"]: row for row in rows}
    if not set(task_ids).issubset(mapping):
        raise ValueError("missing requested task IDs")
    return [mapping[task] for task in task_ids]


def _box_at(text, start):
    opening = text.find("{", start)
    depth = 0
    for position in range(opening, len(text)):
        char = text[position]
        preceding_slashes = 0
        previous = position - 1
        while previous >= 0 and text[previous] == "\\":
            preceding_slashes += 1
            previous -= 1
        if preceding_slashes % 2:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[opening + 1:position].strip(), position + 1
    return None, None


def extract_final_answer(completion):
    """Only a terminal box or a terminal explicitly labelled answer is eligible.

    Earlier correct-looking intermediate expressions are never candidates. If the
    final box is incomplete, do not fall back to an earlier complete box.
    """
    text = completion.strip()
    boxes = list(re.finditer(r"\\(?:boxed|fbox)\s*\{", text))
    labels = list(re.finditer(r"(?im)^\s*(?:\*\*)?(?:final\s+answer|answer)\s*(?:\*\*)?\s*(?::|is\s*:?)\s*", text))
    if boxes and (not labels or boxes[-1].start() > labels[-1].start()):
        answer, end = _box_at(text, boxes[-1].start())
        if answer is None:
            return None, "incomplete_final_box"
        # Closing math delimiters, punctuation, and Markdown are permitted only.
        tail = text[end:].replace(r"\]", "").replace(r"\)", "")
        if re.fullmatch(r"[\s$.*`]*", tail) is None:
            return None, "nonterminal_box"
        return (answer, "terminal_box") if answer else (None, "empty_final_box")
    if labels:
        answer = text[labels[-1].end():].strip()
        if not answer or "\n" in answer or len(answer) > 512:
            return None, "invalid_terminal_answer_line"
        # Keep one final expression; do not scan surrounding prose for matches.
        return answer.rstrip(".* "), "terminal_answer_line"
    return None, "no_terminal_answer"


def scoring_protocol():
    return {"version": 1, "math_verify": MATH_VERIFY_VERSION,
            "answer_extraction": "terminal_balanced_box_or_last_terminal_explicit_answer_line_v1",
            "parse": "one_extracted_answer_wrapped_as_latex;no_fallback;first_match",
            "verify": {"strict": True, "float_rounding": 6, "numeric_precision": 15},
            "deadline_seconds": 10, "timeout": "external_process_deadline;internal_timeouts_disabled"}


def _score_without_deadline(gold, answer):
    # Called only by the worker that has an external wall-clock deadline.
    from math_verify import LatexExtractionConfig, parse, verify
    if importlib.metadata.version("math-verify") != MATH_VERIFY_VERSION:
        raise RuntimeError(f"math-verify=={MATH_VERIFY_VERSION} required")
    config = [LatexExtractionConfig(try_extract_without_anchor=True, boxed_match_priority=-1)]

    def parse_one(value):
        value = value.strip()
        for left, right in (("$$", "$$"), (r"\[", r"\]"), (r"\(", r"\)"), ("$", "$")):
            if value.startswith(left) and value.endswith(right):
                value = value[len(left):-len(right)].strip()
                break
        return parse("$" + value + "$", extraction_config=config, fallback_mode="no_fallback",
                     extraction_mode="first_match", parsing_timeout=None)

    reference = parse_one(gold)
    if not reference:
        return {"status": "reference_parse_error", "correct": None, "parsed_answer": None}
    prediction = parse_one(answer)
    if not prediction:
        return {"status": "prediction_parse_failure", "correct": False, "parsed_answer": None}
    correct = bool(verify(reference, prediction, strict=True, float_rounding=6,
                          numeric_precision=15, timeout_seconds=None))
    return {"status": "scored", "correct": correct, "parsed_answer": [str(value) for value in prediction]}


def _worker(connection):
    try:
        # Import before accepting tasks; initialization has its own deadline.
        import math_verify  # noqa: F401
        actual = importlib.metadata.version("math-verify")
        if actual != MATH_VERIFY_VERSION:
            raise RuntimeError(f"expected math-verify {MATH_VERIFY_VERSION}, got {actual}")
        connection.send({"ready": True})
        while True:
            item = connection.recv()
            if item is None:
                return
            try:
                result = _score_without_deadline(*item)
            except Exception as exc:
                result = {"status": "checker_error", "correct": None,
                          "parsed_answer": None, "error": f"{type(exc).__name__}: {exc}"}
            connection.send(result)
    except EOFError:
        pass
    except Exception as exc:
        try:
            connection.send({"ready": False, "error": f"{type(exc).__name__}: {exc}"})
        except (BrokenPipeError, EOFError):
            pass
    finally:
        connection.close()


class MathScorer:
    """One reusable CPU subprocess; kill/recreate it on a hard scoring timeout."""
    def __init__(self, timeout_seconds=10):
        self.timeout_seconds = timeout_seconds
        self.process = self.connection = None

    def close(self):
        if self.process is not None:
            if self.process.is_alive():
                self.process.terminate()
            self.process.join(timeout=2)
            if self.process.is_alive():
                self.process.kill()
                self.process.join(timeout=2)
            self.connection.close()
            self.process = self.connection = None

    def _start(self):
        context = multiprocessing.get_context("spawn")
        self.connection, child = context.Pipe()
        self.process = context.Process(target=_worker, args=(child,), daemon=True)
        self.process.start()
        child.close()
        if not self.connection.poll(30):
            self.close()
            raise RuntimeError("math scorer initialization timed out")
        ready = self.connection.recv()
        if not ready.get("ready"):
            self.close()
            raise RuntimeError(ready.get("error", "math scorer failed to initialize"))

    def score(self, gold, completion):
        started = time.perf_counter()
        answer, extraction = extract_final_answer(completion)
        if answer is None:
            result = {"status": "prediction_parse_failure", "correct": False, "parsed_answer": None}
        else:
            if self.process is None:
                self._start()
            try:
                self.connection.send((gold, answer))
                if self.connection.poll(self.timeout_seconds):
                    result = self.connection.recv()
                else:
                    self.close()
                    result = {"status": "checker_timeout", "correct": None, "parsed_answer": None}
            except (BrokenPipeError, EOFError, OSError) as exc:
                self.close()
                result = {"status": "checker_error", "correct": None, "parsed_answer": None,
                          "error": f"{type(exc).__name__}: {exc}"}
        return dict(result, extracted_answer=answer, extraction=extraction,
                    scoring_elapsed_seconds=time.perf_counter() - started)


def result_metrics(rows, task_ids):
    selected = project_rows(rows, task_ids)
    unresolved = [row["task_id"] for row in selected if type(row.get("correct")) is not bool]
    correct = sum(row.get("correct") is True for row in selected)
    n = len(selected)
    return {"n": n, "correct_count": correct, "accuracy": correct / n if not unresolved else None,
            "unresolved_count": len(unresolved), "unresolved_ids": unresolved,
            "accuracy_lower_bound": correct / n, "accuracy_upper_bound": (correct + len(unresolved)) / n,
            "status_counts": dict(Counter(row["status"] for row in selected))}


def transition_counts(reference, current, task_ids):
    before, after = project_rows(reference, task_ids), project_rows(current, task_ids)
    unresolved = [left["task_id"] for left, right in zip(before, after)
                  if type(left.get("correct")) is not bool or type(right.get("correct")) is not bool]
    if unresolved:
        return {"n": len(task_ids), "lost": None, "gained": None, "accuracy_drop": None,
                "unresolved_pair_ids": unresolved}
    lost = [left["task_id"] for left, right in zip(before, after) if left["correct"] and not right["correct"]]
    gained = [left["task_id"] for left, right in zip(before, after) if not left["correct"] and right["correct"]]
    base_count = sum(row["correct"] for row in before)
    return {"n": len(task_ids), "lost": len(lost), "gained": len(gained), "lost_ids": lost,
            "gained_ids": gained, "accuracy_drop": (len(lost) - len(gained)) / len(task_ids),
            "reference_accuracy": base_count / len(task_ids),
            "current_accuracy": sum(row["correct"] for row in after) / len(task_ids),
            "reference_zero_floor": base_count == 0}
