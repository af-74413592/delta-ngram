import pytest
import torch

from delta_ngram.alpha_memory import LazyDecayState
from delta_ngram.batching import BatchMemoryState
from delta_ngram.state import SparseMemoryState
from delta_ngram.qwen38_capacity_probe import expand_zero_capacity, select_capacity_frames
from delta_ngram.qwen38_seven_forward import state_payload, restore_state, TASKS


def seed():
    memory = BatchMemoryState(SparseMemoryState(torch.tensor([2, 7]),
        torch.arange(320, dtype=torch.float32).reshape(2, 160), torch.tensor([1., 3.])), 1, 20)
    state = LazyDecayState(memory, torch.tensor([4]), torch.zeros(1, dtype=torch.float64),
        torch.tensor([2, 4]), torch.zeros(2, dtype=torch.float64))
    return state_payload(state)


def test_capacity_retains_effective_values_clocks_and_seed_bytes():
    original = seed()
    padded, receipt = expand_zero_capacity(original, torch.tensor([1, 2, 5, 7, 10]), expected_rows=5)
    state = restore_state(padded)
    values, counts = state.lookup(torch.tensor([[2, 7, 1, 5, 10]]))
    assert torch.equal(values[0, :2], original['values'])
    assert torch.equal(counts[0, :2], original['counts'])
    assert not values[0, 2:].any() and not counts[0, 2:].any()
    assert state.block_clock.tolist() == [4]
    assert state.row_last_written_block.tolist() == [0, 2, 0, 4, 0]
    assert receipt['scientific_endpoint'] is False and receipt['zero_placeholder_rows'] == 3
    assert original['rows'].tolist() == [2, 7]


@pytest.mark.parametrize('rows', [[1, 2, 5], [1, 2, 2, 7], [0, 2, 7], [2, 7, 21]])
def test_capacity_rejects_invalid_or_missing_addresses(rows):
    with pytest.raises(ValueError):
        expand_zero_capacity(seed(), torch.tensor(rows), expected_rows=len(rows))


def test_frame_selection_has_all_domains_longest_and_unique_sources():
    data = {task: [{'task':task, 'source_row_index':i, 'input_ids':[1]*(i%100+1)}
                   for i in range(5000)] for task in TASKS}
    chosen = select_capacity_frames(data)
    assert len(chosen) == len({(r['task'],r['source_row_index']) for r in chosen}) == 16
    assert {r['task'] for r in chosen if len(r['input_ids']) == 100} == set(TASKS)
