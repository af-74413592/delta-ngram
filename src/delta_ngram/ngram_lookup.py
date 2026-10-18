"""External causal token hash, version 1; row zero is padding."""
import torch


def ngram_rows(ids, mask, buckets, orders=(2, 3)):
    """Return [batch, tokens, orders]; 0 is padding/incomplete n-gram.

    Only tokens at or before each position participate. The attention mask,
    not a token ID, distinguishes padding (EOS may also be the pad token).
    """
    result = []
    for slot, order in enumerate(orders):
        hashed = torch.zeros_like(ids)
        valid = torch.ones_like(ids, dtype=torch.bool)
        for lag in range(order - 1, -1, -1):
            shifted = torch.zeros_like(ids)
            present = torch.zeros_like(mask, dtype=torch.bool)
            if lag == 0:
                shifted, present = ids, mask.bool()
            elif lag < ids.shape[1]:
                shifted[:, lag:] = ids[:, :-lag]
                present[:, lag:] = mask[:, :-lag].bool()
            hashed = (hashed * 1000003 + shifted + 1).remainder(2147483647)
            valid &= present
        row = 1 + slot * buckets + hashed.remainder(buckets)
        result.append(torch.where(valid, row, 0))
    return torch.stack(result, dim=-1)
