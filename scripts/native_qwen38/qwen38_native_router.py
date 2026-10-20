"""Use the native training top-k order in both gradient modes.

The pinned MCore router normally chooses sorted=torch.is_grad_enabled().
Its existing top-k callback lets us fix that single choice without changing
global torch functions, router scores, expert selection, or aux-loss logic.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import inspect
from pathlib import Path


MOE_UTILS_SHA256 = "629c48538f2d69a2e110ec5f5b5925d20ccf541a7e960adff43ebb61d4528713"
PROTOCOL = "qwen38_native_router_sorted_topk_v1"


class SortedTopK:
    """Stateless callback; no routing history is recorded or replayed."""

    def __init__(self):
        self.calls = 0

    def get_replay_topk(self, scores, topk, num_groups, group_topk, compute_topk):
        import torch
        if group_topk:
            raise ValueError("sorted policy does not support group-limited routing")
        self.calls += 1
        return torch.topk(scores, k=topk, dim=1, sorted=True)


@contextmanager
def _install_sorted_topk(routers, identity):
    if not routers:
        raise ValueError("no native routers found")
    for name, router in routers:
        if router.training or router.router_replay is not None:
            raise ValueError(f"{name}: requires eval mode and no existing replay")
        if router.config.moe_router_fusion or router.config.moe_router_group_topk:
            raise ValueError(f"{name}: requires unfused, non-grouped native routing")
    policies = [(name, router, SortedTopK()) for name, router in routers]
    report = {"identity": identity, "routers": [], "restored": False}
    try:
        for _, router, policy in policies:
            router.router_replay = policy
        yield report
    finally:
        for name, router, policy in policies:
            if router.router_replay is not policy:
                raise RuntimeError(f"{name}: routing policy changed inside scope")
        for name, router, policy in policies:
            router.router_replay = None
            report["routers"].append({"path": name, "topk_calls": policy.calls})
        report["restored"] = all(router.router_replay is None for _, router, _ in policies)


@contextmanager
def native_router_dispatch(model):
    """Install sorted=True on every native router in one local model shard."""
    from megatron.core.transformer.moe import moe_utils
    from mcore_bridge.model.modules.topk_router import TopKRouter
    source = Path(inspect.getfile(moe_utils))
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    if source_sha != MOE_UTILS_SHA256:
        raise ValueError("MCore routing source differs from the audited version")
    routers = [(name, module) for name, module in model.named_modules()
               if isinstance(module, TopKRouter)]
    identity = {
        "protocol": PROTOCOL, "moe_utils_sha256": source_sha,
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "topk_sorted": True,
        "scope": "native top-k callback only; no score or expert formula changes",
        "training_topk_order_preserved": True,
        "routing_history_replayed": False,
    }
    with _install_sorted_topk(routers, identity) as report:
        yield report
