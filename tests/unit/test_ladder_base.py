"""Seeded-routing ladder base is a per-model consensus parameter: 100 on
DeepSeek-V4, 0 elsewhere (MiniMax cells were frozen with 0)."""
import torch

from gonka_poc.poc import decode_random as dr


def test_ladder_base_by_model():
    assert dr.set_ladder_base_for_model("deepseek_v4") == 100
    assert dr.set_ladder_base_for_model("minimax_m2") == 0
    assert dr.set_ladder_base_for_model(None) == 0


def test_forced_logits_follow_base():
    seed = torch.tensor([7], dtype=torch.int64)
    dr.set_ladder_base_for_model("minimax_m2")
    lo = dr._forced_logits(seed, 16, 4, torch.device("cpu"))
    dr.set_ladder_base_for_model("deepseek_v4")
    hi = dr._forced_logits(seed, 16, 4, torch.device("cpu"))
    dr.set_ladder_base_for_model(None)
    chosen = lo[0] > -1.0e3
    assert torch.equal(chosen, hi[0] > -1.0e3)          # same seeded set
    assert lo[0][chosen].max().item() == 4.0            # ladder 4..1
    assert hi[0][chosen].max().item() == 104.0          # ladder 104..101


def test_forced_logits_take_the_base_from_a_tensor():
    """Traced code passes the base as a tensor (graph input), never a Python int:
    a Python int is baked into the compiled graph and survives in vLLM's compile
    cache across boots with a different base."""
    import torch
    from gonka_poc.poc import decode_random as dr

    seed = torch.tensor([7, 300], dtype=torch.int64)
    base_t = torch.tensor(100.0, dtype=torch.float32)
    got = dr._forced_logits(seed, 16, 4, torch.device("cpu"), base_t)
    ref = dr._forced_logits(seed, 16, 4, torch.device("cpu"))  # global base (0)
    top_t = torch.topk(got, 4).values
    top_r = torch.topk(ref, 4).values
    assert torch.equal(top_t, top_r + 100.0)
    assert torch.equal(torch.topk(got, 4).indices, torch.topk(ref, 4).indices)
    assert float(dr.ladder_base_tensor(torch.device("cpu"))) == float(dr.ladder_base())
