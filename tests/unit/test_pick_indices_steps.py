import torch

from gonka_poc.poc.decode_random import (
    random_pick_indices_decode,
    random_pick_indices_decode_steps,
)


def test_batched_steps_match_per_step_calls():
    """Пачка индексов для эмиссии траектории (encode_sph_slices) побитово, включая
    порядок, равна пошаговым вызовам эталона — иначе артефакты несовместимы."""
    cpu = torch.device("cpu")
    bh, pk, nonce, dim, k = "0xdeadbeef", "pk1", 4242, 256, 16
    steps = list(range(1 + 256))  # префилл + decode-шаги, как в проде
    batched = random_pick_indices_decode_steps(bh, pk, nonce, dim, k, cpu, steps)
    assert batched.shape == (len(steps), k)
    for s in steps:
        single = random_pick_indices_decode(bh, pk, [nonce], dim, k, cpu, step=s)[0]
        assert torch.equal(batched[s], single), f"step {s}: порядок/набор индексов разошёлся"
