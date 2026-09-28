import numpy as np
import torch

from sih.stage1.mesh import icosahedron, refine
from sih.stage2.diffusion import K, physics_losses, violation_index


def test_icosahedral_refinement_node_count():
    v, f = icosahedron()
    for level in range(1, 4):
        v, f = refine(v, f)
        assert len(v) == 10 * 4 ** level + 2
        assert np.allclose(np.linalg.norm(v, axis=1), 1.0)


def test_conservation_loss_is_zero_when_block_means_match():
    coarse = torch.rand(2, 1, 4, 4) * 20
    fine = coarse.repeat_interleave(K, -1).repeat_interleave(K, -2)
    cons, conv = physics_losses(fine, coarse, torch.zeros_like(fine))
    assert torch.allclose(cons, torch.zeros(2), atol=1e-5)
    assert torch.allclose(conv, torch.zeros(2))


def test_convergence_penalty_hits_rain_in_divergent_air_only():
    rain = torch.full((1, 1, 12, 12), 30.0)
    coarse = torch.nn.functional.avg_pool2d(rain, K)
    _, conv_div = physics_losses(rain, coarse, torch.full_like(rain, 10.0))
    _, conv_con = physics_losses(rain, coarse, torch.full_like(rain, -10.0))
    assert conv_div.item() > 0 and conv_con.item() == 0


def test_violation_index():
    mm = np.array([[20.0, 20.0], [0.0, 0.0]])
    mfd = np.array([[1.0, -1.0], [1.0, 1.0]])
    assert violation_index(mm, mfd) == 0.5
