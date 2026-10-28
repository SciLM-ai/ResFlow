"""The tiled sliding-neighbourhood attention must equal FlexAttention with the same mask."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from resflow.models import dit3d as D  # noqa: E402

flex = pytest.importorskip('torch.nn.attention.flex_attention')


@pytest.mark.parametrize('grid, radius, tile', [((20, 18, 16), (5, 4, 12), (8, 8)),
                                                ((13, 29, 16), (12, 12, 12), (8, 8)),
                                                ((16, 16, 16), (3, 2, 1), (4, 8)),
                                                ((9, 7, 5), (12, 12, 12), (8, 8))])
def test_tiled_matches_flex(grid, radius, tile):
    torch.manual_seed(0)
    n = grid[0] * grid[1] * grid[2]
    q, k, v = (torch.randn(2, 3, n, 16, dtype=torch.float64) for _ in range(3))
    bm = flex.create_block_mask(D.neighbourhood_mask_mod(grid, radius), None, None, n, n,
                                device='cpu', _compile=False)
    ref = flex.flex_attention(q, k, v, block_mask=bm)
    out = D.tiled_neighbourhood_attention(q, k, v, grid, radius, tile=tile, max_scores=5000)
    assert torch.allclose(out, ref, atol=1e-12)
