"""Phase 4.4 correctness gate for `BiasedGATv2Conv`.

The non-negotiable invariant: with a zero (or absent) attention bias,
`BiasedGATv2Conv` must reproduce stock `GATv2Conv` bit-for-bit. If this
fails, every downstream feedback-loop result is meaningless — a nonzero
"bias" that merely reflects a different base layer proves nothing.

We also assert the converse: a nonzero bias *must* change the output,
otherwise the injection path is inert (the failure mode that got the
previous Step 4 attempt reverted).

Runs under pytest, or standalone: `python -m tests.test_biased_gat`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from torch_geometric.nn import GATv2Conv

from src.models.feedback_classifier import BiasedGATv2Conv

TOLERANCE = 1e-5
NUM_NODES = 20
NUM_EDGES = 50
IN_DIM = 10
OUT_DIM = 4
EDGE_DIM = 5
HEADS = 8
SEED = 0


def _synthetic_graph(seed: int = SEED):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(NUM_NODES, IN_DIM, generator=g)
    edge_index = torch.randint(0, NUM_NODES, (2, NUM_EDGES), generator=g)
    edge_attr = torch.randn(NUM_EDGES, EDGE_DIM, generator=g)
    return x, edge_index, edge_attr


def _paired_layers():
    """A stock and a biased layer sharing identical weights."""
    torch.manual_seed(SEED)
    stock = GATv2Conv(
        IN_DIM, OUT_DIM, heads=HEADS, edge_dim=EDGE_DIM, add_self_loops=False
    ).eval()
    biased = BiasedGATv2Conv(
        IN_DIM, OUT_DIM, heads=HEADS, edge_dim=EDGE_DIM, add_self_loops=False
    ).eval()
    biased.load_state_dict(stock.state_dict())
    return stock, biased


def test_zero_bias_matches_stock() -> None:
    """bias=None and bias=zeros must both equal stock GATv2Conv."""
    x, edge_index, edge_attr = _synthetic_graph()
    stock, biased = _paired_layers()

    with torch.no_grad():
        ref = stock(x, edge_index, edge_attr)
        out_none = biased(x, edge_index, edge_attr=edge_attr, edge_attn_bias=None)
        zeros = torch.zeros(NUM_EDGES, HEADS)
        out_zero = biased(x, edge_index, edge_attr=edge_attr, edge_attn_bias=zeros)
        zeros_b = torch.zeros(NUM_EDGES, 1)  # broadcast form
        out_zero_b = biased(x, edge_index, edge_attr=edge_attr, edge_attn_bias=zeros_b)

    assert torch.allclose(ref, out_none, atol=TOLERANCE), (
        f"bias=None diverges from stock: max|Δ|={float((ref - out_none).abs().max()):.2e}"
    )
    assert torch.allclose(ref, out_zero, atol=TOLERANCE), (
        f"bias=zeros diverges from stock: max|Δ|={float((ref - out_zero).abs().max()):.2e}"
    )
    assert torch.allclose(ref, out_zero_b, atol=TOLERANCE), (
        f"broadcast bias=zeros diverges: max|Δ|={float((ref - out_zero_b).abs().max()):.2e}"
    )


def test_nonzero_bias_changes_output() -> None:
    """A non-uniform bias must actually move the output (injection is live).

    Note: a *uniform* bias is deliberately a no-op — softmax is invariant to
    a constant added across a node's incoming edges — so the bias must vary
    per edge to shift the attention distribution.
    """
    x, edge_index, edge_attr = _synthetic_graph()
    _, biased = _paired_layers()

    g = torch.Generator().manual_seed(SEED + 1)
    bias = torch.randn(NUM_EDGES, 1, generator=g) * 1.5  # per-edge, ~1.5 nats
    with torch.no_grad():
        ref = biased(x, edge_index, edge_attr=edge_attr, edge_attn_bias=None)
        out = biased(x, edge_index, edge_attr=edge_attr, edge_attn_bias=bias)

    delta = float((ref - out).abs().max())
    assert delta > 1e-3, f"nonzero bias did not change output (max|Δ|={delta:.2e})"


def test_self_loops_guard() -> None:
    """Supplying a bias with add_self_loops=True must raise (misalignment)."""
    x, edge_index, edge_attr = _synthetic_graph()
    torch.manual_seed(SEED)
    layer = BiasedGATv2Conv(
        IN_DIM, OUT_DIM, heads=HEADS, edge_dim=EDGE_DIM, add_self_loops=True
    )
    try:
        layer(
            x,
            edge_index,
            edge_attr=edge_attr,
            edge_attn_bias=torch.zeros(NUM_EDGES, 1),
        )
    except ValueError:
        return
    raise AssertionError("expected ValueError when add_self_loops=True with a bias")


def main() -> None:
    test_zero_bias_matches_stock()
    print("PASS  zero/None bias == stock GATv2Conv (< 1e-5)")
    test_nonzero_bias_changes_output()
    print("PASS  nonzero bias moves the output")
    test_self_loops_guard()
    print("PASS  add_self_loops guard raises")
    print("\nAll Phase 4.4 equivalence checks passed.")


if __name__ == "__main__":
    main()
