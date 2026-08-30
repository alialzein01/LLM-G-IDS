from __future__ import annotations

import unittest

import torch

from src.models.baselines.e_graphsage import EGraphSAGE


def _toy_graph() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    edge_index = torch.tensor([[0, 1, 2, 0], [1, 2, 0, 2]], dtype=torch.long)
    edge_attr = torch.randn(4, 5)
    x = torch.randn(3, 10)
    return x, edge_index, edge_attr


class EGraphSAGEShapeTest(unittest.TestCase):
    def test_emits_one_logit_row_per_edge(self) -> None:
        x, edge_index, edge_attr = _toy_graph()
        model = EGraphSAGE(edge_dim=5, num_classes=10)
        logits = model(x, edge_index, edge_attr)
        self.assertEqual(tuple(logits.shape), (4, 10))

    def test_ones_init_ignores_node_features(self) -> None:
        """The published model sets x_v = {1,...,1}; data.x must not reach it."""
        x, edge_index, edge_attr = _toy_graph()
        model = EGraphSAGE(edge_dim=5, num_classes=10, node_init="ones")
        model.eval()
        with torch.no_grad():
            a = model(x, edge_index, edge_attr)
            b = model(torch.randn_like(x) * 100.0, edge_index, edge_attr)
        torch.testing.assert_close(a, b)

    def test_node_feature_variant_consumes_node_features(self) -> None:
        """The labelled variant must actually depend on data.x."""
        x, edge_index, edge_attr = _toy_graph()
        model = EGraphSAGE(
            edge_dim=5, num_classes=10, node_init="node_features", node_feat_dim=10
        )
        model.eval()
        with torch.no_grad():
            a = model(x, edge_index, edge_attr)
            b = model(torch.randn_like(x) * 100.0, edge_index, edge_attr)
        self.assertFalse(torch.allclose(a, b))
