from __future__ import annotations

import unittest

import numpy as np

from src.pipeline.baselines.preprocess import load_aligned_edges
from src.pipeline.baselines.preprocess import (
    egraphsage_edge_features,
    te_g_sage_edge_features,
)
from src.pipeline.common.datasets import get_dataset_config

# NOTE: import ONLY load_aligned_edges here. The featurizers do not exist yet, and
# a module-level import of a missing name fails COLLECTION -- pytest would report an
# ImportError for the whole file instead of running the alignment test. Step 5 adds
# the featurizer import alongside the tests that need it.


class AlignmentTest(unittest.TestCase):
    def test_csv_rows_align_with_graph_edges(self) -> None:
        for key in ("unsw_nb15", "ton_iot"):
            with self.subTest(dataset=key):
                config = get_dataset_config(key)
                df, data = load_aligned_edges(config)
                self.assertEqual(len(df), int(data.edge_index.shape[1]))
                label_names = list(config.label_names)
                csv_labels = np.array(
                    [label_names.index(a.strip()) for a in df["Attack"]]
                )
                np.testing.assert_array_equal(
                    csv_labels, data.edge_label.cpu().numpy()
                )


class EGraphSAGEFeatureTest(unittest.TestCase):
    def test_five_standard_scaled_columns(self) -> None:
        config = get_dataset_config("unsw_nb15")
        df, _ = load_aligned_edges(config)
        feats = egraphsage_edge_features(df)
        self.assertEqual(feats.shape, (len(df), 5))
        self.assertEqual(feats.dtype, np.float32)
        np.testing.assert_allclose(feats.mean(axis=0), 0.0, atol=1e-5)
        np.testing.assert_allclose(feats.std(axis=0), 1.0, atol=1e-5)


class TEGSageFeatureTest(unittest.TestCase):
    def test_onehot_blocks_and_rare_folding(self) -> None:
        config = get_dataset_config("unsw_nb15")
        df, _ = load_aligned_edges(config)
        feats, names = te_g_sage_edge_features(df, rare_min_freq=50)
        self.assertEqual(feats.shape[0], len(df))
        self.assertEqual(feats.shape[1], len(names))
        onehot = [n for n in names if n.startswith("most_common_")]
        self.assertTrue(onehot, "expected one-hot categorical columns")
        # One-hot blocks are indicator columns.
        block = feats[:, [names.index(n) for n in onehot]]
        self.assertTrue(set(np.unique(block)).issubset({0.0, 1.0}))

    def test_rare_categories_collapse_at_our_scale(self) -> None:
        config = get_dataset_config("unsw_nb15")
        df, _ = load_aligned_edges(config)
        strict, strict_names = te_g_sage_edge_features(df, rare_min_freq=50)
        loose, loose_names = te_g_sage_edge_features(df, rare_min_freq=2)
        self.assertLess(
            len([n for n in strict_names if n.startswith("most_common_port")]),
            len([n for n in loose_names if n.startswith("most_common_port")]),
            "rare_min_freq=50 must collapse more port categories than =2",
        )
