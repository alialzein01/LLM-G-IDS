from __future__ import annotations

import unittest

import pandas as pd

from src.pipeline.step2.knowledge_graph import (
    SEMANTIC_RELATIONS,
    add_relation_names,
    assert_label_free,
    build_nl_triples,
    discretize,
)


class KnowledgeGraphContractsTest(unittest.TestCase):
    def _sample(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "IPV4_SRC_ADDR": ["10.0.0.1", "10.0.0.1", "10.0.0.2"],
                "IPV4_DST_ADDR": ["10.0.0.3", "10.0.0.4", "10.0.0.5"],
                "Attack": ["Benign", "ddos", "ddos"],
                "flow_count": [1, 5, 10],
                "total_bytes": [100, 5000, 20000],
                "avg_duration": [10.0, 50.0, 100.0],
                "most_common_protocol": [6, 17, 6],
                "most_common_port": [80, 53, 443],
            }
        )

    def test_semantic_relation_mapping_is_structured_only(self) -> None:
        df = add_relation_names(self._sample())
        self.assertEqual(df.loc[0, "relation_name"], SEMANTIC_RELATIONS["Benign"])
        self.assertEqual(df.loc[1, "relation_name"], SEMANTIC_RELATIONS["ddos"])

    def test_discretization_is_per_attack_class(self) -> None:
        df = discretize(self._sample())
        for col in ("avg_bytes_level", "flow_count_level", "avg_duration_level"):
            self.assertTrue(set(df[col]).issubset({"low", "medium", "high"}))
            self.assertFalse(df[col].isna().any())

    def test_natural_language_triples_are_label_free(self) -> None:
        df = add_relation_names(discretize(self._sample()))
        sentences = build_nl_triples(df)
        assert_label_free(sentences, {"Benign", "ddos"})
        joined = "\n".join(sentences).lower()
        self.assertNotIn("ddos", joined)
        self.assertNotIn("communicated with", joined)


if __name__ == "__main__":
    unittest.main()
