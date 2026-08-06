from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "unsw_nb15_instructor_pipeline.ipynb"
README_PATH = ROOT / "notebooks" / "README.md"
COLAB_REQUIREMENTS_PATH = ROOT / "notebooks" / "colab-requirements.txt"
CHECKPOINT_MANIFEST_PATH = ROOT / "notebooks" / "drive-checkpoint-manifest.template.json"


def _load_notebook() -> dict:
    return json.loads(NOTEBOOK_PATH.read_text())


def _source(cell: dict) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else str(source)


def _all_source() -> str:
    return "\n".join(_source(cell) for cell in _load_notebook()["cells"])


class InstructorNotebookContractTest(unittest.TestCase):
    def test_notebook_and_delivery_docs_exist(self) -> None:
        self.assertTrue(NOTEBOOK_PATH.exists())
        self.assertTrue(README_PATH.exists())
        self.assertTrue(COLAB_REQUIREMENTS_PATH.exists())
        self.assertTrue(CHECKPOINT_MANIFEST_PATH.exists())

    def test_notebook_has_ordered_phase_tags(self) -> None:
        notebook = _load_notebook()
        self.assertEqual(notebook["nbformat"], 4)
        source = _all_source()
        expected = [
            "PHASE: setup",
            "PHASE: step0_preprocess",
            "PHASE: step1_graph_splits",
            "PHASE: step2_kg",
            "PHASE: step3_gnn",
            "PHASE: step4_llm",
            "PHASE: step5_agaf",
            "PHASE: step6_feedback",
            "PHASE: final_ladder",
        ]
        positions = [source.index(tag) for tag in expected]
        self.assertEqual(positions, sorted(positions))

    def test_notebook_uses_canonical_production_modules(self) -> None:
        source = _all_source()
        required_snippets = [
            "from src.pipeline.unsw_nb15.preprocess import run_preprocess",
            "from src.pipeline.step1 import run_step1",
            "from src.pipeline.common.build_splits import main as build_splits",
            "from src.pipeline.step2.knowledge_graph import run_step2, assert_label_free",
            "from src.pipeline.step4.build_oof_predictions import build_oof_logits",
            "from src.pipeline.step3.build_oof_gnn_embeddings import build_oof_gnn_embeddings",
            "from src.pipeline.step3.encode_kg import run_encode_kg",
            "from src.pipeline.step4.build_prototypes import build_prototypes",
            "from src.pipeline.step3.train_fusion import main as train_fusion",
            "from src.pipeline.step4.train_feedback import train_feedback",
            "from src.pipeline.step4.assemble_ladder import assemble_ladder",
        ]
        for snippet in required_snippets:
            with self.subTest(snippet=snippet):
                self.assertIn(snippet, source)

    def test_notebook_guards_against_known_leakage_and_head_paths(self) -> None:
        source = _all_source()
        self.assertIn("OOF_GNN_EMBEDDINGS_PATH", source)
        self.assertIn("gnn_emb_path=str(OOF_GNN_EMBEDDINGS_PATH)", source)
        self.assertIn("use_head_logits=False", source)
        self.assertIn('modes=["real"]', source)
        self.assertIn("use_llm_head=False", source)
        forbidden = [
            "build_llm_heads",
            "llm_head_logits",
            "use_llm_head=True",
            "use_head_logits=True",
            "0.808",
            "0.826",
        ]
        for snippet in forbidden:
            with self.subTest(snippet=snippet):
                self.assertNotIn(snippet, source)

    def test_notebook_documents_canonical_results_and_limits(self) -> None:
        source = _all_source()
        for snippet in [
            "0.5496",
            "0.7353",
            "0.7459",
            "0.7764",
            "TOP_K_PERCENT",
            "16.0",
            "BIAS_CONFIDENCE_FRAC",
            "0.5",
            "label-conditioned edge aggregation",
            "confidence interval crosses zero",
        ]:
            with self.subTest(snippet=snippet):
                self.assertIn(snippet, source)

    def test_notebook_enforces_deterministic_colab_execution(self) -> None:
        source = _all_source()
        for snippet in [
            'os.environ["IDS_FORCE_CPU"] = "1"',
            'os.environ["OMP_NUM_THREADS"] = "1"',
            "torch.set_num_threads(1)",
            "torch.use_deterministic_algorithms(True, warn_only=True)",
            "SEED = 42",
        ]:
            with self.subTest(snippet=snippet):
                self.assertIn(snippet, source)

    def test_colab_requirements_are_pinned(self) -> None:
        text = COLAB_REQUIREMENTS_PATH.read_text()
        for snippet in [
            "pandas==",
            "numpy==",
            "networkx==",
            "scikit-learn==",
            "transformers==",
            "torch-geometric==",
            "ipywidgets==",
        ]:
            with self.subTest(snippet=snippet):
                self.assertIn(snippet, text)

    def test_readme_explains_drive_resume_and_raw_files(self) -> None:
        text = README_PATH.read_text()
        for snippet in [
            "Google Drive",
            "UNSW-NB15_1.csv",
            "UNSW-NB15_4.csv",
            "RESTORE_IF_AVAILABLE",
            "FORCE_REBUILD",
            "edge_embeddings_oof.pt",
            "trained LLM head",
            "not statistical proof",
        ]:
            with self.subTest(snippet=snippet):
                self.assertIn(snippet, text)

    def test_drive_manifest_template_excludes_noncanonical_artifacts(self) -> None:
        payload = json.loads(CHECKPOINT_MANIFEST_PATH.read_text())
        included = "\n".join(payload["included_artifact_patterns"])
        excluded = "\n".join(payload["excluded_artifact_patterns"])

        self.assertIn("edge_embeddings_oof.pt", included)
        self.assertIn("feedback_oof_real.pt", included)
        self.assertIn("llm_head_logits.pt", excluded)
        self.assertIn("top_k_sweep", excluded)
        self.assertIn("semantic_confidence_sweep", excluded)
        self.assertIn("step3_gnn/edge_embeddings.pt", excluded)


if __name__ == "__main__":
    unittest.main()
