from __future__ import annotations

import unittest
import os
from unittest.mock import patch

import torch

from src.pipeline.step3 import encode_kg


class _FakeTokenizer:
    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs):
        _FakeTokenizer.calls.append((model_id, kwargs))
        return cls()

    calls: list[tuple[str, dict]] = []

    def __call__(self, sentences, **kwargs):
        return {
            "input_ids": torch.ones((len(sentences), 2), dtype=torch.long),
            "attention_mask": torch.ones((len(sentences), 2), dtype=torch.long),
        }


class _FakeModel:
    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs):
        _FakeModel.calls.append((model_id, kwargs))
        return cls()

    calls: list[tuple[str, dict]] = []

    def eval(self):
        return self

    def to(self, device):
        return self

    def __call__(self, **kwargs):
        batch_size, seq_len = kwargs["input_ids"].shape
        return type(
            "Output",
            (),
            {"last_hidden_state": torch.ones((batch_size, seq_len, 768))},
        )()


class EncodeKGRevisionTest(unittest.TestCase):
    def test_encode_forwards_revision_to_huggingface_loaders(self) -> None:
        _FakeTokenizer.calls.clear()
        _FakeModel.calls.clear()

        with (
            patch.object(encode_kg, "AutoTokenizer", _FakeTokenizer),
            patch.object(encode_kg, "AutoModel", _FakeModel),
            patch.dict(os.environ, {"IDS_FORCE_CPU": "1"}),
        ):
            embeddings = encode_kg.encode(
                ["edge description"],
                model_id="example/model",
                revision="abc123",
                hf_token="token",
            )

        self.assertEqual(tuple(embeddings.shape), (1, 768))
        self.assertEqual(
            _FakeTokenizer.calls,
            [("example/model", {"token": "token", "revision": "abc123"})],
        )
        self.assertEqual(
            _FakeModel.calls,
            [("example/model", {"token": "token", "revision": "abc123"})],
        )


if __name__ == "__main__":
    unittest.main()
