from __future__ import annotations

import os
from pathlib import Path

import torch
from dotenv import load_dotenv
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

load_dotenv()

MODEL_ID = "markusbayer/CySecBERT"
NL_TRIPLES_PATH = "data/ton_iot/processed/step2/kg_triples_nl.txt"
OUTPUT_DIR = "data/ton_iot/processed/step3_llm"
BATCH_SIZE = 32


def _select_device() -> torch.device:
    # Set IDS_FORCE_CPU=1 to pin encoding to CPU. MPS and CPU give slightly
    # different embeddings, which changes every downstream number.
    if os.environ.get("IDS_FORCE_CPU") == "1":
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _mean_pool(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Average token vectors, ignoring padding positions."""
    mask = attention_mask.unsqueeze(-1).float()
    return (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)


def load_sentences(path: str) -> list[str]:
    sentences = Path(path).read_text().strip().splitlines()
    print(f"Loaded {len(sentences)} NL sentences from {path}")
    return sentences


def encode(
    sentences: list[str],
    model_id: str = MODEL_ID,
    batch_size: int = BATCH_SIZE,
    hf_token: str | None = None,
) -> torch.Tensor:
    device = _select_device()
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    model = AutoModel.from_pretrained(model_id, token=hf_token)
    model.eval()
    model.to(device)
    print(f"Loaded {model_id}")

    all_embeddings: list[torch.Tensor] = []

    with torch.no_grad():
        for i in tqdm(range(0, len(sentences), batch_size), desc="Encoding"):
            batch = sentences[i : i + batch_size]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            output = model(**encoded)
            embeddings = _mean_pool(output.last_hidden_state, encoded["attention_mask"])
            # Move back to CPU before collecting — avoids accumulating on MPS
            all_embeddings.append(embeddings.cpu())

    return torch.cat(all_embeddings, dim=0)


def run_encode_kg(
    nl_path: str = NL_TRIPLES_PATH,
    output_dir: str = OUTPUT_DIR,
    hf_token: str | None = None,
) -> torch.Tensor:
    if hf_token is None:
        hf_token = os.environ.get("HF_TOKEN")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    sentences = load_sentences(nl_path)
    embeddings = encode(sentences, hf_token=hf_token)

    assert embeddings.shape == (len(sentences), 768), (
        f"Expected shape ({len(sentences)}, 768), got {embeddings.shape}"
    )

    output_path = out / "edge_embeddings.pt"
    torch.save(embeddings, output_path)
    print(f"Saved embeddings: shape={list(embeddings.shape)}, path={output_path}")

    return embeddings


if __name__ == "__main__":
    run_encode_kg()
