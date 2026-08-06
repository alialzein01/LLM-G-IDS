# UNSW-NB15 Instructor Colab Notebook

This folder contains a clean Google Colab walkthrough for reproducing the
prototype-only LLM-G-IDS pipeline on UNSW-NB15. The notebook is designed for an
instructor to run each phase separately while keeping the implementation identical
to the production modules under `src/`.

## Files

- `unsw_nb15_instructor_pipeline.ipynb` - the phase-by-phase Colab notebook.
- `colab-requirements.txt` - pinned Python packages installed by the notebook.
- `drive-checkpoint-manifest.template.json` - the curated include/exclude
  contract for the Drive checkpoint bundle.

## Google Drive Layout

Create a shared Google Drive folder, add it as a shortcut under the instructor's
My Drive, and set `DRIVE_ROOT` in the notebook to that folder. The expected raw
dataset layout is:

```text
LLM-G-IDS-Instructor/
  raw/
    UNSW-NB15_1.csv
    UNSW-NB15_2.csv
    UNSW-NB15_3.csv
    UNSW-NB15_4.csv
  checkpoints/
```

Only `UNSW-NB15_1.csv` through `UNSW-NB15_4.csv` are used by the pipeline. The
smaller `UNSW_NB15_training-set.csv` and `UNSW_NB15_testing-set.csv` files are
not used by this implementation.

## Resume Controls

The notebook exposes two controls:

- `RESTORE_IF_AVAILABLE`: if true, a phase restores its saved Google Drive
  checkpoint instead of recomputing when all expected outputs are present.
- `FORCE_REBUILD`: if true, the selected phase recomputes and overwrites its
  checkpoint.

Each phase writes only its declared artifacts to
`DRIVE_ROOT/checkpoints/<phase>/`. The large Step 0 normalized CSV is checkpointed
once; downstream artifacts are much smaller.

## Canonical Architecture

The notebook wraps production code instead of reimplementing it:

- Step 0 preprocesses raw UNSW CSVs.
- Step 1 constructs the PyTorch Geometric communication graph and stratified
  splits.
- Step 2 builds structured KG triples and label-free natural-language triples.
- Step 3 builds honest GNN out-of-fold logits and `edge_embeddings_oof.pt`.
- Step 4 encodes KG text with CySecBERT and builds whitened class prototypes.
- Step 5 trains AGAF using `edge_embeddings_oof.pt`, not leaky fit-on-all GNN
  embeddings.
- Step 6 trains the prototype-only feedback loop with `n=16` and semantic
  confidence fraction `0.50`.

The trained LLM head is intentionally excluded from the canonical notebook. It
is an experimental/ablation path, not the LLM used by the final ladder.

## Expected Results

The expected pooled out-of-fold macro-F1 ladder is:

```text
GNN 0.5496 < LLM 0.7353 < AGAF 0.7459 < Feedback loop 0.7764
```

The notebook checks this order and compares exact values against
`results/unsw_nb15_current.json` with a small tolerance. CPU determinism is used
by default because GPU execution can change graph-attention floating-point order
and move the final decimals.

## Important Limitations

The ladder supports the observed point-estimate claim for this controlled
research pipeline, but it is not statistical proof that feedback universally
beats AGAF. The feedback-vs-AGAF confidence interval crosses zero. Also, edges
are aggregated with the attack label in the grouping key, so these are research
cross-validation estimates rather than deployment-valid estimates.

## Runtime Notes

Step 0 is the heavy phase because it computes NetworkX centrality over the raw
2.54M-flow IP graph and produces a roughly 925 MB normalized CSV. Later phases
operate on a 49-node, 656-edge graph and are much smaller, but Step 4 feedback
still retrains five fold-specific models.
