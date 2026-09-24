# Instructor revision — 2026-09-24

All five items in the instructor's new message are addressed.

1. Abstract rewritten as one continuous paragraph: IoT, machine learning, deep learning, graph models, language models, GLASS, datasets, findings and scope. No section references. The suggested example's unsupported claims of monotonic improvement and feedback superiority over the trained head were not adopted.
2. RQ2 now answers fusion versus GNN and fusion versus the prototype semantic stage separately. The missing comparison was computed from saved predictions using the existing primary and secondary bootstrap procedures (2,000 iterations, bootstrap seed 42, training seeds 42/1/2). No retraining. Reconstructed scores are asserted against the existing multi-seed artifact. Both intervals exclude zero on both datasets; fusion versus GNN remains unseparated under the primary procedure.
   - Reproduction: `OMP_NUM_THREADS=1 .venv/bin/python scripts/close_rq2_interval.py`
   - New evidence: `results/rq2_fusion_semantic_interval.json`
   - Report locations: Table 5, Results §4.2, Discussion §5.1, Abstract and Conclusion.
   - Existing canonical contracts and model code are unchanged. The trained head remains a separate component comparison, not a fifth ladder stage.
3. Edge representation and adaptive gate equations now have numbers and references.
4. Introduction follows the requested progression, includes a specific GLASS contribution paragraph after prior work, and ends with a concise report roadmap. Prior iterative work (DAS) is acknowledged rather than incorrectly described as exclusively post-hoc.
5. Conclusion follows the same progression, with more detail than the Abstract and less than the Introduction. GLASS is used throughout the main chapters (23 occurrences across section sources).

Validation: number audit passes with zero unknown numbers, zero unverified provenance and zero banned terms. The two remaining terminology warnings refer to an explicit distinction from generative models and a literal legacy CLI flag. Generated tables are current. Exactly two body paragraphs remain in each Results subsection. PDF references and equation numbering checked, full-document render inspected. No backend training or full backend test suite was run for these report edits; the new statistical calculation finished successfully.
