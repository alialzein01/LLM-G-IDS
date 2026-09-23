# Supervisor revision, 2026-09-23

Deliverable: `build/finalReport.pdf`, rebuilt from `main.tex`. The normal build now updates this filename as well as `build/main.pdf`. The availability sentence retains `https://github.com/alialzein01/LLM-G-IDS`, as requested. No branch or remote operations were performed.

## Comment coverage

| Request | Where addressed |
|---|---|
| All models and trained-head justification | Method semantic/feedback subsections; Results 4.2 and 4.4; Discussion 5.3 and 5.4. `tab:metrics` is now a separate model/component comparison including the head alone; the four-stage ladder has no added row. |
| Add Dr Fouad Al Tfaily below Dr Hussein Hazimeh | Cover under “Supervisors”; acknowledgements aligned. |
| Clearer methodology/results | Revised semantic and feedback explanations; rewritten Results and Discussion; simpler abstract and conclusion. |
| Two paragraphs per Results subsection | All eight subsections: first summarises measurements, second explains them and their limits. Supporting detail retained in appendices. |
| Unnumbered section introductions | All main numbered sections and appendices. |
| M1: mechanism framing | Abstract, Introduction 1.3, Results 4.3, Discussion and Conclusion distinguish complete-system gains from unestablished injection gains and the head-alone comparison. |
| M2: deployment caveat repetition | Full explanation and required changes in Method 3.2; short references elsewhere. |
| M3: small-class caution | Beside headline results in 4.2 and developed in 4.7. |
| M4: data/code availability | Unnumbered statement with existing dataset citations and the user's repository URL. |
| m1: repeated prior-work positioning | Removed repeated disclaimers in Related Work 2.5; retained the consolidated positioning. |
| m2: abstract readability | Opens with the two main findings in plain language. |
| m3: Lists of Figures and Tables | Added after Contents, with short list entries and linked contents entries. |
| m4: orientation paragraphs | Included as above. |
| m5: baseline settings | Generated `tab:baseline-settings` in Method 3.8 reads `baselines.<arch>.<variant>.hyperparameters` from each dataset's saved baseline artifact. Common settings are in the caption. |
| m6: knowledge representation wording | Replaced the claimed internal knowledge representation with fixed-template controlled-language edge descriptions. |
| m7: centrality averaging | Method 3.2 explains that averaging recovers a per-IP constant. Independently checked both processed edge CSVs; evidence saved locally in `tmp/pdfs/revision/centrality_check.json`. |
| Defense questions | Boundary selection, unadapted encoder, missing message-passing/flat-classifier controls and weighted-branch ablation are stated as limitations/future experiments. No new training run. |

## Additional corrections verified during revision

- The isolation configuration is disclosed in Results 4.3, its table and figure captions, and Appendix A.5: earlier consultation percentiles, with unchanged injection scales. The abstract explicitly reports the trained-head injection loss on NF-ToN-IoT.
- The prototype estimates a whitening transform and labelled class means. It is no longer described as “fitting nothing” or measuring the encoder alone.
- The graph-only `head_only` control is distinct from the standalone semantic `head_alone`. Its semantic output fusion is disabled in the model code; the old “fusion active in all arms” caption was corrected.
- Source code uses AdamW for the primary pipeline and trained semantic head; the baseline artifacts specify Adam. Those roles are kept separate.
- The handoff's small-class example mistakenly names UNSW where it means ToN. The report follows `results/multiseed_head_per_class.json`.
- Churn's dependence on untracked local traces is disclosed. The old statement that all processed graphs were committed was removed after checking the tracked files.

## Validation scope

- Normal LaTeX build; final PDF visually reviewed, with no overfull text or unresolved references.
- `scripts/check_report_numbers.py`: passes; existing terminology warnings concern a contrast with generative models, prior work, and a literal CLI flag.
- `tables/make_tables.py --check`: passes; generated tables match their sources.
- All eight Results subsections have exactly two body paragraphs, excluding figures, tables and captions.
- PDF text checks cover supervisor order, figure/table lists, repository URL and the separated-loss statement.
- Result contracts and pipeline source were not changed; no training or new performance experiment was run. The full pipeline test result in the report is explicitly a recorded run, not a claim that it was rerun for this editorial revision.
