# Research Report Workflow

This directory contains the single M2 research report for the project. The report is
written and reviewed one guideline part at a time. A later part is not drafted until the
current part is approved.

## Authoritative inputs

1. `../reports/Report_Guideline.pdf` controls the report's five-part shape and writing
   principles.
2. `../../TODO.md` defines the intended research system.
3. The source code and committed result contracts on the current working branch define
   what was actually implemented and measured.
4. When the intention and evidence differ, the report states the difference explicitly.

## Section states

| Part | Section | Status |
|---|---|---|
| 1 | Introduction and Context | Review |
| 2 | Background and State of the Art | Waiting |
| 3 | Method | Waiting |
| 4 | Results | Waiting |
| 5 | Discussion and Conclusion | Waiting |
| Final | Abstract, references, appendices, full QA | Waiting |

## Writing standard

- Use natural academic prose with varied sentence structure and explicit transitions.
- Prefer precise claims over promotional language.
- Attach quantitative claims to a result artifact, table, or figure.
- Report unsuccessful experiments alongside successful ones.
- Distinguish observed point estimates from statistically supported differences.
- State limitations and open work where they affect interpretation.
- Keep edge-level classification terminology consistent throughout.
- Terminology is locked for Parts 1--5: call CySecBERT the **pretrained cybersecurity
  language encoder** on first use and the **semantic branch** thereafter; call Phase 2
  **uncertainty-guided semantic feedback**. Do not describe CySecBERT as a generative LLM
  or rename the feedback mechanism in later sections.
- At final QA, replace the provisional cover wording with the title **Graph-Based
  Intrusion Detection with a Cybersecurity Language Encoder** and the subtitle
  **Structural--Semantic Fusion with Uncertainty-Guided Feedback**.

## Historical experiment evidence

Some earlier unsuccessful experiments may have been run outside the artifacts currently
available in this repository. Before such an experiment is described in the Results or
Discussion, drafting pauses and the user is given a precise request to retrieve the
following from Claude:

- experiment name and date;
- hypothesis and reason for running it;
- exact configuration, dataset split, and random seed;
- baseline and measured metrics;
- raw output, log, table, or artifact path;
- observed failure mode and the conclusion drawn; and
- whether the experiment can still be reproduced.

A recollection without a supporting artifact is not treated as measured evidence. It may
be recorded separately as unverified project history, but it is not used to support a
result claim.

## Diagram protocol

Architecture and explanatory diagrams are supplied by the user through Claude. When a
section needs a diagram, drafting pauses before the diagram-dependent passage. The
request must specify the diagram's purpose, complete node and arrow labels, visual
hierarchy, color guidance, aspect ratio, export format, and target filename. Preferred
deliverables are SVG and PDF, with PNG only as a preview.

## Current scope

The report evaluates the structural--semantic intrusion-detection pipeline only on
NF-UNSW-NB15. All research questions, methods, results, figures, and conclusions must
remain within that dataset scope.
