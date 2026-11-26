# When More Trades Do Not Mean More Evidence: Selection Bias Induced by Minimum Trade Count Filters in Backtesting

Research repository for a Practitioner Paper intended for *Applied Stochastic Models in Business and Industry* (ASMBI).

The project studies selection bias induced by requiring a minimum number of trades in a backtest.

## Repository structure

- `src/` — Reusable research code.
- `experiments/` — Experiment configurations and runner scripts.
- `tests/` — Software tests. The initial test checks repository layout.
- `data/` — Data provenance and access documentation.
- `figures/` — Figure-generation scripts and reviewed outputs.
- `docs/` — Research methods and reproducibility documentation.
- `manuscript/` — Manuscript materials approved for public release.

## Run the tests

Use Python 3.12 and run `python -m unittest discover -s tests -v` from the repository root. The initial test checks the repository layout. The author-review manuscript and partial research artifacts are now publicly available. The initial layout test is not evidence of scientific reproducibility.


## Pre-submission review (2026-11-26)

- [Manuscript V1](manuscript/Manuscript_V1_author_review.md): full draft for review, not accepted or submission-certified.
- [Independent confirmation summary](data/ENSI_confirmation_summary.csv) and [corrected round-four summary](data/ENSI_round4_corrected_summary.csv): recovered historical synthetic simulation summaries, not newly generated results.
- [Math check](experiments/ensi_math_check.py) and [round-four diagnostic](experiments/ensi_round4.py): recovered scripts; both depend on the missing original `sr_stopping_v03.py` module. The original path-level Parquet input is also outstanding. Do not claim end-to-end replication yet.
- Figures referenced in the manuscript and the four-part restructuring remain outstanding in this repository.
- Formal author identity, funding/conflict statements, literature verification, independent human checks and final submission approval remain outstanding.

Review priority: check mathematical assumptions and proofs; compare conditional versus overall rejection denominators; recover original code and outputs; review the four-part Practitioner Paper structure; verify references and AI disclosure. Report findings with exact file/section references, distinguish critical issues from suggestions, and do not fabricate results.
