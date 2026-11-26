# When More Trades Do Not Mean More Evidence: Selection Bias Induced by Minimum Trade Count Filters in Backtesting

Research repository for a Practitioner Paper intended for *Applied Stochastic Models in Business and Industry* (ASMBI).

Author: MIN HAN HSIEH, Independent Researcher, Taiwan. No external research funding; no known conflicts of interest.

## Current review version

- [Manuscript V2.2](manuscript/Manuscript_V2_PractitionerPaper_Draft.md) is the current author-confirmed submission manuscript, structured according to ASMBI's four-part guidance.
- [New simulation](experiments/replication_v2.py) is independently specified, not a recovery of the historical simulator.
- [New summary](data/recomputed/summary.csv) contains nine contrasts from 30,000 windows. The AI-assisted second review reported reproduction to six decimal places using the pinned dependencies in a clean environment.
- [V1](manuscript/Manuscript_V1_author_review.md), the original diagnostic scripts and historical CSVs are retained for provenance only. The original `sr_stopping_v03.py` and historical path-level files have not been recovered. Do not execute the legacy scripts as part of V2 replication.
- Author name, research independence, funding and conflict declarations have been confirmed by the author. On 26 November 2026, the author confirmed research leadership, independent verification of mathematical derivations, code, results, references and manuscript content, full authorship responsibility and compliance with Wiley publication ethics and AI disclosure requirements.

## Reproduce

Use Python 3.12.

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python experiments/replication_v2.py --out /tmp/asmbi-recomputed --n 10000 --seed 20261126
python tests/check_replication.py data/recomputed/summary.csv /tmp/asmbi-recomputed/summary.csv /tmp/asmbi-recomputed/paths.csv.gz
```

The run writes the full per-window summary records, aggregate CSV and a rejection-rate figure. The records contain counts, validity/rejection indicators and the mean of the completed-event rewards; they do not contain all individual Gaussian increments or trade rewards.

## Review status

The historical results cannot be exactly rerun with the available historical files. The new experiment is the primary computational illustration. AI-assisted reviews do not constitute independent human author verification; the author has separately attested to completing that verification. No journal acceptance or editorial approval is implied.

See [AI disclosure](AI_DISCLOSURE.md) and [citation metadata](CITATION.cff).
