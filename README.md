# PAIEC 2026 entry: shared-label item response models

Entry to the NeurIPS 2026 Predictive AI Evaluation Competition (Stanford AIMS, Codabench).
The final submission is `submissions/v4_stacked` (`model.py` + `stack_model.json`).
The technical report is in `report/` (NeurIPS template).

## Method in one paragraph
For the benchmark of the query, the model fits `logit p = c + ability(subject) + difficulty(item)`
by penalised Newton updates on every label revealed for that benchmark (labels of other subjects
included), and estimates the two prior scales from the fit (empirical Bayes). A tree ensemble
(`stack_model.json`, trained offline on simulated episodes, validated by leave-benchmark-out
cross-validation) then corrects the probability. It predicts 0.5 when no label of the benchmark is
available. Pure Python, no dependency, no pretrained model.

## Layout
| Path | Content |
| --- | --- |
| `submissions/v1_shrinkage` | v1: `(k+1)/(n+2)` per subject-benchmark pair (score 0.195943) |
| `submissions/v2_shared_irt` | v2: shared-label IRT, fixed prior scales (score 0.171685) |
| `submissions/v3_shared_irt_eb` | v3: shared-label IRT, empirical Bayes scales (score 0.170854), with the submitted ZIP |
| `submissions/v4_stacked` | v4 (final): v3 plus the learned second stage, with the ZIP |
| `experiments/cell01.py` ... `cell14.py` | Kaggle scripts: data compaction, simulator, embeddings, all experiments of the report (comments are in French) |
| `results/` | raw result tables of cells 7 to 11, 13 and 14 (Kaggle runs). `cell12.py` (posterior-variance predictions) was only run on a small local sample and is not reported |
| `tools/build_zip.py` | builds a submission ZIP from a `model.py` |
| `tools/make_figure.py` | regenerates `report/figures/results.pdf` |
| `report/` | LaTeX source of the report |

## Rebuild and check the submission
```bash
python tools/build_zip.py submissions/v4_stacked/model.py dist/stacked_v4.zip submissions/v4_stacked/stack_model.json
python check_submission_zip.py dist/stacked_v4.zip   # script of the official baseline repository
```

## Reproduce the experiments
1. Run `experiments/cell01.py` in a Kaggle notebook (Internet on, secret `HF_TOKEN`) to download
   `aims-foundations/measurement-db` and write the compact binary-response table.
2. Add that output as a notebook input and run `cell02.py`, `cell06.py` to `cell11.py`, `cell13.py`, `cell14.py` on CPU
   (`cell03.py`, `cell04.py` build item texts and embeddings; `cell04.py` needs a GPU).
3. `cell14.py` writes the trees (`stack_model14.json`, copied as `stack_model.json` in the submission). Each script prints its result tables and saves a `results_cellNN.csv`. Seeds are fixed.

## Training data and disclosures
Only the public `measurement-db` was used. `intfloat/multilingual-e5-small` was used only in the
text-embedding experiments, never in the submission. Code written with the help of an LLM coding
assistant (Claude, Anthropic) under the author's direction.

License: MIT.
