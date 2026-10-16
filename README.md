# Pattern Recognition Course Project

## Project Layout

- `data/train/` and `data/test/` hold the shared MAT files used by both tasks.
- `data/source/` keeps the original training and test archives.
- `task1_svm/scripts/`, `models/`, and `results/` contain Task 1 SVM code and artifacts.
- `task2_mlp/scripts/`, `models/`, and `results/` contain Task 2 MLP code and artifacts.
- Each task's `metadata/` contains its scheme registry, legacy-file mapping, and per-run records.
- `docs/assignment/` contains the course presentation, SVM guide, readmes, and random output-format example.
- `common/experiment_tracking.py` allocates run numbers and writes result, model, and metadata files.
- Root `optimize_candidates.py` is a compatibility shim required by the serialized historical SVM pipeline.

## Run Scripts

Run commands from the repository root. Paths are resolved from the script location.

```powershell
python task1_svm/scripts/train_svm.py
python task1_svm/scripts/train_groupcv_candidate.py
python task1_svm/scripts/optimize_round2.py
python task1_svm/scripts/optimize_candidates.py
python task1_svm/scripts/evaluate_ensembles.py
python task2_mlp/scripts/train_task2_mlp.py
```

The candidate search scripts can take longer than a single model fit. They use the same shared data files as the other scripts.

## Versioned Outputs

Prediction files follow `task_<METHOD>_<SCHEME>_<ITERATION>.txt`, for example `task_SVM_1_1.txt` and `task_MLP_1_1.txt`. A scheme ID identifies a fixed feature/model configuration; each new run of that scheme increments its iteration. Results, fitted models, and run metadata share the same basename and are placed in their task's `results/`, `models/`, and `metadata/` directories. Existing historical runs occupy their original iteration numbers; new training runs will use the next number.

Each run record includes the script and data hashes, model configuration, validation results when available, sample counts, label counts, and output paths. The test set has no known labels in this project, so reported accuracy is cross-validation accuracy on training data; the generated test files are predictions, not measured test accuracy.

The archived SVM results occupy iteration 1 for schemes 1 through 21; schemes 20 and 21 are separate grouped-CV workflows. The MLP history occupies scheme 1, iteration 1. Repeating any of these configurations starts at iteration 2.
