# Pattern Recognition Course Project

## Project Layout

- `data/train/` and `data/test/` hold the shared MAT files used by both tasks; `data/source/` keeps the original archives.
- `task1_svm/scripts/` and `task2_mlp/scripts/` hold the Task 1 SVM and Task 2 MLP scripts.
- Each task has separate `models/`, `results/`, and `metadata/` directories for its saved artifacts.
- Root `accuracy/` is a read-only directory for manually supplied reference prediction TXT files.
- Each task's `metadata/schemes.json` defines its stable scheme IDs; `legacy_manifest.csv` maps imported historical files.
- `docs/assignment/` contains the course presentation, SVM guide, readmes, and output-format example.
- `common/experiment_tracking.py` allocates run numbers and saves each run's outputs without overwriting existing files.
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

Every run uses the basename `task_<METHOD>_<SCHEME>_<ITERATION>`, where `METHOD` is `SVM` or `MLP`, `SCHEME` is the stable scheme ID, and `ITERATION` is the run number. For example, SVM scheme 1, run 2 produces:

```text
task1_svm/results/task_SVM_1_2.txt
task1_svm/models/task_SVM_1_2.joblib
task1_svm/metadata/task_SVM_1_2.json
```

The prediction text file contains sample ID and predicted label columns. The `.joblib` file contains the fitted estimator; for a voting scheme it contains the fitted member estimators and voting settings. The JSON file records the parameters, validation details, dataset and script hashes, and paths to the other artifacts. MLP files use the same naming format under `task2_mlp/`.

Scheme IDs are listed in each task's `metadata/schemes.json`. Keep an ID when rerunning the same feature and model configuration; assign a new ID when the configuration itself changes. Every run increments the iteration for that scheme. The allocator checks results, models, and metadata, so an iteration already used by any artifact is not reused. Files are opened in exclusive-create mode: an existing model file cannot be overwritten. Each new training run saves a new `.joblib` model alongside its prediction and metadata files. Some imported historical prediction files have no model because their original fitted estimator was not available; this does not apply to new training runs.

Place reference prediction TXT files in `accuracy/` manually. Training scripts read compatible two-column files from either method and, when sample IDs match, print the number of prediction differences. If a filename ends in a decimal suffix such as `_0.86.txt`, that suffix records a known accuracy of 86%; no fixed filename format is otherwise required. Training scripts do not write files into `accuracy/`.

Each run record includes the script and data hashes, model configuration, validation results when available, sample counts, label counts, and output paths. The test set has no known labels in this project, so reported accuracy is cross-validation accuracy on training data; the generated test files are predictions, not measured test accuracy.

The archived SVM results occupy iteration 1 for schemes 1 through 21; schemes 20 and 21 are separate grouped-CV workflows. The MLP history occupies scheme 1, iteration 1. Repeating any of these configurations starts at iteration 2.
