# Task Model and Result Organization Design

## Goal

Organize the existing Task 1 SVM and Task 2 MLP project files so scripts, trained models, prediction files, shared data, and course documents have clear locations. Preserve historical prediction results and make each future run produce a uniquely numbered record.

## Current State

- Task 1 and Task 2 training scripts both load `train_data.mat`, `train_label.mat`, and `test_data.mat`; the data is shared.
- Task 1 has several SVM experiments and historical prediction files. Some older files do not have enough run metadata to recover their exact parameters.
- `svm_model.joblib` is the raw-pixel RBF SVM and matches the raw-pixel RBF prediction snapshot. `train_groupcv_model.joblib` corresponds to the first grouped-CV optimized result. `train_optimized_round2_model.joblib` corresponds to the second grouped-CV optimization result.
- `task2_mlp_model.joblib` corresponds to the MLP output in `task2_result.txt`.
- The local project and the GitHub working copy contain matching versions of their shared project files. The GitHub working copy is clean at the start of this change.

## Directory Layout

```text
data/
  source/
    train(Task_2).zip
    test(Task_2).zip
  train/
    train_data.mat
    train_label.mat
  test/
    test_data.mat
docs/
  assignment/
    大作业2.pptx
    SVMguide.pdf
    ReadMe.txt
  superpowers/specs/
    2026-10-16-task-model-organization-design.md
task1_svm/
  scripts/
  models/
  results/
  metadata/
task2_mlp/
  scripts/
  models/
  results/
  metadata/
```

Task-specific scripts read the shared data from `data/`. Task 1 scripts and artifacts live under `task1_svm/`; Task 2 MLP code and artifacts live under `task2_mlp/`. The assignment-provided random-output `demo.py` is kept under `docs/assignment/examples/` and labeled as a format example, not a trained model or production prediction script.

Temporary editor files and Python bytecode caches are not project deliverables and will not be migrated.

## Result and Model Versioning

Every run uses the basename `task_<METHOD>_<SCHEME>_<ITERATION>`:

- `METHOD` is `SVM` or `MLP`.
- `SCHEME` is a stable positive integer identifying a fixed model/feature configuration. A new feature pipeline or model family receives a new scheme number.
- `ITERATION` starts at 1 and increments for each subsequent run of the same scheme. Existing files are never overwritten.

For example, `task_SVM_1_1.txt`, `task_SVM_1_1.joblib`, and `task_SVM_1_1.json` are the prediction, fitted model, and metadata from the first run of SVM scheme 1. Output files have one-based sample IDs and predicted labels separated by one ASCII space, matching the existing submission format. If a run does not produce a fitted model, only the result and metadata files are written.

Each task has a scheme registry in `metadata/` that records the stable scheme ID and its human-readable definition. A run writes a metadata JSON containing at least the timestamp, script name and SHA-256, model/feature definition, fitted parameters, validation method and score when available, training-data and test-data SHA-256 hashes, sample counts, and produced artifact paths. The output allocator chooses the next unused iteration for the selected scheme and refuses to overwrite an existing artifact.

## Historical Files

Existing results and models are migrated without changing their prediction contents. A legacy manifest records each old path, its new numbered path, the scheme definition, model correspondence when known, and any missing historical details. Identical duplicate result files may be represented by one canonical numbered result plus explicit duplicate entries in the manifest, but only after byte-for-byte equality is verified. Different outputs remain separate files. Unknown training parameters or scores are marked unavailable rather than inferred from filenames.

The initial scheme registry will assign a separate scheme to each distinct identifiable SVM configuration. The two grouped-CV training runs use the same optimization scheme with iterations 1 and 2. MLP starts with its first scheme and iteration. Existing duplicated aliases of the raw-pixel RBF result are documented in the legacy manifest.

## Script Changes

All training and optimization scripts will resolve paths relative to the project root rather than the current shell directory. They will read from `data/`, save results to the appropriate task `results/` folder, save fitted estimators to `models/`, and create or update metadata and legacy mappings. Scripts that compare candidate outputs will read their baselines through the task results directory. No model-selection or prediction logic is changed as part of this organization.

## Verification

- Confirm all moved source files are present and have unchanged bytes.
- Compile Python scripts to catch syntax errors without running model training.
- Run lightweight path and version-allocation checks to confirm outputs target the right task folders and the next iteration increments without overwriting prior artifacts.
- Compare the migrated historical result files against their pre-migration contents and confirm sample IDs and labels are unchanged.
- Confirm both the local project and GitHub working copy have the same intended structure and contents.
