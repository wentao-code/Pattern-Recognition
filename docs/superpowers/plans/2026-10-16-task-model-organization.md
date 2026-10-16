# Task Model Organization Implementation Plan

> **For agentic workers:** Use inline execution with the existing approved design. Steps use checkbox syntax for tracking.

**Goal:** Place shared data, course documents, Task 1 SVM files, and Task 2 MLP files in clear folders, with stable numbered outputs and run metadata.

**Architecture:** Keep one shared `data/` directory because both model scripts consume the same MAT files. Put model-specific scripts, models, results, and metadata in each task directory. Use `common/experiment_tracking.py` for non-overwriting run numbers, result/model persistence, and JSON metadata.

**Tech Stack:** Python, NumPy, SciPy, scikit-learn, joblib, PowerShell, Git.

## Global Constraints

- Preserve every distinct historical prediction and fitted model without changing its bytes.
- Use `task_<METHOD>_<SCHEME>_<ITERATION>` names; `METHOD` is `SVM` or `MLP`.
- A scheme ID permanently identifies a model/feature configuration; later runs increment only its iteration ID.
- Every new run records the script SHA-256, data SHA-256 hashes, parameters, validation method and score when available, counts, and artifact paths.
- Scripts resolve all paths from their location, never from the shell's current directory.
- Do not alter model-selection or prediction logic while reorganizing.
- Keep `.idea`, `__pycache__`, and the temporary PowerPoint lock file out of the project migration.

---

### Task 1: Add Shared Run Tracking

**Files:**
- Create: `common/__init__.py`
- Create: `common/experiment_tracking.py`
- Create: `task1_svm/metadata/schemes.json`
- Create: `task2_mlp/metadata/schemes.json`

**Interfaces:**
- `next_iteration(results_dir: Path, method: str, scheme_id: int, models_dir: Path | None = None, metadata_dir: Path | None = None) -> int` scans numbered result, model, and metadata files, then returns the next available iteration.
- `save_run(method, scheme_id, result, model, results_dir, models_dir, metadata_dir, metadata) -> dict[str, Path]` refuses collisions, writes the two-column integer result, optionally saves a joblib model, and writes JSON metadata under the shared basename.
- `sha256_file(path: Path) -> str` streams a file to compute its SHA-256.

- [x] Implement the helper with basename regex `^task_<METHOD>_<SCHEME>_<ITERATION>` and collision checks across `.txt`, `.joblib`, and `.json` artifacts.
- [x] Define registry IDs 1-21 for Task 1 in `schemes.json`; define MLP scheme 1 in Task 2. IDs are stable and descriptions name the feature/model configuration or explicitly say `legacy configuration unknown`.
- [x] Check the helper and registries are free of overwrite paths and invalid JSON.

### Task 2: Create the Numbered Historical Manifest and Folders

**Files:**
- Create: `task1_svm/scripts/`, `task1_svm/models/`, `task1_svm/results/`, `task1_svm/metadata/`
- Create: `task2_mlp/scripts/`, `task2_mlp/models/`, `task2_mlp/results/`, `task2_mlp/metadata/`
- Create: `task1_svm/metadata/legacy_manifest.csv`
- Create: `task2_mlp/metadata/legacy_manifest.csv`

**Interfaces:**
- Each manifest row records `legacy_path`, `numbered_path`, `artifact_type`, and `notes`.
- Scheme 1 is raw-pixel RBF SVM; scheme 20 is grouped-CV raw-pixel SVM; scheme 21 is grouped-CV pixel/HOG search.
- Other identifiable historical variants have distinct stable scheme IDs. Unknown legacy outputs use explicit unknown-config schemes.

- [x] Map each verified duplicate result pair to one canonical file: `task1_result.txt` with `task1_result_hog_rbf.txt` as `task_SVM_14_1.txt`, and `task1_result_best_known.txt` with `task1_result_raw_pixel_rbf.txt` as `task_SVM_1_1.txt`.
- [x] Map the grouped-CV raw-pixel result/model to `task_SVM_20_1` and the separate grouped-CV feature-search result/model to `task_SVM_21_1`; map `svm_model.joblib` and its matching raw-pixel RBF result to `task_SVM_1_1`.
- [x] Map the MLP result/model pair to `task_MLP_1_1`; preserve the one-column label file as `task_SVM_14_1_labels.txt` because it matches the HOG/RBF result.
- [x] Record every old path in the manifest, including duplicate aliases, and verify every unique source result/model has a destination.

### Task 3: Move Data, Documents, Scripts, Models, and Results

**Files:**
- Move: `train(Task_2)/` and `test(Task_2)/` to `data/train/` and `data/test/`
- Move: training/test zip files to `data/source/`
- Move: course PPTX, SVM PDF, and readmes to `docs/assignment/`
- Move: Task 1 scripts into `task1_svm/scripts/`; move the MLP script into `task2_mlp/scripts/`
- Move: model binaries and result files into their numbered task artifact folders
- Move: the random-format `demo.py` to `docs/assignment/examples/demo.py`

**Interfaces:**
- All three MAT files remain a single shared source of truth.
- The example script remains labeled as a random format sample, not a trained result.

- [x] Create destination directories before moving files; keep the already committed design document under `docs/superpowers/specs/`.
- [x] Move files according to the manifest; retain exact result and model bytes. Remove a duplicate physical result only when its canonical destination and manifest entry exist.
- [x] Add a root `README.md` with the directory map, Python run commands, shared data locations, naming rules, and latest known run IDs.

### Task 4: Update Training and Optimization Scripts

**Files:**
- Modify: `task1_svm/scripts/train_svm.py`
- Modify: `task1_svm/scripts/optimize_candidates.py`
- Modify: `task1_svm/scripts/optimize_round2.py`
- Modify: `task1_svm/scripts/train_groupcv_candidate.py`
- Modify: `task1_svm/scripts/evaluate_ensembles.py`
- Modify: `task2_mlp/scripts/train_task2_mlp.py`
- Create: `optimize_candidates.py` as a compatibility shim for the moved feature functions used by the old joblib model

**Interfaces:**
- Every script derives the project root from `Path(__file__).resolve()` and loads arrays from `data/train/` and `data/test/`.
- Every prediction-producing run calls `common.experiment_tracking.save_run` with its stable scheme ID and run metadata.
- Comparison scripts load baseline files from `task1_svm/results/` by their numbered manifest paths.

- [x] Add the project root to `sys.path` before importing `common.experiment_tracking`, so scripts run both from the project root and by absolute script path.
- [x] Replace hard-coded top-level result/model paths with `save_run`; preserve all current estimators, search grids, folds, and predictions.
- [x] Keep optimization result comparisons tied to the canonical numbered baseline paths and print the allocated output basename.
- [x] Record CV summaries and selected parameters for each candidate/ensemble run; include script/data hashes and sample counts for every run.

### Task 5: Mirror and Verify Both Work Copies

**Files:**
- Sync project deliverables to: `C:\Users\WWT\Desktop\课程课件\大作业2 to student\大作业2 to student`
- Commit and push: `Pattern-Recognition` `main`

- [x] Compare pre-migration SHA-256 snapshots with migrated data, documents, model binaries, and unique historical result files.
- [x] Parse all moved Python scripts with `ast.parse` without running model training.
- [x] Inspect manifests and registry JSON; confirm every historical file maps to a destination and next IDs are 2 for SVM schemes 1, 20, and 21 and MLP scheme 1.
- [x] Mirror the organized project to the local non-Git directory while leaving `.idea`, `__pycache__`, and temporary editor files in place.
- [ ] Confirm the repository diff contains only the planned changes, commit the implementation, and push `main` to `origin`.
