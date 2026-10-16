import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import (
    GridSearchCV,
    GroupKFold,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.experiment_tracking import save_run


TRAIN_DIR = ROOT / "data" / "train"
TEST_DIR = ROOT / "data" / "test"
RESULTS_DIR = ROOT / "task1_svm" / "results"
MODELS_DIR = ROOT / "task1_svm" / "models"
METADATA_DIR = ROOT / "task1_svm" / "metadata"


def make_contiguous_groups(labels, chunks=5):
    groups = np.full(labels.size, -1, dtype=int)
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        if not np.array_equal(indices, np.arange(indices[0], indices[-1] + 1)):
            raise ValueError("Training rows are not contiguous by class")
        if indices.size % chunks:
            raise ValueError("Each class count must divide evenly into the group count")
        chunk_size = indices.size // chunks
        for chunk in range(chunks):
            start = chunk * chunk_size
            groups[indices[start : start + chunk_size]] = chunk
    if np.any(groups < 0):
        raise ValueError("Could not assign every training row to a validation group")
    return groups


def main():
    x_train = loadmat(TRAIN_DIR / "train_data.mat")["train_data"]
    y_train = loadmat(TRAIN_DIR / "train_label.mat")["train_label"].ravel()
    x_test = loadmat(TEST_DIR / "test_data.mat")["test_data"]

    if x_train.shape[0] != y_train.size:
        raise ValueError("Training data and labels have different sample counts")
    if x_train.shape[1] != x_test.shape[1] or x_train.shape[1] != 19 * 19:
        raise ValueError("Expected matching 19x19 grayscale feature vectors")
    if not np.isfinite(x_train).all() or not np.isfinite(x_test).all():
        raise ValueError("Input data contains NaN or infinite values")

    pipeline = Pipeline(
        [
            ("scale", MinMaxScaler(feature_range=(-1, 1))),
            ("svm", SVC(kernel="rbf")),
        ]
    )
    parameter_grid = {
        "svm__C": [2.0**power for power in np.arange(1, 5.5, 0.5)],
        "svm__gamma": [2.0**power for power in np.arange(-8, 0.5, 0.5)],
    }
    random_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search = GridSearchCV(
        pipeline,
        parameter_grid,
        scoring="accuracy",
        cv=random_cv,
        n_jobs=-1,
        refit=True,
        return_train_score=False,
    )
    search.fit(x_train, y_train)

    groups = make_contiguous_groups(y_train)
    grouped_cv = GroupKFold(n_splits=5)
    grouped_scores = cross_val_score(
        search.best_estimator_,
        x_train,
        y_train,
        cv=grouped_cv,
        groups=groups,
        scoring="accuracy",
        n_jobs=-1,
    )

    # This raw-pixel candidate has the highest user-reported test score so far.
    predictions = search.best_estimator_.predict(x_test).astype(int)
    result = np.column_stack((np.arange(1, predictions.size + 1), predictions))
    saved = save_run(
        method="SVM",
        scheme_id=1,
        result=result,
        model=search.best_estimator_,
        results_dir=RESULTS_DIR,
        models_dir=MODELS_DIR,
        metadata_dir=METADATA_DIR,
        metadata={
            "script_path": Path(__file__),
            "dataset_paths": {
                "training_data": TRAIN_DIR / "train_data.mat",
                "training_labels": TRAIN_DIR / "train_label.mat",
                "test_data": TEST_DIR / "test_data.mat",
            },
            "model_family": "SVC with RBF kernel",
            "features": "19x19 raw pixels scaled to [-1, 1] with MinMaxScaler",
            "training_sample_count": int(x_train.shape[0]),
            "test_sample_count": int(x_test.shape[0]),
            "best_parameters": search.best_params_,
            "validation": {
                "selection": "5-fold shuffled stratified CV",
                "selection_accuracy": float(search.best_score_),
                "grouped_cv": "5-fold contiguous-group CV",
                "grouped_cv_mean_accuracy": float(grouped_scores.mean()),
                "grouped_cv_std_accuracy": float(grouped_scores.std()),
            },
        },
    )

    print(f"Training samples: {x_train.shape[0]}")
    print(f"Test samples: {x_test.shape[0]}")
    print("Selected model: MinMax-scaled raw pixels + RBF SVM")
    print(f"Best parameters: {search.best_params_}")
    print(f"5-fold shuffled stratified CV accuracy: {search.best_score_:.4%}")
    print(
        "5-fold contiguous-group CV accuracy: "
        f"{grouped_scores.mean():.4%} +/- {grouped_scores.std():.4%}"
    )
    print(f"Predicted labels: {dict(zip(*np.unique(predictions, return_counts=True)))}")
    print(f"Saved result: {saved['result']}")
    print(f"Saved model: {saved['model']}")
    print(f"Saved metadata: {saved['metadata']}")


if __name__ == "__main__":
    main()
