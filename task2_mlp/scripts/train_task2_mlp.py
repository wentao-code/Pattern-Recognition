import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.experiment_tracking import save_run


TRAIN_DIR = ROOT / "data" / "train"
TEST_DIR = ROOT / "data" / "test"
RESULTS_DIR = ROOT / "task2_mlp" / "results"
MODELS_DIR = ROOT / "task2_mlp" / "models"
METADATA_DIR = ROOT / "task2_mlp" / "metadata"


def make_contiguous_groups(labels, n_splits=5):
    groups = np.full(labels.size, -1, dtype=int)
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        if not np.array_equal(indices, np.arange(indices[0], indices[-1] + 1)):
            raise ValueError("Training rows are not contiguous by class")
        for group, chunk in enumerate(np.array_split(indices, n_splits)):
            groups[chunk] = group
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
    if not np.isin(y_train, (-1, 1)).all():
        raise ValueError("Expected training labels to be -1 or +1")

    pipeline = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    solver="adam",
                    learning_rate_init=0.001,
                    max_iter=1000,
                    early_stopping=True,
                    validation_fraction=0.15,
                    n_iter_no_change=25,
                    random_state=42,
                ),
            ),
        ]
    )
    parameter_grid = {
        "mlp__hidden_layer_sizes": [(64,), (128,), (256,), (128, 64)],
        "mlp__activation": ["relu", "tanh"],
        "mlp__alpha": [0.0001, 0.001, 0.01],
    }
    groups = make_contiguous_groups(y_train)
    search = GridSearchCV(
        pipeline,
        parameter_grid,
        scoring="accuracy",
        cv=GroupKFold(n_splits=5),
        n_jobs=-1,
        refit=True,
        return_train_score=False,
    )
    search.fit(x_train, y_train, groups=groups)

    predictions = search.best_estimator_.predict(x_test).astype(int)
    result = np.column_stack((np.arange(1, predictions.size + 1), predictions))
    saved = save_run(
        method="MLP",
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
            "model_family": "MLPClassifier with StandardScaler",
            "training_sample_count": int(x_train.shape[0]),
            "test_sample_count": int(x_test.shape[0]),
            "best_parameters": search.best_params_,
            "validation": {
                "method": "5-fold GroupKFold over contiguous class groups",
                "accuracy": float(search.best_score_),
                "fold_standard_deviation": float(
                    search.cv_results_["std_test_score"][search.best_index_]
                ),
            },
        },
    )

    best_index = search.best_index_
    counts = dict(zip(*np.unique(predictions, return_counts=True)))
    print(f"Training samples: {x_train.shape[0]}")
    print(f"Test samples: {x_test.shape[0]}")
    print(f"Best 5-fold grouped CV accuracy: {search.best_score_:.4%}")
    print(
        "Best CV fold standard deviation: "
        f"{search.cv_results_['std_test_score'][best_index]:.4%}"
    )
    print(f"Best parameters: {search.best_params_}")
    print(f"Predicted labels: {counts}")
    print(f"Saved result: {saved['result']}")
    print(f"Saved model: {saved['model']}")
    print(f"Saved metadata: {saved['metadata']}")


if __name__ == "__main__":
    main()
