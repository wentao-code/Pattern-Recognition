import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

from train_svm import make_contiguous_groups


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.experiment_tracking import save_run


def main():
    train_dir = ROOT / "data" / "train"
    test_dir = ROOT / "data" / "test"
    x_train = loadmat(train_dir / "train_data.mat")["train_data"]
    y_train = loadmat(train_dir / "train_label.mat")["train_label"].ravel()
    x_test = loadmat(test_dir / "test_data.mat")["test_data"]
    groups = make_contiguous_groups(y_train)

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
        method="SVM",
        scheme_id=20,
        result=result,
        model=search.best_estimator_,
        results_dir=ROOT / "task1_svm" / "results",
        models_dir=ROOT / "task1_svm" / "models",
        metadata_dir=ROOT / "task1_svm" / "metadata",
        metadata={
            "script_path": Path(__file__),
            "dataset_paths": {
                "training_data": train_dir / "train_data.mat",
                "training_labels": train_dir / "train_label.mat",
                "test_data": test_dir / "test_data.mat",
            },
            "model_family": "SVC with RBF kernel",
            "features": "MinMax-scaled raw pixels",
            "training_sample_count": int(x_train.shape[0]),
            "test_sample_count": int(x_test.shape[0]),
            "best_parameters": search.best_params_,
            "validation": {
                "method": "5-fold GroupKFold over contiguous class groups",
                "accuracy": float(search.best_score_),
            },
        },
    )

    print(f"Best grouped CV accuracy: {search.best_score_:.4%}")
    print(f"Best parameters: {search.best_params_}")
    print(f"Candidate prediction counts: {dict(zip(*np.unique(predictions, return_counts=True)))}")
    for filename in (
        "task_SVM_5_1.txt",
        "task_SVM_8_1.txt",
        "task_SVM_1_1.txt",
    ):
        baseline_path = ROOT / "task1_svm" / "results" / filename
        baseline = np.loadtxt(baseline_path, dtype=int)
        if not np.array_equal(result[:, 0], baseline[:, 0]):
            raise ValueError(f"Sample IDs do not match in {filename}")
        different = result[:, 1] != baseline[:, 1]
        print(f"Different from {filename}: {int(different.sum())} samples")
    print(f"Saved result: {saved['result']}")
    print(f"Saved model: {saved['model']}")
    print(f"Saved metadata: {saved['metadata']}")


if __name__ == "__main__":
    main()
