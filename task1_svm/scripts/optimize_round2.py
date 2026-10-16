import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler
from sklearn.svm import SVC

from optimize_candidates import hog_features, image_transform
from train_svm import make_contiguous_groups


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.experiment_tracking import save_run


BASELINES = (
    "task_SVM_5_1.txt",
    "task_SVM_8_1.txt",
    "task_SVM_1_1.txt",
)


def candidates():
    items = []
    for method in ("raw", "demean", "zscore", "percentile", "local_contrast"):
        pipe = Pipeline(
            [
                ("features", FunctionTransformer(image_transform, kw_args={"method": method}, validate=False)),
                ("scale", MinMaxScaler(feature_range=(-1, 1))),
                ("svm", SVC(kernel="rbf")),
            ]
        )
        grid = {
            "svm__C": [0.5, 2, 8, 32],
            "svm__gamma": ["scale", 0.001, 0.003, 0.01, 0.03, 0.1],
        }
        items.append((f"pixel_{method}", pipe, grid))

    for cell, orientations in ((3, 8), (4, 8), (4, 9), (5, 8), (5, 12), (6, 8)):
        pipe = Pipeline(
            [
                (
                    "features",
                    FunctionTransformer(
                        hog_features,
                        kw_args={"cell": cell, "orientations": orientations},
                        validate=False,
                    ),
                ),
                ("scale", MinMaxScaler(feature_range=(-1, 1))),
                ("svm", SVC(kernel="rbf")),
            ]
        )
        grid = {
            "svm__C": [0.5, 2, 8, 32],
            "svm__gamma": ["scale", 0.003, 0.01, 0.03, 0.1],
        }
        items.append((f"hog_c{cell}_o{orientations}", pipe, grid))
    return items


def main():
    train_dir = ROOT / "data" / "train"
    test_dir = ROOT / "data" / "test"
    x_train = loadmat(train_dir / "train_data.mat")["train_data"]
    y_train = loadmat(train_dir / "train_label.mat")["train_label"].ravel()
    x_test = loadmat(test_dir / "test_data.mat")["test_data"]
    groups = make_contiguous_groups(y_train)
    cv = GroupKFold(n_splits=5)

    summaries = []
    for name, pipeline, grid in candidates():
        search = GridSearchCV(
            pipeline,
            grid,
            scoring="accuracy",
            cv=cv,
            n_jobs=-1,
            refit=True,
            return_train_score=False,
        )
        search.fit(x_train, y_train, groups=groups)
        summaries.append((search.best_score_, name, search))
        print(f"{name}: grouped CV={search.best_score_:.4%}, params={search.best_params_}", flush=True)

    score, name, best_search = max(summaries, key=lambda item: item[0])
    predictions = best_search.best_estimator_.predict(x_test).astype(int)
    result = np.column_stack((np.arange(1, predictions.size + 1), predictions))
    saved = save_run(
        method="SVM",
        scheme_id=21,
        result=result,
        model=best_search.best_estimator_,
        results_dir=ROOT / "task1_svm" / "results",
        models_dir=ROOT / "task1_svm" / "models",
        metadata_dir=ROOT / "task1_svm" / "metadata",
        accuracy_dir=ROOT / "accuracy",
        metadata={
            "script_path": Path(__file__),
            "dataset_paths": {
                "training_data": train_dir / "train_data.mat",
                "training_labels": train_dir / "train_label.mat",
                "test_data": test_dir / "test_data.mat",
            },
            "model_family": "SVC with RBF kernel selected from pixel and HOG candidates",
            "selected_candidate": name,
            "training_sample_count": int(x_train.shape[0]),
            "test_sample_count": int(x_test.shape[0]),
            "best_parameters": best_search.best_params_,
            "validation": {
                "method": "5-fold GroupKFold over contiguous class groups",
                "accuracy": float(score),
            },
        },
    )

    print(f"Selected: {name}; grouped CV accuracy={score:.4%}")
    print(f"Best parameters: {best_search.best_params_}")
    print(f"Prediction counts: {dict(zip(*np.unique(predictions, return_counts=True)))}")
    for filename in BASELINES:
        baseline_path = ROOT / "task1_svm" / "results" / filename
        baseline = np.loadtxt(baseline_path, dtype=int)
        if baseline.shape != result.shape or not np.array_equal(result[:, 0], baseline[:, 0]):
            raise ValueError(f"Sample IDs or row count differ in {filename}")
        print(f"Different from {filename}: {int(np.count_nonzero(result[:, 1] != baseline[:, 1]))} samples")
    print(f"Saved result: {saved['result']}")
    print(f"Saved model: {saved['model']}")
    print(f"Saved metadata: {saved['metadata']}")


if __name__ == "__main__":
    main()
