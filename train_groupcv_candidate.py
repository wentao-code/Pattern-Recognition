from pathlib import Path

import joblib
import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

from train_svm import make_contiguous_groups


ROOT = Path(__file__).resolve().parent


def main():
    train_dir = ROOT / "train(Task_2)"
    test_dir = ROOT / "test(Task_2)"
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
    output_path = ROOT / "task1_result_train_optimized.txt"
    np.savetxt(output_path, result, fmt="%d", delimiter=" ")
    joblib.dump(search.best_estimator_, ROOT / "train_groupcv_model.joblib")

    print(f"Best grouped CV accuracy: {search.best_score_:.4%}")
    print(f"Best parameters: {search.best_params_}")
    print(f"Candidate prediction counts: {dict(zip(*np.unique(predictions, return_counts=True)))}")
    for filename in (
        "task1_result_pixel_zscore.txt",
        "task1_result_hog_c3_o8.txt",
        "task1_result_raw_pixel_rbf.txt",
    ):
        baseline = np.loadtxt(ROOT / filename, dtype=int)
        if not np.array_equal(result[:, 0], baseline[:, 0]):
            raise ValueError(f"Sample IDs do not match in {filename}")
        different = result[:, 1] != baseline[:, 1]
        print(f"Different from {filename}: {int(different.sum())} samples")
    print(f"Saved: {output_path.name}, train_groupcv_model.joblib")


if __name__ == "__main__":
    main()
