import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler
from sklearn.svm import SVC

from optimize_candidates import image_transform, hog_features


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.experiment_tracking import save_run


def build_models():
    definitions = {
        "pixel_raw": (image_transform, {"method": "raw"}, 2, 0.01),
        "pixel_zscore": (image_transform, {"method": "zscore"}, 8, 0.01),
        "pixel_percentile": (image_transform, {"method": "percentile"}, 8, 0.01),
        "hog_c3": (hog_features, {"cell": 3, "orientations": 8}, 0.5, "scale"),
    }
    models = {}
    for name, (function, kwargs, c, gamma) in definitions.items():
        models[name] = Pipeline([
            ("features", FunctionTransformer(function, kw_args=kwargs, validate=False)),
            ("scale", MinMaxScaler((-1, 1))),
            ("svm", SVC(kernel="rbf", C=c, gamma=gamma)),
        ])
    return models


def main():
    train_dir = ROOT / "data" / "train"
    test_dir = ROOT / "data" / "test"
    x = loadmat(train_dir / "train_data.mat")["train_data"]
    y = loadmat(train_dir / "train_label.mat")["train_label"].ravel()
    test = loadmat(test_dir / "test_data.mat")["test_data"]
    groups = np.tile(np.repeat(np.arange(5), 80), 2)
    oof = {name: np.empty_like(y) for name in build_models()}
    test_predictions = {}

    for fold, (train_index, valid_index) in enumerate(GroupKFold(5).split(x, y, groups)):
        for name, model in build_models().items():
            model.fit(x[train_index], y[train_index])
            oof[name][valid_index] = model.predict(x[valid_index])
            if fold == 0:
                full_model = build_models()[name]
                full_model.fit(x, y)
                test_predictions[name] = full_model.predict(test)

    for name, prediction in oof.items():
        print(name, "block OOF accuracy", round(float(np.mean(prediction == y)), 5))

    ensembles = {
        "hog_zscore": ["hog_c3", "pixel_zscore"],
        "hog_percentile": ["hog_c3", "pixel_percentile"],
        "raw_hog_zscore": ["pixel_raw", "hog_c3", "pixel_zscore"],
        "four_model_vote": list(oof),
    }
    for name, members in ensembles.items():
        stacked = np.stack([oof[member] for member in members])
        vote = np.where(np.sum(stacked == 1, axis=0) > len(members) / 2, 1, -1)
        accuracy = float(np.mean(vote == y))
        test_stacked = np.stack([test_predictions[member] for member in members])
        test_vote = np.where(np.sum(test_stacked == 1, axis=0) > len(members) / 2, 1, -1)
        result = np.column_stack((np.arange(1, len(test) + 1), test_vote))
        scheme_id = {
            "hog_zscore": 16,
            "hog_percentile": 15,
            "raw_hog_zscore": 17,
            "four_model_vote": 18,
        }[name]
        saved = save_run(
            method="SVM",
            scheme_id=scheme_id,
            result=result,
            model=None,
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
                "model_family": "majority-vote ensemble of SVM classifiers",
                "members": members,
                "training_sample_count": int(x.shape[0]),
                "test_sample_count": int(test.shape[0]),
                "validation": {
                    "method": "5-fold GroupKFold out-of-fold predictions",
                    "accuracy": accuracy,
                },
            },
        )
        print(name, "block OOF accuracy", round(accuracy, 5),
              "test counts", dict(zip(*np.unique(test_vote, return_counts=True))),
              "saved", saved["result"].name)


if __name__ == "__main__":
    main()
