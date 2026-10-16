from pathlib import Path

import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler
from sklearn.svm import SVC

from optimize_candidates import image_transform, hog_features


ROOT = Path(__file__).resolve().parent


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
    x = loadmat(ROOT / "train(Task_2)" / "train_data.mat")["train_data"]
    y = loadmat(ROOT / "train(Task_2)" / "train_label.mat")["train_label"].ravel()
    test = loadmat(ROOT / "test(Task_2)" / "test_data.mat")["test_data"]
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
        np.savetxt(ROOT / f"task1_result_{name}.txt", result, fmt="%d", delimiter=" ")
        print(name, "block OOF accuracy", round(accuracy, 5),
              "test counts", dict(zip(*np.unique(test_vote, return_counts=True))))


if __name__ == "__main__":
    main()
