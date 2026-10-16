import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.io import loadmat
from sklearn.model_selection import GridSearchCV, GroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler
from sklearn.svm import SVC
from skimage.feature import hog


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.experiment_tracking import save_run


def image_transform(data, method):
    images = data.reshape(-1, 19, 19).astype(float)
    if method == "raw":
        return data
    if method == "demean":
        return (images - images.mean(axis=(1, 2), keepdims=True)).reshape(len(data), -1)
    if method == "zscore":
        centered = images - images.mean(axis=(1, 2), keepdims=True)
        return (centered / (images.std(axis=(1, 2), keepdims=True) + 1e-6)).reshape(len(data), -1)
    if method == "percentile":
        lo = np.percentile(images, 5, axis=(1, 2), keepdims=True)
        hi = np.percentile(images, 95, axis=(1, 2), keepdims=True)
        normalized = (images - lo) / (hi - lo + 1e-6)
        return np.clip(normalized, -1, 2).reshape(len(data), -1)
    if method == "local_contrast":
        smooth = gaussian_filter(images, sigma=(0, 1.0, 1.0))
        residual = images - smooth
        local_energy = np.sqrt(gaussian_filter(residual**2, sigma=(0, 1.5, 1.5)) + 1e-5)
        return (residual / local_energy).reshape(len(data), -1)
    raise ValueError(method)


def hog_features(data, cell, orientations):
    return np.stack([
        hog(row.reshape(19, 19), orientations=orientations,
            pixels_per_cell=(cell, cell), cells_per_block=(2, 2))
        for row in data
    ])


def main():
    train_dir = ROOT / "data" / "train"
    test_dir = ROOT / "data" / "test"
    x = loadmat(train_dir / "train_data.mat")["train_data"]
    y = loadmat(train_dir / "train_label.mat")["train_label"].ravel()
    test = loadmat(test_dir / "test_data.mat")["test_data"]
    scheme_ids = {
        "pixel_raw": 3,
        "pixel_demean": 4,
        "pixel_zscore": 5,
        "pixel_percentile": 6,
        "pixel_local_contrast": 7,
        "hog_c3_o8": 8,
        "hog_c4_o8": 9,
        "hog_c4_o9": 10,
        "hog_c5_o8": 11,
        "hog_c5_o12": 12,
        "hog_c6_o8": 13,
    }
    groups = np.tile(np.repeat(np.arange(5), 80), 2)
    splitters = {
        "block": list(GroupKFold(5).split(x, y, groups)),
        "random": list(StratifiedKFold(5, shuffle=True, random_state=42).split(x, y)),
    }
    candidates = []
    for method in ["raw", "demean", "zscore", "percentile", "local_contrast"]:
        transform = FunctionTransformer(image_transform, kw_args={"method": method}, validate=False)
        pipe = Pipeline([("features", transform), ("scale", MinMaxScaler((-1, 1))), ("svm", SVC())])
        grid = {"svm__kernel": ["rbf"], "svm__C": [0.5, 2, 8, 32],
                "svm__gamma": ["scale", 0.001, 0.003, 0.01, 0.03, 0.1]}
        candidates.append((f"pixel_{method}", pipe, grid))
    for cell, orientations in [(3, 8), (4, 8), (4, 9), (5, 8), (5, 12), (6, 8)]:
        transform = FunctionTransformer(hog_features, kw_args={"cell": cell, "orientations": orientations}, validate=False)
        pipe = Pipeline([("features", transform), ("svm", SVC())])
        grid = {"svm__kernel": ["rbf"], "svm__C": [0.5, 2, 8, 32],
                "svm__gamma": ["scale", 0.003, 0.01, 0.03, 0.1]}
        candidates.append((f"hog_c{cell}_o{orientations}", pipe, grid))

    summaries = []
    for name, pipe, grid in candidates:
        scores = {}
        best_models = {}
        for cv_name, cv in splitters.items():
            search = GridSearchCV(pipe, grid, scoring="accuracy", cv=cv, n_jobs=-1, refit=True)
            search.fit(x, y)
            scores[cv_name] = search.best_score_
            best_models[cv_name] = search
        block_prediction = best_models["block"].predict(test).astype(int)
        output = np.column_stack((np.arange(1, len(test) + 1), block_prediction))
        saved = save_run(
            method="SVM",
            scheme_id=scheme_ids[name],
            result=output,
            model=best_models["block"].best_estimator_,
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
                "model_family": "SVC with RBF kernel",
                "feature_pipeline": name,
                "training_sample_count": int(x.shape[0]),
                "test_sample_count": int(test.shape[0]),
                "validation": {
                    cv_name: {
                        "accuracy": float(scores[cv_name]),
                        "best_parameters": best_models[cv_name].best_params_,
                    }
                    for cv_name in scores
                },
            },
        )
        print(name, "block", round(scores["block"], 5), best_models["block"].best_params_,
              "random", round(scores["random"], 5), best_models["random"].best_params_,
              "test label counts", dict(zip(*np.unique(block_prediction, return_counts=True))),
              "saved", saved["result"].name)
        summaries.append((scores["block"], name, best_models["block"]))
    print("BEST BLOCK VALIDATION", [(round(score, 5), name) for score, name, _ in sorted(summaries, reverse=True)[:10]])


if __name__ == "__main__":
    main()
