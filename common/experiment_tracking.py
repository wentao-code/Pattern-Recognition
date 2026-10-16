from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np


_METHODS = {"SVM", "MLP"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def next_iteration(
    results_dir: Path,
    method: str,
    scheme_id: int,
    models_dir: Path | None = None,
    metadata_dir: Path | None = None,
) -> int:
    if method not in _METHODS or scheme_id < 1:
        raise ValueError("method must be SVM or MLP and scheme_id must be positive")
    pattern = re.compile(rf"^task_{method}_{scheme_id}_(\d+)(?:_labels)?\.[^.]+$", re.IGNORECASE)
    directories = tuple(
        directory for directory in (results_dir, models_dir, metadata_dir) if directory is not None
    )
    iterations = [
        int(match.group(1))
        for directory in directories
        for path in directory.glob(f"task_{method}_{scheme_id}_*")
        if (match := pattern.match(path.name))
    ]
    return max(iterations, default=0) + 1


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"Cannot serialize {type(value).__name__} to JSON")


def save_run(
    method: str,
    scheme_id: int,
    result: np.ndarray,
    model: Any | None,
    results_dir: Path,
    models_dir: Path,
    metadata_dir: Path,
    metadata: dict[str, Any],
) -> dict[str, Path]:
    if method not in _METHODS or scheme_id < 1:
        raise ValueError("method must be SVM or MLP and scheme_id must be positive")

    result = np.asarray(result)
    if result.ndim != 2 or result.shape[1] != 2 or result.shape[0] == 0:
        raise ValueError("result must be a non-empty two-column array")
    if not np.isfinite(result).all() or not np.equal(result, np.floor(result)).all():
        raise ValueError("result must contain finite integer values")
    if not np.array_equal(result[:, 0], np.arange(1, result.shape[0] + 1)):
        raise ValueError("result sample IDs must be contiguous and start at 1")
    if not np.isin(result[:, 1], (-1, 1)).all():
        raise ValueError("result labels must be -1 or +1")

    script_path = Path(metadata["script_path"]).resolve()
    dataset_paths = {
        name: Path(path).resolve()
        for name, path in metadata["dataset_paths"].items()
    }
    for path in (script_path, *dataset_paths.values()):
        if not path.is_file():
            raise FileNotFoundError(path)

    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    iteration = next_iteration(
        results_dir,
        method,
        scheme_id,
        models_dir=models_dir,
        metadata_dir=metadata_dir,
    )
    basename = f"task_{method}_{scheme_id}_{iteration}"
    result_path = results_dir / f"{basename}.txt"
    model_path = models_dir / f"{basename}.joblib" if model is not None else None
    metadata_path = metadata_dir / f"{basename}.json"
    targets = [result_path, metadata_path]
    if model_path is not None:
        targets.append(model_path)
    collisions = [path for path in targets if path.exists()]
    if collisions:
        raise FileExistsError(f"Refusing to overwrite: {', '.join(map(str, collisions))}")

    np.savetxt(result_path, result.astype(np.int64), fmt="%d", delimiter=" ")
    if model_path is not None:
        joblib.dump(model, model_path)

    predicted_labels, counts = np.unique(result[:, 1].astype(int), return_counts=True)
    record = dict(metadata)
    record.pop("dataset_paths")
    record.update(
        {
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "method": method,
            "scheme_id": scheme_id,
            "iteration": iteration,
            "script_path": str(script_path),
            "script_sha256": sha256_file(script_path),
            "datasets": {
                name: {"path": str(path), "sha256": sha256_file(path)}
                for name, path in dataset_paths.items()
            },
            "sample_count": int(result.shape[0]),
            "predicted_label_counts": {
                str(int(label)): int(count)
                for label, count in zip(predicted_labels, counts)
            },
            "result_path": str(result_path),
            "model_path": str(model_path) if model_path is not None else None,
            "metadata_path": str(metadata_path),
        }
    )
    metadata_path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    saved = {"result": result_path, "metadata": metadata_path}
    if model_path is not None:
        saved["model"] = model_path
    return saved
