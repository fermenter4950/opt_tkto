"""CV fold の再利用（1-fold 済み成果から 5-fold を続行する）。"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

_ADAPTER_MARKERS = ("adapter_config.json", "adapter_model.safetensors", "adapter_model.bin")


def fold_is_complete(fold_dir: Path) -> bool:
    """学習済み fold として metrics / predictions / adapter が揃っているか。"""
    if not fold_dir.is_dir():
        return False
    if not (fold_dir / "metrics.json").is_file():
        return False
    if not (fold_dir / "predictions.csv").is_file():
        return False
    return any((fold_dir / name).is_file() for name in _ADAPTER_MARKERS)


def load_fold_results(fold_dir: Path) -> tuple[dict, pd.DataFrame]:
    with open(fold_dir / "metrics.json", encoding="utf-8") as f:
        result = json.load(f)
    pred_df = pd.read_csv(fold_dir / "predictions.csv")
    return result, pred_df


def copy_fold_dir(src_dir: Path, dst_dir: Path) -> None:
    """fold 成果物を別 RUN ディレクトリへコピーする。"""
    dst_dir.mkdir(parents=True, exist_ok=True)
    for path in src_dir.iterdir():
        if path.name == "_checkpoints":
            continue
        target = dst_dir / path.name
        if path.is_dir():
            shutil.copytree(path, target, dirs_exist_ok=True)
        else:
            shutil.copy2(path, target)


def resolve_resume_model_dir(
    *,
    model_parent: Path,
    current_run: str,
    resume_from_run: str,
) -> Path | None:
    """RESUME_FROM_RUN を解釈して参照先ディレクトリを返す。

    - 空: なし
    - ``auto``: ``*_5fold`` RUN なら対応する ``*_1fold`` を探す
    - 絶対/相対パス: そのまま
    - それ以外: ``model_parent / resume_from_run``
    """
    raw = resume_from_run.strip()
    if not raw:
        return None
    if raw.lower() == "auto":
        if current_run.endswith("_5fold"):
            candidate = model_parent / current_run.replace("_5fold", "_1fold", 1)
            return candidate if candidate.is_dir() else None
        return None
    path = Path(raw)
    if path.is_dir():
        return path
    return model_parent / raw


def try_reuse_fold(
    fold: int,
    *,
    model_dir: Path,
    resume_model_dir: Path | None,
    skip_existing: bool,
) -> tuple[dict, pd.DataFrame] | None:
    """既存 fold を再利用できるなら (metrics, predictions) を返す。"""
    dest = model_dir / f"fold_{fold}"

    if skip_existing and fold_is_complete(dest):
        print(f"[resume] fold {fold}: 既存成果を利用 → {dest}", flush=True)
        return load_fold_results(dest)

    if resume_model_dir is not None:
        src = resume_model_dir / f"fold_{fold}"
        if fold_is_complete(src):
            print(f"[resume] fold {fold}: {src} から取り込み → {dest}", flush=True)
            copy_fold_dir(src, dest)
            return load_fold_results(dest)

    return None
