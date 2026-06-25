"""TKTO 実行結果（step ディレクトリ）の収集・集計。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from datasets import load_from_disk

from src.infrastructure.harmony import (
    kto_validation_issues,
    parse_harmony_output,
)


@dataclass(frozen=True)
class StepRef:
    epoch: int
    step: int
    path: Path


_RUN_ID_PATTERN = re.compile(r"^\d{8}_\d{6}$")


def resolve_analysis_root(result_root: str | Path, *, run_id: str | None = None) -> Path:
    """分析対象の run ルートを決める。

    - run_id 指定 → {model}/{run_id}
    - YYYYMMDD_HHMMSS サブディレクトリがあれば最新を選択
    - epoch_* が直下にある → そのまま（旧レイアウト）
    """
    root = Path(result_root)
    if not root.is_dir():
        raise FileNotFoundError(f"結果ディレクトリが見つかりません: {root}")
    if run_id:
        run_dir = root / run_id
        if not run_dir.is_dir():
            raise FileNotFoundError(f"run ディレクトリが見つかりません: {run_dir}")
        return run_dir
    run_dirs = sorted(
        d for d in root.iterdir() if d.is_dir() and _RUN_ID_PATTERN.match(d.name)
    )
    if run_dirs:
        return run_dirs[-1]
    if any(root.glob("epoch_*")):
        return root
    return root


def list_run_dirs(result_root: str | Path) -> list[Path]:
    root = Path(result_root)
    run_dirs = sorted(
        d for d in root.iterdir() if d.is_dir() and _RUN_ID_PATTERN.match(d.name)
    )
    if run_dirs:
        return run_dirs
    if any(root.glob("epoch_*")):
        return [root]
    return []


def discover_steps(result_root: str | Path, *, run_id: str | None = None) -> list[StepRef]:
    """result/{model}/[{run_id}/]epoch_*/step_* を epoch・step 順に列挙する。"""
    root = resolve_analysis_root(result_root, run_id=run_id)

    steps: list[StepRef] = []
    for epoch_dir in sorted(root.glob("epoch_*")):
        epoch_match = re.search(r"epoch_(\d+)$", epoch_dir.name)
        if not epoch_match:
            continue
        epoch = int(epoch_match.group(1))
        for step_dir in sorted(epoch_dir.glob("step_*"), key=_step_sort_key):
            step_match = re.search(r"step_(\d+)$", step_dir.name)
            if not step_match:
                continue
            steps.append(StepRef(epoch=epoch, step=int(step_match.group(1)), path=step_dir))
    return steps


def _step_sort_key(path: Path) -> int:
    match = re.search(r"step_(\d+)$", path.name)
    return int(match.group(1)) if match else 0


def _enrich_from_candidates_csv(df: pd.DataFrame) -> pd.DataFrame:
    """candidates.csv 用の高速 enrich（行ごと apply しない）。"""
    out = df.copy()
    out["thinking_text"] = out.get("thinking", pd.Series(dtype=str)).fillna("").astype(str)
    out["final_text"] = out.get("completion", pd.Series(dtype=str)).fillna("").astype(str)
    out["final_len"] = out["final_text"].str.len()
    out["thinking_len"] = out["thinking_text"].str.len()
    if "kto_valid" in out.columns:
        out["kto_valid"] = out["kto_valid"].astype(bool)
    else:
        out["kto_invalid_reasons"] = ""
        out["kto_valid"] = True
    if "kto_invalid_reasons" not in out.columns:
        out["kto_invalid_reasons"] = ""
    if "label" in out.columns:
        out["label_name"] = out["label"].map({True: "positive", False: "negative"})
    return out


def _enrich_completion_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if {"thinking", "completion", "kto_valid"}.issubset(out.columns):
        return _enrich_from_candidates_csv(out)

    out["final_text"] = out.apply(_final_text, axis=1)
    out["thinking_text"] = out.apply(_thinking_text, axis=1)
    out["final_len"] = out["final_text"].str.len()
    out["thinking_len"] = out["thinking_text"].str.len()
    out["kto_invalid_reasons"] = out.apply(
        lambda r: ";".join(kto_validation_issues(r["thinking_text"], r["final_text"])),
        axis=1,
    )
    out["kto_valid"] = out["kto_invalid_reasons"] == ""
    if "label" in out.columns:
        out["label_name"] = out["label"].map({True: "positive", False: "negative"})
    return out


def _final_text(row: pd.Series) -> str:
    if isinstance(row.get("completion"), str) and row["completion"].strip():
        return row["completion"].strip()
    harmony = row.get("harmony_completion")
    if isinstance(harmony, str) and harmony.strip():
        _, final = parse_harmony_output(harmony)
        if final.strip():
            return final.strip()
    thinking = str(row.get("thinking") or "")
    return thinking.strip()


def _thinking_text(row: pd.Series) -> str:
    if isinstance(row.get("thinking"), str) and row["thinking"].strip():
        return row["thinking"].strip()
    harmony = row.get("harmony_completion")
    if isinstance(harmony, str) and harmony.strip():
        analysis, _ = parse_harmony_output(harmony)
        return analysis.strip()
    return ""


def load_step_candidates(step_dir: str | Path) -> pd.DataFrame | None:
    """train_one_step が保存した candidates.csv（除外含む全候補）を読む。"""
    path = Path(step_dir) / "candidates.csv"
    if not path.is_file():
        return None
    return _enrich_from_candidates_csv(pd.read_csv(path))


def load_step_dataset(step_dir: str | Path) -> pd.DataFrame | None:
    """save_to_disk 済み PCL データセットを DataFrame で返す。未保存なら None。"""
    step_path = Path(step_dir)
    info_path = step_path / "dataset_info.json"
    if not info_path.is_file():
        return None
    df = load_from_disk(str(step_path)).to_pandas()
    return _enrich_completion_metrics(df)


def _quick_summarize_csv(df: pd.DataFrame, *, epoch: int, step: int, path: Path) -> dict:
    n = len(df)
    positive = int(df["label"].sum()) if "label" in df.columns else 0
    final_len = df["completion"].fillna("").astype(str).str.len() if "completion" in df.columns else pd.Series([0])
    valid_rate = float(df["kto_valid"].mean()) if "kto_valid" in df.columns else 0.0
    valid_n = int(df["kto_valid"].sum()) if "kto_valid" in df.columns else 0
    return {
        "epoch": epoch,
        "step": step,
        "path": str(path),
        "n_samples": n,
        "n_positive": positive,
        "n_negative": n - positive,
        "positive_rate": positive / n if n else 0.0,
        "final_len_mean": float(final_len.mean()) if n else 0.0,
        "final_len_p95": float(final_len.quantile(0.95)) if n else 0.0,
        "kto_valid_rate": valid_rate,
        "kto_valid_n": valid_n,
        "dataset_saved": False,
        "source": "candidates.csv",
    }


def summarize_step(df: pd.DataFrame, *, epoch: int, step: int, path: Path) -> dict:
    positive = int((df["label"] == True).sum()) if "label" in df.columns else 0  # noqa: E712
    n = len(df)
    return {
        "epoch": epoch,
        "step": step,
        "path": str(path),
        "n_samples": n,
        "n_positive": positive,
        "n_negative": n - positive,
        "positive_rate": positive / n if n else 0.0,
        "final_len_mean": float(df["final_len"].mean()) if n else 0.0,
        "final_len_p95": float(df["final_len"].quantile(0.95)) if n else 0.0,
        "kto_valid_rate": float(df["kto_valid"].mean()) if n else 0.0,
        "kto_valid_n": int(df["kto_valid"].sum()) if n else 0,
        "dataset_saved": True,
        "source": "dataset",
    }


def _load_step_for_analysis(
    ref: StepRef,
    *,
    load_samples: bool,
) -> tuple[dict, pd.DataFrame | None]:
    """candidates.csv を優先。load_samples=False なら CSV のみでサマリー。"""
    cand_path = ref.path / "candidates.csv"
    if cand_path.is_file():
        raw = pd.read_csv(cand_path)
        summary = _quick_summarize_csv(raw, epoch=ref.epoch, step=ref.step, path=ref.path)
        samples = _enrich_from_candidates_csv(raw) if load_samples else None
        return summary, samples

    if not load_samples:
        info_path = ref.path / "dataset_info.json"
        if not info_path.is_file():
            return _empty_summary(ref), None
        return {
            **_empty_summary(ref),
            "dataset_saved": True,
            "source": "dataset",
        }, None

    df = load_step_dataset(ref.path)
    if df is None:
        return _empty_summary(ref), None
    summary = summarize_step(df, epoch=ref.epoch, step=ref.step, path=ref.path)
    return summary, df


def _empty_summary(ref: StepRef) -> dict:
    return {
        "epoch": ref.epoch,
        "step": ref.step,
        "path": str(ref.path),
        "n_samples": 0,
        "n_positive": 0,
        "n_negative": 0,
        "positive_rate": 0.0,
        "final_len_mean": 0.0,
        "final_len_p95": 0.0,
        "kto_valid_rate": 0.0,
        "kto_valid_n": 0,
        "dataset_saved": False,
        "source": "none",
    }


def collect_run_summary(
    result_root: str | Path,
    *,
    epoch: int | None = None,
    step: int | None = None,
    load_samples: bool = True,
    run_id: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """step サマリー表と（任意で）サンプル DataFrame を返す。

    epoch / step を指定するとその step のみ（高速）。
    load_samples=False なら candidates.csv からサマリーのみ読む。
    """
    refs = discover_steps(result_root, run_id=run_id)
    if epoch is not None:
        refs = [r for r in refs if r.epoch == epoch]
    if step is not None:
        refs = [r for r in refs if r.step == step]

    summaries: list[dict] = []
    frames: list[pd.DataFrame] = []

    for ref in refs:
        summary, df = _load_step_for_analysis(ref, load_samples=load_samples)
        summaries.append(summary)
        if df is not None:
            frames.append(df.assign(epoch=ref.epoch, tkto_step=ref.step))

    summary_df = pd.DataFrame(summaries)
    samples_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return summary_df, samples_df


def load_kto_metrics(step_dir: str | Path) -> pd.DataFrame | None:
    """step 内の最新 checkpoint-* / trainer_state.json から KTO メトリクスを読む。"""
    step_path = Path(step_dir)
    checkpoints = sorted(step_path.glob("checkpoint-*/trainer_state.json"), key=_checkpoint_sort_key)
    if not checkpoints:
        return None

    history: list[dict] = []
    for ckpt_path in checkpoints:
        ckpt_step = int(re.search(r"checkpoint-(\d+)", ckpt_path.parent.name).group(1))  # type: ignore[union-attr]
        with ckpt_path.open(encoding="utf-8") as f:
            state = json.load(f)
        for row in state.get("log_history", []):
            if "loss" not in row:
                continue
            history.append({"checkpoint": ckpt_step, **row})

    if not history:
        return None
    return pd.DataFrame(history)


def summarize_invalid_reasons(df: pd.DataFrame) -> pd.Series:
    """除外理由ごとの件数（複数理由は分解してカウント）。"""
    if df.empty or "kto_invalid_reasons" not in df.columns:
        return pd.Series(dtype=int)
    invalid = df[~df["kto_valid"].fillna(True)]
    if invalid.empty:
        return pd.Series(dtype=int)
    reasons = invalid["kto_invalid_reasons"].astype(str).str.split(";").explode()
    reasons = reasons[reasons.str.len() > 0]
    return reasons.value_counts()


def _checkpoint_sort_key(path: Path) -> int:
    match = re.search(r"checkpoint-(\d+)", str(path))
    return int(match.group(1)) if match else 0
