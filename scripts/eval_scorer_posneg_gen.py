"""保存済み生成SFT adapter を読み込み、検証セットの正解率だけを再計算する。

学習（train_scorer_posneg_gen.py）が評価直前にクラッシュした場合などに、
再学習せず checkpoint からメトリクスだけを出すための補助スクリプト。

train_scorer_posneg_gen.py と同じ設定（SCORER_NAME / RUN_NAME / MIN_RATERS /
POS_THRESHOLD / SEED / N_FOLDS）を環境変数で揃えて実行すること。fold 分割も
学習時と同一の StratifiedGroupKFold で再現する。

    SCORER_NAME=tokyotech-llm/Qwen3-Swallow-32B-RL-v0.2 GPU_ID=1 USE_4BIT=1 \
    RUN_NAME=posneg_gen_fp3_32b_1fold \
    ADAPTER_DIR=models/scorer/Qwen3-Swallow-32B-RL-v0.2/posneg_gen_fp3_32b_1fold/fold_0/_checkpoints/checkpoint-795 \
    EVAL_FOLD=0 \
    .venv/bin/python scripts/eval_scorer_posneg_gen.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import train_scorer_posneg_gen as T  # 設定・関数を流用（import 時にモデルは読まない）

import torch
from peft import PeftModel
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from transformers import AutoModelForCausalLM, BitsAndBytesConfig


def _metrics_at(y: np.ndarray, prob: np.ndarray, prob_th: float) -> dict:
    pred = (prob >= prob_th).astype(int)
    yt = y.astype(bool)
    yp = pred.astype(bool)
    tp = int((yt & yp).sum())
    tn = int((~yt & ~yp).sum())
    fp = int((~yt & yp).sum())
    fn = int((yt & ~yp).sum())
    n = len(y)
    pos_prec = tp / (tp + fp) if tp + fp else 0.0
    pos_rec = tp / (tp + fn) if tp + fn else 0.0
    neg_prec = tn / (tn + fn) if tn + fn else 0.0
    neg_rec = tn / (tn + fp) if tn + fp else 0.0
    pos_f1 = 2 * pos_prec * pos_rec / (pos_prec + pos_rec) if pos_prec + pos_rec else 0.0
    neg_f1 = 2 * neg_prec * neg_rec / (neg_prec + neg_rec) if neg_prec + neg_rec else 0.0
    return {
        "n": n,
        "accuracy": (tp + tn) / n if n else 0.0,
        "macro_f1": (pos_f1 + neg_f1) / 2,
        "pos_prec": pos_prec,
        "pos_rec": pos_rec,
        "pred_pos": int(pred.sum()),
    }


def analyze_confidence(pred_df: pd.DataFrame) -> dict:
    """margin / log_label_mass と prob_positive の組み合わせで閾値候補を探索する。"""
    y = pred_df["true_label"].to_numpy()
    prob = pred_df["prob_positive"].to_numpy()
    margin = pred_df["margin"].to_numpy()
    log_mass = pred_df["log_label_mass"].to_numpy()

    print("\n" + "=" * 50)
    print("確信度指標の分布")
    print("=" * 50)
    for arr, name in [
        (margin, "margin = |lp_pos - lp_neg|（2語の尤度差）"),
        (log_mass, "log_label_mass = log P(pos)+P(neg)"),
    ]:
        print(f"\n  [{name}]")
        for label, lname in [(1, "pos true"), (0, "neg true"), (None, "all")]:
            sub = arr if label is None else arr[y == label]
            print(
                f"    {lname:8s}: mean={sub.mean():.4f} median={np.median(sub):.4f} "
                f"p25={np.percentile(sub,25):.4f} p75={np.percentile(sub,75):.4f}"
            )

    print("\nmargin 帯域ごとの精度（prob_th=0.45 固定）")
    edges = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 10.0]
    band_rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (margin >= lo) & (margin < hi)
        if mask.sum() == 0:
            continue
        m = _metrics_at(y[mask], prob[mask], 0.45)
        band_rows.append({"margin_lo": lo, "margin_hi": hi, **m})
        print(
            f"  margin∈[{lo:.2f},{hi:.2f}): n={m['n']:4d} acc={m['accuracy']:.3f} "
            f"macroF1={m['macro_f1']:.3f} pos_rec={m['pos_rec']:.3f} pred_pos={m['pred_pos']}"
        )

    print("\nmargin 閾値 × prob 閾値（macro-F1 最大化）")
    best = None
    margin_ths = sorted(set(round(x, 3) for x in np.quantile(margin, [0.1, 0.25, 0.5, 0.75, 0.9, 0.95])))
    prob_ths = list(np.linspace(0.40, 0.55, 16))
    for mt in margin_ths:
        mask = margin >= mt
        if mask.sum() < 30:
            continue
        for pt in prob_ths:
            m = _metrics_at(y[mask], prob[mask], pt)
            row = {"margin_th": mt, "prob_th": pt, **m}
            if best is None or m["macro_f1"] > best["macro_f1"]:
                best = row

    if best:
        print(
            f"  最良: margin>={best['margin_th']:.3f} & prob>={best['prob_th']:.3f} → "
            f"n={best['n']} acc={best['accuracy']:.3f} macroF1={best['macro_f1']:.3f} "
            f"pos_prec={best['pos_prec']:.3f} pos_rec={best['pos_rec']:.3f}"
        )
        print(f"  カバレッジ: {best['n']/len(y):.1%}（残りは「判定保留」）")

    print("\n高確信ラベル付けの提案（3段階, prob>=0.45）")
    tier_rows = []
    for tier_name, mt in [("高確信", 0.875), ("中確信", 0.500), ("低確信", 0.000)]:
        mask = margin >= mt
        m = _metrics_at(y[mask], prob[mask], 0.45)
        tier_rows.append({"tier": tier_name, "margin_th": mt, "prob_th": 0.45, **m})
        print(
            f"  {tier_name}: margin>={mt:.2f} → n={m['n']:4d} ({m['n']/len(y):.1%}) "
            f"acc={m['accuracy']:.3f} pos_prec={m['pos_prec']:.3f} pos_rec={m['pos_rec']:.3f}"
        )

    return {"band_metrics": band_rows, "grid_best": best, "tiers": tier_rows}


def _is_zeroshot(adapter_dir: str) -> bool:
    """ADAPTER_DIR が none/base/空 のとき、素のベースモデル評価（zero-shot）。"""
    return adapter_dir.strip().lower() in {"", "none", "base", "zeroshot", "zero-shot"}


def load_base_with_adapter(adapter_dir: str, tokenizer):
    quant_config = None
    if T.USE_4BIT:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    base = AutoModelForCausalLM.from_pretrained(
        T.MODEL_ID,
        torch_dtype=torch.bfloat16,
        quantization_config=quant_config,
        device_map={"": 0} if T.USE_4BIT else None,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    base.config.pad_token_id = tokenizer.pad_token_id
    if _is_zeroshot(adapter_dir):
        print("[zero-shot] アダプタを読み込まず、素のベースモデルで評価します", flush=True)
        model = base
    else:
        model = PeftModel.from_pretrained(base, adapter_dir)
    if not T.USE_4BIT:
        model.to(T.DEVICE)
    model.eval()
    return model


def get_val_df(eval_fold: int):
    df = T.prepare_dataframe()
    sgkf = StratifiedGroupKFold(
        n_splits=max(T.N_FOLDS, 2), shuffle=True, random_state=T.SEED
    )
    for fold, (_, val_idx) in enumerate(
        sgkf.split(df, df["label"], groups=df["message_id"])
    ):
        if fold == eval_fold:
            return df.iloc[val_idx].reset_index(drop=True)
    raise ValueError(f"fold {eval_fold} が見つかりません")


def main() -> None:
    adapter_dir = os.getenv(
        "ADAPTER_DIR",
        str(T.MODEL_DIR / "fold_0" / "_checkpoints" / "checkpoint-795"),
    )
    eval_fold = int(os.getenv("EVAL_FOLD", "0"))
    print(f"adapter: {adapter_dir}")
    print(f"base   : {T.MODEL_ID}")
    print(f"eval fold: {eval_fold}")

    tokenizer = T.build_tokenizer()
    val_df = get_val_df(eval_fold)
    n_pos = int(val_df["label"].sum())
    print(f"検証セット {len(val_df)} 行（pos={n_pos} / neg={len(val_df) - n_pos}）")

    model = load_base_with_adapter(adapter_dir, tokenizer)

    scores = T.predict_prob_positive(model, tokenizer, val_df)
    prob_pos = scores["prob_positive"]
    labels = val_df["label"].to_numpy()
    m = T.classification_metrics(prob_pos, labels)
    auc = roc_auc_score(labels, prob_pos)

    print("=" * 50)
    print(f"accuracy         : {m['accuracy']:.4f}")
    print(f"positive recall  : {m['positive_accuracy']:.4f}")
    print(f"negative recall  : {m['negative_accuracy']:.4f}")
    print(f"macro F1         : {m['macro_f1']:.4f}")
    print(f"AUC              : {auc:.4f}")
    print(f"tp={m['tp']} tn={m['tn']} fp={m['fp']} fn={m['fn']}")
    print("=" * 50)

    if _is_zeroshot(adapter_dir):
        out_dir = Path(os.getenv("OUT_DIR", str(T.MODEL_DIR.parent / "posneg_gen_zeroshot" / f"fold_{eval_fold}")))
    else:
        out_dir = Path(os.getenv("OUT_DIR", str(Path(adapter_dir).parent.parent)))  # fold_x/
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_df = val_df[T.GROUP_KEYS + ["acceptance_s", "n_raters", "label"]].copy()
    pred_df = pred_df.rename(columns={"label": "true_label"}).reset_index(drop=True)
    pred_df["prob_positive"] = prob_pos
    pred_df["log_label_mass"] = scores["log_label_mass"]
    pred_df["margin"] = scores["margin"]
    pred_df["logprob_pos"] = scores["logprob_pos"]
    pred_df["logprob_neg"] = scores["logprob_neg"]
    pred_df["pred_label"] = (prob_pos >= 0.5).astype(int)
    pred_df["correct"] = pred_df["true_label"] == pred_df["pred_label"]

    confidence_analysis = analyze_confidence(pred_df)

    pred_df.to_csv(out_dir / "predictions.csv", index=False)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"fold": eval_fold, "auc": auc, **m, "confidence_analysis": confidence_analysis}, f, ensure_ascii=False, indent=2)
    print(f"\n保存: {out_dir/'predictions.csv'} , {out_dir/'metrics.json'}")


if __name__ == "__main__":
    main()
