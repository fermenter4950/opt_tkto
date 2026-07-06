"""受容度（intention）の positive/negative 2値分類スコアラー fine-tuning。

KTO 用に「メッセージ＋受信者属性 → 受信者が運動に前向きになるか（positive/negative）」
を 2 値分類で予測する採点器を LoRA で学習する。

設計（元 tkto / paper のベースラインを踏襲しつつ KTO 向けに調整）:
- 受容度は intention（実際に運動しようと思う）単独。
- メッセージ×受信者属性（プロンプトに与える gender/age_group/stage）ごとに
  回答者の intention を加重平均（＝paper 式1）して受容度 s を作る。
- 回答者数が MIN_RATERS 未満のグループは「平均が不安定」なので除外する。
- 受容度 s > POS_THRESHOLD を positive、それ以外を negative とする（中立クラスなし）。
  POS_THRESHOLD=4.0 は 7 件法の中点。ちょうど 4.0 は negative 側になる。
- 2 値分類（CrossEntropy + class_weight で不均衡対策）。
  出力は positive 確率も保存するので、判定の確率閾値は後から調整できる。
- メッセージのリーク防止に StratifiedGroupKFold（groups=message_id）で分割する。

各 fold で accuracy / positive 再現率 / negative 再現率 / macro-F1 を算出し、
1 件ごとの予測（positive 確率・予測ラベル）も保存する。

    python scripts/train_scorer_posneg.py

環境変数で上書き可能:
    SCORER_NAME    採点器名（保存先の階層・読み込むモデル。既定: Qwen3-Swallow-8B-RL-v0.2）
    BASE_MODEL_PATH ベースモデルのローカルパス or HF ID（SCORER_NAME より優先）
    GPU_ID         使用 GPU（0 か 1。既定: 1）
    USE_4BIT       1 で 4bit QLoRA（既定）、0 で bf16 そのまま
    MIN_RATERS     集約に残す最小回答者数（既定: 3）
    POS_THRESHOLD  この値より大きい受容度を positive とみなす（既定: 4.0）
    MAX_FOLDS      学習する fold 数の上限（既定: N_FOLDS=5。1 なら 1 fold だけ）
    RESUME_FROM_RUN  完了済み fold の参照先ディレクトリ（分類版は通常不要。
                     同一 ``posneg_intention/`` 内の fold を SKIP_EXISTING_FOLDS で再利用）
    SKIP_EXISTING_FOLDS  1（既定）で model_dir 内の完了済み fold を再学習しない。
    QUICK_MODE     1 で 1 fold・少数サンプルの動作確認
"""

from __future__ import annotations

import gc
import json
import os
import socket
import traceback
import urllib.request
from dataclasses import dataclass
from pathlib import Path

# 使用する GPU を明示指定（0 か 1）。環境変数 GPU_ID で上書き可。
# torch import 前に CUDA_VISIBLE_DEVICES を設定する必要がある。
GPU_ID = os.getenv("GPU_ID", "1")
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID
print(f"使用 GPU: cuda:{GPU_ID}（CUDA_VISIBLE_DEVICES={GPU_ID}）")

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

from scorer_fold_resume import resolve_resume_model_dir, try_reuse_fold

from src.domain.scorer_prompt import format_scorer_prompt

# ---------------------------------------------------------------------------
# 設定
# ---------------------------------------------------------------------------
SCORER_NAME = os.getenv("SCORER_NAME", "tokyotech-llm/Qwen3-Swallow-8B-RL-v0.2")
TASK = "posneg_intention"  # intention 受容度の pos/neg 2値分類
RESUME_FROM_RUN = os.getenv("RESUME_FROM_RUN", "")
SKIP_EXISTING_FOLDS = os.getenv("SKIP_EXISTING_FOLDS", "1") == "1"

USE_4BIT = os.getenv("USE_4BIT", "1") == "1"

N_FOLDS = 5
SEED = 42
# 学習する fold 数の上限（分割は N_FOLDS のまま）。MAX_FOLDS=1 で 1 fold だけ。
MAX_FOLDS = int(os.getenv("MAX_FOLDS", str(N_FOLDS)))

QUICK_MODE = os.getenv("QUICK_MODE", "0") == "1"
QUICK_FOLDS = 1
QUICK_MAX_SAMPLES = 400

# 受容度の元次元（前向きになれるか＝intention 単独）
SOURCE_DIM = "intention"
# 集約キー（メッセージ×プロンプトに与える受信者属性）
GROUP_KEYS = ["message_id", "gender", "age_group", "stage"]
# 集約に残す最小回答者数。これ未満のグループは平均が不安定なので除外
MIN_RATERS = int(os.getenv("MIN_RATERS", "3"))
# 受容度 > POS_THRESHOLD を positive。4.0 は 7件法の中点（ちょうど4は negative 側）
POS_THRESHOLD = float(os.getenv("POS_THRESHOLD", "4.0"))

MAX_LENGTH = 512

# 学習ハイパラ。実効バッチ = BATCH_SIZE * GRAD_ACCUM = 16
NUM_EPOCHS = 3
BATCH_SIZE = 4
GRAD_ACCUM = 4
LR = 2e-4
LORA_R = 32
LORA_ALPHA = 16

REPO_ROOT = Path(__file__).resolve().parent.parent

DATA_CSV = REPO_ROOT / "data" / "scorer_finetune_raw" / "all_splits.csv"
MODEL_ID = os.getenv("BASE_MODEL_PATH", SCORER_NAME)
OUTPUT_DIR = REPO_ROOT / "models" / "scorer"
DIR_NAME = Path(SCORER_NAME).name
MODEL_DIR = OUTPUT_DIR / DIR_NAME / TASK  # 例: models/scorer/Qwen3-Swallow-8B-RL-v0.2/posneg_intention

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_text(row) -> str:
    return format_scorer_prompt(
        message=row["メッセージ"],
        sex=row["gender"],
        age=row["age_group"],
        stage=row["stage"],
    )


def slack_notify(message: str) -> None:
    """SLACK_WEBHOOK_URL があれば Slack へ通知。未設定・失敗時も学習は止めない。"""
    url = os.getenv("SLACK_WEBHOOK_URL")
    if not url:
        return
    text = f"[{socket.gethostname()}] [{DIR_NAME}/{TASK}] {message}"
    try:
        data = json.dumps({"text": text}).encode("utf-8")
        req = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"}
        )
        urllib.request.urlopen(req, timeout=15)
    except Exception as e:  # noqa: BLE001 通知失敗で学習を落とさない
        print(f"[slack] 通知失敗: {e}", flush=True)


def prepare_dataframe() -> pd.DataFrame:
    """CSV を読み込み、メッセージ×属性で集約して受容度→pos/neg ラベルを作る。

    - 受容度 s = グループ内の intention 平均（paper 式1 の加重平均と等価）。
    - 回答者数 < MIN_RATERS のグループは除外（平均が不安定なため）。
    - s > POS_THRESHOLD を positive(1)、それ以外を negative(0)。
    """
    raw = pd.read_csv(DATA_CSV)
    agg = (
        raw.groupby(GROUP_KEYS)
        .agg(
            acceptance_s=(SOURCE_DIM, "mean"),
            n_raters=(SOURCE_DIM, "size"),
            メッセージ=("メッセージ", "first"),
        )
        .reset_index()
    )
    before = len(agg)
    agg = agg[agg["n_raters"] >= MIN_RATERS].reset_index(drop=True)
    print(f"集約 {before} → n_raters>={MIN_RATERS} で {len(agg)} グループに絞り込み")

    agg["text"] = agg.apply(to_text, axis=1)
    agg["label"] = (agg["acceptance_s"] > POS_THRESHOLD).astype(int)
    return agg


def build_tokenizer() -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def build_model(tokenizer: AutoTokenizer):
    """fold ごとに新しいモデル（ベース + 新規 LoRA）を作る。num_labels=2 の分類。"""
    quant_config = None
    if USE_4BIT:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID,
        num_labels=2,
        torch_dtype=torch.bfloat16,
        quantization_config=quant_config,
        device_map={"": 0} if USE_4BIT else None,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    if USE_4BIT:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True
        )

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_CLS",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        modules_to_save=["score"],
    )
    model = get_peft_model(model, lora_config)
    if not USE_4BIT:
        model.to(DEVICE)
        model.gradient_checkpointing_enable()
    return model


def make_tok_dataset(part_df: pd.DataFrame, tokenizer: AutoTokenizer) -> Dataset:
    ds = Dataset.from_pandas(part_df[["text", "label"]], preserve_index=False)

    def tokenize(batch):
        out = tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH)
        out["labels"] = [int(x) for x in batch["label"]]
        return out

    return ds.map(tokenize, batched=True, remove_columns=ds.column_names)


@dataclass
class ClassificationCollator:
    tokenizer: AutoTokenizer

    def __call__(self, features):
        labels = [f.pop("labels") for f in features]
        batch = self.tokenizer.pad(features, return_tensors="pt")
        batch["labels"] = torch.tensor(labels, dtype=torch.long)
        return batch


class WeightedTrainer(Trainer):
    """class_weight 付き CrossEntropy で不均衡に対応する Trainer。"""

    def __init__(self, class_weights: torch.Tensor | None = None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(
        self, model, inputs, return_outputs=False, **kwargs
    ):  # noqa: ARG002 num_items_in_batch 等は無視
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        # CrossEntropy は float を要求するため bf16 logits を float32 にキャストする
        logits = outputs.logits.float()
        weight = (
            self.class_weights.to(device=logits.device, dtype=logits.dtype)
            if self.class_weights is not None
            else None
        )
        loss_fct = torch.nn.CrossEntropyLoss(weight=weight)
        loss = loss_fct(logits.view(-1, 2), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def classification_metrics_from_logits(logits: np.ndarray, labels: np.ndarray) -> dict:
    preds = logits.argmax(axis=-1)
    yt = labels.astype(bool)
    yp = preds.astype(bool)
    tp = int((yt & yp).sum())
    tn = int((~yt & ~yp).sum())
    fp = int((~yt & yp).sum())
    fn = int((yt & ~yp).sum())
    n = len(labels)
    acc = (tp + tn) / n if n else 0.0
    pos_rec = tp / (tp + fn) if (tp + fn) else 0.0
    neg_rec = tn / (tn + fp) if (tn + fp) else 0.0
    pos_prec = tp / (tp + fp) if (tp + fp) else 0.0
    neg_prec = tn / (tn + fn) if (tn + fn) else 0.0
    pos_f1 = 2 * pos_prec * pos_rec / (pos_prec + pos_rec) if (pos_prec + pos_rec) else 0.0
    neg_f1 = 2 * neg_prec * neg_rec / (neg_prec + neg_rec) if (neg_prec + neg_rec) else 0.0
    return {
        "accuracy": acc,
        "positive_accuracy": pos_rec,
        "negative_accuracy": neg_rec,
        "macro_f1": (pos_f1 + neg_f1) / 2,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    m = classification_metrics_from_logits(np.asarray(logits), np.asarray(labels))
    return {
        "accuracy": m["accuracy"],
        "positive_accuracy": m["positive_accuracy"],
        "negative_accuracy": m["negative_accuracy"],
        "macro_f1": m["macro_f1"],
    }


def softmax_positive_prob(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    logits = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(logits)
    probs = exp / exp.sum(axis=-1, keepdims=True)
    return probs[:, 1]


def train_one_fold(
    fold: int,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    tokenizer: AutoTokenizer,
) -> tuple[dict, pd.DataFrame]:
    """1 fold を学習し、検証セットの指標と 1 件ごとの予測 DataFrame を返す。"""
    fold_dir = MODEL_DIR / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(tokenizer)
    train_tok = make_tok_dataset(train_df, tokenizer)
    val_tok = make_tok_dataset(val_df, tokenizer)

    # class_weight（balanced）を train ラベルから計算して不均衡に対応
    classes = np.array([0, 1])
    weights = compute_class_weight(
        class_weight="balanced", classes=classes, y=train_df["label"].to_numpy()
    )
    class_weights = torch.tensor(weights, dtype=torch.float32)
    print(
        f"=== fold {fold}: train={len(train_df)} (pos={int(train_df['label'].sum())}), "
        f"val={len(val_df)} (pos={int(val_df['label'].sum())}), "
        f"class_weight={weights.round(3).tolist()} ===",
        flush=True,
    )

    args = TrainingArguments(
        output_dir=str(fold_dir / "_checkpoints"),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.001,
        bf16=True,
        gradient_checkpointing=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=50,
        report_to="none",
        seed=SEED,
    )

    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        data_collator=ClassificationCollator(tokenizer),
        compute_metrics=compute_metrics,
    )
    trainer.train()

    metrics = trainer.evaluate(val_tok)
    result = {
        "fold": fold,
        "accuracy": float(metrics["eval_accuracy"]),
        "positive_accuracy": float(metrics["eval_positive_accuracy"]),
        "negative_accuracy": float(metrics["eval_negative_accuracy"]),
        "macro_f1": float(metrics["eval_macro_f1"]),
    }

    # 1 件ごとの予測（positive 確率と予測ラベル）を保存
    raw_logits = trainer.predict(val_tok).predictions
    prob_pos = softmax_positive_prob(np.asarray(raw_logits))
    pred_label = (prob_pos >= 0.5).astype(int)
    pred_df = val_df[GROUP_KEYS + ["acceptance_s", "n_raters", "label"]].copy().reset_index(drop=True)
    pred_df = pred_df.rename(columns={"label": "true_label"})
    pred_df["fold"] = fold
    pred_df["prob_positive"] = prob_pos
    pred_df["pred_label"] = pred_label
    pred_df["correct"] = pred_df["true_label"] == pred_df["pred_label"]

    trainer.save_model(str(fold_dir))
    with open(fold_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    pred_df.to_csv(fold_dir / "predictions.csv", index=False)
    print(
        f"fold {fold}: acc={result['accuracy']:.3f} "
        f"pos={result['positive_accuracy']:.3f} neg={result['negative_accuracy']:.3f} "
        f"macroF1={result['macro_f1']:.3f}",
        flush=True,
    )
    slack_notify(
        f"fold {fold} 完了: acc={result['accuracy']:.3f} "
        f"pos={result['positive_accuracy']:.3f} neg={result['negative_accuracy']:.3f} "
        f"macroF1={result['macro_f1']:.3f}"
    )

    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()
    return result, pred_df


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"読み込みデータ：{DATA_CSV}")
    print(f"ベースモデル：{MODEL_ID}")
    print(f"モデル保存先：{MODEL_DIR}")
    print(f"デバイス：{DEVICE}")
    print(f"受容度の元次元：{SOURCE_DIM}（集約キー: {GROUP_KEYS}, 最小回答者数: {MIN_RATERS}）")
    print(f"pos/neg 閾値：受容度 > {POS_THRESHOLD} を positive（中立4はnegative側）")
    print(f"SKIP_EXISTING_FOLDS：{SKIP_EXISTING_FOLDS}")
    if RESUME_FROM_RUN:
        print(f"RESUME_FROM_RUN：{RESUME_FROM_RUN}")

    resume_model_dir = None
    if RESUME_FROM_RUN:
        resume_model_dir = resolve_resume_model_dir(
            model_parent=OUTPUT_DIR / DIR_NAME,
            current_run=TASK,
            resume_from_run=RESUME_FROM_RUN,
        )
        if resume_model_dir is not None:
            print(f"再利用参照先：{resume_model_dir}")

    n_folds_planned = QUICK_FOLDS if QUICK_MODE else MAX_FOLDS
    slack_notify(
        f"pos/neg分類（{SOURCE_DIM}, n>={MIN_RATERS}, s>{POS_THRESHOLD}）"
        f"学習開始: {MODEL_ID}（{n_folds_planned}-fold）"
    )

    df = prepare_dataframe()
    n_pos = int(df["label"].sum())
    print(f"学習対象 {len(df)} 行（positive={n_pos} / negative={len(df) - n_pos}, pos率={n_pos/len(df):.1%}）")
    tokenizer = build_tokenizer()

    if QUICK_MODE:
        df = df.sample(n=min(QUICK_MAX_SAMPLES, len(df)), random_state=SEED).reset_index(drop=True)
        print(f"[QUICK_MODE] サンプル数を {len(df)} に制限")

    n_splits = QUICK_FOLDS if QUICK_MODE else N_FOLDS
    sgkf = StratifiedGroupKFold(n_splits=max(n_splits, 2), shuffle=True, random_state=SEED)

    max_folds = QUICK_FOLDS if QUICK_MODE else MAX_FOLDS
    fold_results: list[dict] = []
    pred_frames: list[pd.DataFrame] = []
    for fold, (train_idx, val_idx) in enumerate(
        sgkf.split(df, df["label"], groups=df["message_id"])
    ):
        if fold >= max_folds:
            break
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)
        reused = try_reuse_fold(
            fold,
            model_dir=MODEL_DIR,
            resume_model_dir=resume_model_dir,
            skip_existing=SKIP_EXISTING_FOLDS,
        )
        if reused is not None:
            result, pred_df = reused
        else:
            result, pred_df = train_one_fold(fold, train_df, val_df, tokenizer)
        fold_results.append(result)
        pred_frames.append(pred_df)

    results_df = pd.DataFrame(fold_results)
    all_preds = pd.concat(pred_frames, ignore_index=True)
    all_preds.to_csv(MODEL_DIR / "predictions.csv", index=False)

    # 全 fold の予測をまとめた全体指標（各サンプルは1回だけ検証側に出る）
    overall = classification_metrics_from_logits(
        np.stack([1 - all_preds["prob_positive"], all_preds["prob_positive"]], axis=1),
        all_preds["true_label"].to_numpy(),
    )

    summary = {
        "source_dim": SOURCE_DIM,
        "min_raters": MIN_RATERS,
        "pos_threshold": POS_THRESHOLD,
        "n_samples": len(all_preds),
        "accuracy_mean": float(results_df["accuracy"].mean()),
        "accuracy_std": float(results_df["accuracy"].std()),
        "positive_accuracy_mean": float(results_df["positive_accuracy"].mean()),
        "negative_accuracy_mean": float(results_df["negative_accuracy"].mean()),
        "macro_f1_mean": float(results_df["macro_f1"].mean()),
        "macro_f1_std": float(results_df["macro_f1"].std()),
        "overall_pooled": overall,
        "n_folds": len(results_df),
    }

    results_df.to_csv(MODEL_DIR / "cv_results.csv", index=False)
    with open(MODEL_DIR / "cv_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n=== {len(results_df)}-fold 結果（pos/neg分類: {SOURCE_DIM}, n>={MIN_RATERS}）===")
    print(results_df.to_string(index=False))
    print(
        f"\n全体正答率 = {summary['accuracy_mean']:.3f} ± {summary['accuracy_std']:.3f}"
        f"\npositive 正答率 = {summary['positive_accuracy_mean']:.3f}"
        f"\nnegative 正答率 = {summary['negative_accuracy_mean']:.3f}"
        f"\nmacro F1 = {summary['macro_f1_mean']:.3f} ± {summary['macro_f1_std']:.3f}"
    )
    print(f"保存完了：{MODEL_DIR}")
    slack_notify(
        f"全 {len(results_df)}-fold 完了 ✅\n"
        f"正答率 = {summary['accuracy_mean']:.3f} / macroF1 = {summary['macro_f1_mean']:.3f}\n"
        f"pos = {summary['positive_accuracy_mean']:.3f} / neg = {summary['negative_accuracy_mean']:.3f}\n"
        f"保存先: {MODEL_DIR}"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001 失敗も Slack に通知して再送出
        slack_notify(f"学習が異常終了 ❌\n{type(e).__name__}: {e}")
        print(traceback.format_exc(), flush=True)
        raise
