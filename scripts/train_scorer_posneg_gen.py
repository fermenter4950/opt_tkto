"""受容度（intention）の positive/negative 生成 SFT スコアラー fine-tuning（パターンC）。

KTO 用に「メッセージ＋受信者属性 → 受信者が運動に前向きになるか（positive/negative）」
を **生成方式（SFT）** で学習する。分類ヘッド版（train_scorer_posneg.py）と違い、
モデルに "positive" / "negative" の語そのものを生成させる。

パターンC のポイント:
- 学習は生成 SFT（assistant_only_loss で "positive"/"negative" 部分だけに loss）。
  プロンプトと出力が一致するため、EffectPredictorLLM（生成方式）にそのまま乗る。
- 推論では「positive」「negative」両方の系列対数尤度を計算し、softmax で
  prob_positive を得る（白黒だけでなく連続スコアも取れる）。

データ設計は分類版と同一:
- 受容度は intention 単独。メッセージ×受信者属性（gender/age_group/stage）ごとに
  回答者の intention を平均（＝paper 式1）して受容度 s を作る。
- 回答者数が MIN_RATERS 未満のグループは平均が不安定なので除外。
- s > POS_THRESHOLD を positive、それ以外を negative（中立クラスなし）。
- メッセージのリーク防止に StratifiedGroupKFold（groups=message_id）で分割。

各 fold で accuracy / positive 再現率 / negative 再現率 / macro-F1 を算出し、
1 件ごとの予測（prob_positive・予測ラベル）も保存する。

    python scripts/train_scorer_posneg_gen.py

環境変数で上書き可能:
    SCORER_NAME    採点器名（保存先の階層・読み込むモデル。既定: Qwen3-Swallow-8B-RL-v0.2）
    BASE_MODEL_PATH ベースモデルのローカルパス or HF ID（SCORER_NAME より優先）
    GPU_ID         使用 GPU（0 か 1。既定: 1）
    USE_4BIT       1 で 4bit QLoRA（既定）、0 で bf16 そのまま
    MIN_RATERS     集約に残す最小回答者数（既定: 3）
    POS_THRESHOLD  この値より大きい受容度を positive とみなす（既定: 4.0）
    MAX_FOLDS      学習する fold 数の上限（既定: N_FOLDS=5。1 なら 1 fold だけ）
    RUN_NAME       保存先サブディレクトリ名（既定: posneg_intention_gen）。
                   既存の重みを上書きしたくないときに変える。
    PROMPT_VARIANT プロンプト種別: baseline（既定）/ fp3 / plain
    RESUME_FROM_RUN  完了済み fold の参照先 RUN（例: posneg_gen_baseline_shisa8b_1fold）。
                     ``auto`` で RUN_NAME の ``_5fold`` → ``_1fold`` を自動推定。
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
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

# 使用する GPU を明示指定（0 か 1）。環境変数 GPU_ID で上書き可。
# torch import 前に CUDA_VISIBLE_DEVICES を設定する必要がある。
GPU_ID = os.getenv("GPU_ID", "1")
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID
print(f"使用 GPU: cuda:{GPU_ID}（CUDA_VISIBLE_DEVICES={GPU_ID}）")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from sklearn.model_selection import StratifiedGroupKFold
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer

from scorer_fold_resume import resolve_resume_model_dir, try_reuse_fold

from src.domain.scorer_prompt import SCORER_SYSTEM_PROMPT, format_scorer_prompt

# ---------------------------------------------------------------------------
# 設定
# ---------------------------------------------------------------------------
SCORER_NAME = os.getenv("SCORER_NAME", "tokyotech-llm/Qwen3-Swallow-8B-RL-v0.2")
# 保存先サブディレクトリ名。RUN_NAME で上書きして既存重みの上書きを防げる。
TASK = os.getenv("RUN_NAME", "posneg_intention_gen")  # intention 受容度の pos/neg 生成 SFT

USE_4BIT = os.getenv("USE_4BIT", "1") == "1"

N_FOLDS = 5
SEED = 42
MAX_FOLDS = int(os.getenv("MAX_FOLDS", str(N_FOLDS)))

QUICK_MODE = os.getenv("QUICK_MODE", "0") == "1"
QUICK_FOLDS = 1
QUICK_MAX_SAMPLES = 400

# 受容度の元次元（前向きになれるか＝intention 単独）
SOURCE_DIM = "intention"
GROUP_KEYS = ["message_id", "gender", "age_group", "stage"]
MIN_RATERS = int(os.getenv("MIN_RATERS", "3"))
POS_THRESHOLD = float(os.getenv("POS_THRESHOLD", "4.0"))

MAX_LENGTH = 768

# 学習ハイパラ。実効バッチ = BATCH_SIZE * GRAD_ACCUM = 16
NUM_EPOCHS = 3
BATCH_SIZE = 4
GRAD_ACCUM = 4
LR = 2e-4
LORA_R = 32
LORA_ALPHA = 16

POS_WORD = "positive"
NEG_WORD = "negative"
PROMPT_VARIANT = os.getenv("PROMPT_VARIANT", "baseline")  # baseline | fp3 | plain
RESUME_FROM_RUN = os.getenv("RESUME_FROM_RUN", "")
SKIP_EXISTING_FOLDS = os.getenv("SKIP_EXISTING_FOLDS", "1") == "1"

REPO_ROOT = Path(__file__).resolve().parent.parent

DATA_CSV = REPO_ROOT / "data" / "scorer_finetune_raw" / "all_splits.csv"
MODEL_ID = os.getenv("BASE_MODEL_PATH", SCORER_NAME)
OUTPUT_DIR = REPO_ROOT / "models" / "scorer"
DIR_NAME = Path(SCORER_NAME).name
MODEL_DIR = OUTPUT_DIR / DIR_NAME / TASK

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


SYSTEM_PROMPT = SCORER_SYSTEM_PROMPT

# 旧版（PROMPT_VARIANT=fp3 / plain 用）。既定は src.domain.scorer_prompt の baseline。
PROMPT_TEMPLATE_FP3 = """\
### 健康促進メッセージ ###
'''
{message}
'''

### 受信者の特性 ###
性別: {gender}
年代: {age_group}
行動変容ステージ: {stage}

### 評価基準 ###
以下の3条件を満たすかどうかを判断してください：
1. ステージ適合性: メッセージが受信者の行動変容ステージに適しているか
2. 行動の具体性: メッセージが具体的な行動提案を含んでいるか
3. 実行可能性: メッセージが受信者にとって実行可能な内容であるか

### 回答手順 ###
1. メッセージと受信者の特性を慎重に分析してください.
2. 評価基準に基づいて判断を行ってください.
3. 評価基準の3つの条件のうち、3つすべてを満たす場合のみ「positive」、そうでなければ必ず「negative」と判断してください.
4. 最終的な回答(positive または negative)のみを出力してください.

### 注意事項 ###
- 回答は「positive」または「negative」のみとしてください.
- 説明や追加のコメントは不要です.
- 誤ってpositiveと判断することは重大な評価エラーです。基準を明確に満たす場合のみpositiveとしてください。迷う場合はnegativeとしてください.
- 明示されていない内容を推測して補完してはいけません.
- 各条件は、本文から直接根拠が得られる場合にのみ「満たす」と判定してください.
- 迷う/根拠が弱い場合は、その条件は「満たさない」と判定してください.

回答（positive または negative のみ）:"""

# FP警告・保守的判定指示を除いた版。3条件評価基準は PROMPT_TEMPLATE_FP3 と同一。
PROMPT_TEMPLATE_PLAIN = """\
### 健康促進メッセージ ###
'''
{message}
'''

### 受信者の特性 ###
性別: {gender}
年代: {age_group}
行動変容ステージ: {stage}

### 評価基準 ###
以下の3条件を満たすかどうかを判断してください：
1. ステージ適合性: メッセージが受信者の行動変容ステージに適しているか
2. 行動の具体性: メッセージが具体的な行動提案を含んでいるか
3. 実行可能性: メッセージが受信者にとって実行可能な内容であるか

### 回答手順 ###
1. メッセージと受信者の特性を分析してください.
2. 評価基準に基づいて判断を行ってください.
3. 評価基準の3つの条件のうち、3つすべてを満たす場合は「positive」、そうでなければ「negative」と判断してください.
4. 最終的な回答(positive または negative)のみを出力してください.

### 注意事項 ###
- 回答は「positive」または「negative」のみとしてください.
- 説明や追加のコメントは不要です.

回答（positive または negative のみ）:"""


def format_user_prompt(row) -> str:
    if PROMPT_VARIANT == "baseline":
        return format_scorer_prompt(
            message=row["メッセージ"],
            sex=row["gender"],
            age=row["age_group"],
            stage=row["stage"],
        )
    template = PROMPT_TEMPLATE_PLAIN if PROMPT_VARIANT == "plain" else PROMPT_TEMPLATE_FP3
    if PROMPT_VARIANT not in {"fp3", "plain"}:
        raise ValueError(
            f"未知の PROMPT_VARIANT: {PROMPT_VARIANT!r}（baseline / fp3 / plain）"
        )
    return template.format(
        message=str(row["メッセージ"]).strip(),
        gender=row["gender"],
        age_group=row["age_group"],
        stage=row["stage"],
    )


def build_messages(row, *, with_answer: bool) -> list[dict]:
    user = format_user_prompt(row)
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
    if with_answer:
        word = POS_WORD if int(row["label"]) == 1 else NEG_WORD
        msgs.append({"role": "assistant", "content": word})
    return msgs


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
    """CSV を読み込み、メッセージ×属性で集約して受容度→pos/neg ラベルを作る。"""
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
    agg["label"] = (agg["acceptance_s"] > POS_THRESHOLD).astype(int)
    return agg


# assistant_only_loss=True 用の訓練互換 ChatML テンプレート。
# アシスタント応答を {% generation %} で囲み、その部分だけに loss をかけられるようにする。
# 元モデルのテンプレートに generation マーカーが無い場合（例: Shisa/Qwen3 thinking）に使う。
TRAINING_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\n' }}"
    "{% if message['role'] == 'assistant' %}"
    "{% generation %}{{ message['content'] }}{% endgeneration %}"
    "{{ '<|im_end|>' + '\n' }}"
    "{% else %}"
    "{{ message['content'] + '<|im_end|>' + '\n' }}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)


def build_tokenizer() -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # assistant_only_loss は chat_template に {% generation %} マーカーを要求する。
    # add_generation_prompt という語も "generation" を含むため、マーカーそのものを検出する。
    template = tokenizer.chat_template or ""
    if "{% generation %}" not in template and "{%generation%}" not in template:
        print("[tokenizer] chat_template に generation マーカーが無いため訓練互換 ChatML に差し替え", flush=True)
        tokenizer.chat_template = TRAINING_CHAT_TEMPLATE
    return tokenizer


def build_model(tokenizer: AutoTokenizer):
    """fold ごとに新しいモデル（ベース + 新規 LoRA）を作る。生成 SFT 用 CausalLM。"""
    quant_config = None
    if USE_4BIT:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
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
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)
    if not USE_4BIT:
        model.to(DEVICE)
        model.gradient_checkpointing_enable()
    return model


@torch.no_grad()
def _seq_logprob(model, prompt_ids: torch.Tensor, word_ids: list[int]) -> float:
    """prompt に続けて word_ids を生成する系列対数尤度の合計を返す。"""
    device = model.device
    word = torch.tensor([word_ids], device=device)
    full = torch.cat([prompt_ids, word], dim=1)
    logits = model(full).logits  # [1, T, V]
    # word の各トークンを予測するのは直前位置の logits
    start = prompt_ids.shape[1] - 1
    total = 0.0
    for i, tid in enumerate(word_ids):
        step_logits = logits[0, start + i]
        logp = F.log_softmax(step_logits.float(), dim=-1)[tid]
        total += float(logp)
    return total


@torch.no_grad()
def predict_prob_positive(model, tokenizer, df: pd.DataFrame) -> dict[str, np.ndarray]:
    """各行について prob_positive と生のラベル質量を算出する。

    prob_positive:
        "positive"/"negative" の系列対数尤度を2語だけ softmax した値（合計1）。
    log_label_mass:
        log(exp(lp_pos) + exp(lp_neg))。生の合計質量の log 版（アンダーフロー回避）。
    margin:
        |logprob_pos - logprob_neg|。2語の尤度差。大きいほど判定がはっきり。
    logprob_pos / logprob_neg:
        系列対数尤度（正規化前）。
    """
    model.eval()
    pos_ids = tokenizer.encode(POS_WORD, add_special_tokens=False)
    neg_ids = tokenizer.encode(NEG_WORD, add_special_tokens=False)

    prob_pos_list: list[float] = []
    log_mass_list: list[float] = []
    margin_list: list[float] = []
    lp_pos_list: list[float] = []
    lp_neg_list: list[float] = []

    for _, row in df.iterrows():
        messages = build_messages(row, with_answer=False)
        encoded = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        )
        if isinstance(encoded, torch.Tensor):
            prompt_ids = encoded.to(model.device)
        else:
            prompt_ids = encoded["input_ids"].to(model.device)
        lp_pos = _seq_logprob(model, prompt_ids, pos_ids)
        lp_neg = _seq_logprob(model, prompt_ids, neg_ids)
        log_mass = float(logsumexp([lp_pos, lp_neg]))
        m = max(lp_pos, lp_neg)
        p_pos = np.exp(lp_pos - m) / (np.exp(lp_pos - m) + np.exp(lp_neg - m))
        prob_pos_list.append(float(p_pos))
        log_mass_list.append(log_mass)
        margin_list.append(abs(lp_pos - lp_neg))
        lp_pos_list.append(lp_pos)
        lp_neg_list.append(lp_neg)

    return {
        "prob_positive": np.asarray(prob_pos_list, dtype=np.float64),
        "log_label_mass": np.asarray(log_mass_list, dtype=np.float64),
        "margin": np.asarray(margin_list, dtype=np.float64),
        "logprob_pos": np.asarray(lp_pos_list, dtype=np.float64),
        "logprob_neg": np.asarray(lp_neg_list, dtype=np.float64),
    }


def classification_metrics(prob_pos: np.ndarray, labels: np.ndarray) -> dict:
    preds = (prob_pos >= 0.5).astype(int)
    yt = labels.astype(bool)
    yp = preds.astype(bool)
    tp = int((yt & yp).sum())
    tn = int((~yt & ~yp).sum())
    fp = int((~yt & yp).sum())
    fn = int((yt & ~yp).sum())
    pos_rec = tp / (tp + fn) if (tp + fn) else 0.0
    neg_rec = tn / (tn + fp) if (tn + fp) else 0.0
    pos_prec = tp / (tp + fp) if (tp + fp) else 0.0
    neg_prec = tn / (tn + fn) if (tn + fn) else 0.0
    pos_f1 = (
        2 * pos_prec * pos_rec / (pos_prec + pos_rec) if (pos_prec + pos_rec) else 0.0
    )
    neg_f1 = (
        2 * neg_prec * neg_rec / (neg_prec + neg_rec) if (neg_prec + neg_rec) else 0.0
    )
    n = len(labels)
    return {
        "accuracy": (tp + tn) / n if n else 0.0,
        "positive_accuracy": pos_rec,
        "negative_accuracy": neg_rec,
        "macro_f1": (pos_f1 + neg_f1) / 2,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def make_sft_dataset(part_df: pd.DataFrame) -> Dataset:
    rows = [{"messages": build_messages(r, with_answer=True)} for _, r in part_df.iterrows()]
    return Dataset.from_list(rows)


def train_one_fold(
    fold: int,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    tokenizer: AutoTokenizer,
) -> tuple[dict, pd.DataFrame]:
    """1 fold を生成 SFT し、検証セットの指標と 1 件ごとの予測 DataFrame を返す。"""
    fold_dir = MODEL_DIR / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(tokenizer)
    train_ds = make_sft_dataset(train_df)
    val_ds = make_sft_dataset(val_df)

    print(
        f"=== fold {fold}: train={len(train_df)} (pos={int(train_df['label'].sum())}), "
        f"val={len(val_df)} (pos={int(val_df['label'].sum())}) ===",
        flush=True,
    )

    args = SFTConfig(
        output_dir=str(fold_dir / "_checkpoints"),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.001,
        max_grad_norm=0.3,
        bf16=True,
        gradient_checkpointing=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=50,
        max_length=MAX_LENGTH,
        dataset_text_field="messages",
        packing=False,
        assistant_only_loss=True,
        report_to="none",
        seed=SEED,
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
    )
    trainer.train()

    # 検証セットの prob_positive を系列尤度から算出
    scores = predict_prob_positive(trainer.model, tokenizer, val_df)
    prob_pos = scores["prob_positive"]
    labels = val_df["label"].to_numpy()
    m = classification_metrics(prob_pos, labels)
    result = {
        "fold": fold,
        "accuracy": m["accuracy"],
        "positive_accuracy": m["positive_accuracy"],
        "negative_accuracy": m["negative_accuracy"],
        "macro_f1": m["macro_f1"],
    }

    pred_df = val_df[GROUP_KEYS + ["acceptance_s", "n_raters", "label"]].copy().reset_index(drop=True)
    pred_df = pred_df.rename(columns={"label": "true_label"})
    pred_df["fold"] = fold
    pred_df["prob_positive"] = prob_pos
    pred_df["log_label_mass"] = scores["log_label_mass"]
    pred_df["margin"] = scores["margin"]
    pred_df["logprob_pos"] = scores["logprob_pos"]
    pred_df["logprob_neg"] = scores["logprob_neg"]
    pred_df["pred_label"] = (prob_pos >= 0.5).astype(int)
    pred_df["correct"] = pred_df["true_label"] == pred_df["pred_label"]

    trainer.save_model(str(fold_dir))
    tokenizer.save_pretrained(str(fold_dir))
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
    print("方式：生成 SFT（パターンC, assistant_only_loss）")
    print(f"プロンプト：{PROMPT_VARIANT}")
    if RESUME_FROM_RUN:
        print(f"RESUME_FROM_RUN：{RESUME_FROM_RUN}")
    print(f"SKIP_EXISTING_FOLDS：{SKIP_EXISTING_FOLDS}")

    resume_model_dir = resolve_resume_model_dir(
        model_parent=OUTPUT_DIR / DIR_NAME,
        current_run=TASK,
        resume_from_run=RESUME_FROM_RUN,
    )
    if resume_model_dir is not None:
        print(f"再利用参照先：{resume_model_dir}")

    n_folds_planned = QUICK_FOLDS if QUICK_MODE else MAX_FOLDS
    slack_notify(
        f"pos/neg 生成SFT（{SOURCE_DIM}, n>={MIN_RATERS}, s>{POS_THRESHOLD}）"
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

    overall = classification_metrics(
        all_preds["prob_positive"].to_numpy(),
        all_preds["true_label"].to_numpy(),
    )

    summary = {
        "source_dim": SOURCE_DIM,
        "method": "generative_sft",
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

    print(f"\n=== {len(results_df)}-fold 結果（pos/neg 生成SFT: {SOURCE_DIM}, n>={MIN_RATERS}）===")
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
