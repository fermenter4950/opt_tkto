"""LLM-jp-4 thinking（Harmony 形式）のプロンプト組み立てとパース。"""

from __future__ import annotations

import re
from typing import Literal

from transformers import PreTrainedTokenizer

HarmonyKtoFormat = Literal["split", "joint"]

DEVELOPER_PROMPT = """\
あなたは「運動促進メッセージ」を最適化するアシスタントです。

【タスク】
ベースメッセージと受信者情報をもとに、対象の受信者が運動への意識を高め、
実際に行動に移したくなるよう、メッセージを再構築する。

【思考】
- 日本語で3文以内。受信者に合わせて何を強調・言い換えするかだけ書く。
- 構成案・章立て・「1. 2. 3.」の設計は書かない。
- ルールやチャネル名を思考内に書かない。

【最終回答】
- 最適化した運動促進メッセージ本文のみ（説明・前置き・見出しなし）。
- 1件の投稿として完結。おおむね50〜150字を目安とする。
- 禁止: 見出し（#）、水平線、表、番号付き章立て、長い箇条書き。
- 禁止: 「以下が最適化したメッセージです」などのラッピング文。"""

ANALYSIS_PREFILL = "<|channel|>analysis<|message|>方針として、"
FINAL_CHANNEL_PREFIX = "<|end|><|start|>assistant<|channel|>final<|message|>"

# Harmony 生成の停止文字列（analysis 区切りの <|end|> では止めない）
HARMONY_STOP_STRINGS: tuple[str, ...] = ("<|return|>",)

_META_PATTERN = re.compile(
    r"文字数|カウント|let's count|```|^\*\*|^---|^###",
    re.IGNORECASE | re.MULTILINE,
)
_ENGLISH_FRAGMENT = re.compile(r"[a-zA-Z]{5,}")


def count_text_tokens(tokenizer: PreTrainedTokenizer, text: str) -> int:
    """テキストのトークン数（特殊トークンなし）。"""
    if not text:
        return 0
    return len(tokenizer.encode(text, add_special_tokens=False))


def build_prompt_text(
    tokenizer: PreTrainedTokenizer,
    instruction: str,
    reasoning_effort: str = "low",
) -> str:
    """developer + user（Instruction）+ assistant analysis prefill までのプロンプト文字列。"""
    messages = [
        {"role": "developer", "content": DEVELOPER_PROMPT},
        {"role": "user", "content": instruction},
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
        reasoning_effort=reasoning_effort,
    )
    return prompt_text + ANALYSIS_PREFILL


def parse_harmony_output(raw: str) -> tuple[str, str]:
    """assistant 応答（タグ付き）から analysis / final を抽出。"""
    text = raw
    if not text.lstrip().startswith("<|channel|>") and "<|channel|>analysis" not in text:
        text = "<|channel|>analysis<|message|>" + text

    analysis = ""
    final = ""

    analysis_match = re.search(
        r"<\|channel\|>analysis<\|message\|>(.*?)(?:<\|end\|>|<\|channel\|>final)",
        text,
        flags=re.DOTALL,
    )
    if analysis_match:
        analysis = analysis_match.group(1).strip()

    final_match = re.search(
        r"<\|channel\|>final<\|message\|>(.*?)(?:<\|return\|>|<\|end\|>|$)",
        text,
        flags=re.DOTALL,
    )
    if final_match:
        final = final_match.group(1).strip()

    if not final and "assistantfinal" in text.lower():
        parts = re.split(r"assistant\s*final", text, maxsplit=1, flags=re.IGNORECASE)
        if len(parts) == 2:
            if not analysis:
                analysis = parts[0].strip()
            final = parts[1].strip()

    return analysis, final


_ANALYSIS_PREFILL_PREFIXES = ("方針として")


def analysis_suffix(analysis: str) -> str:
    """analysis 本文（prefill 由来の接頭辞が残っていれば除去）。"""
    text = analysis.strip()
    for prefix in _ANALYSIS_PREFILL_PREFIXES:
        if text.startswith(prefix):
            return text[len(prefix) :].strip()
    return text


def build_harmony_completion(analysis: str, final: str) -> str:
    """analysis prefill 直後からの Harmony 継続（KTO・保存・分析用）。

    analysis と final は同一 rollout のペアとして joint に学習する。
    """
    suffix = analysis_suffix(analysis)
    return f"{suffix}{FINAL_CHANNEL_PREFIX}{final.strip()}<|return|>"


def build_kto_pair(base_prompt: str, analysis: str, final: str) -> tuple[str, str]:
    """Pattern 2: analysis を prompt に、final のみ completion（thinking loss マスク相当）。"""
    suffix = analysis_suffix(analysis)
    kto_prompt = base_prompt + suffix + FINAL_CHANNEL_PREFIX
    kto_completion = f"{final.strip()}<|return|>"
    return kto_prompt, kto_completion


def preprocess_harmony_for_kto(
    base_prompt: str,
    thinking: str,
    final: str,
    *,
    mode: HarmonyKtoFormat = "split",
) -> dict[str, str]:
    """Harmony rollout を TRL KTO 用 prompt/completion へ変換する。

    split（推奨）:
      - thinking (= analysis) を prompt に組み込み → loss マスク
      - completion = final 本文 + <|return|>
      - 採点器ラベル（final の良し悪し）と loss 対象が一致

    joint（Pattern 1）:
      - completion = analysis + final の Harmony 全文
      - rollout 全体に KTO 信号
    """
    harmony = build_harmony_completion(thinking, final)
    if mode == "joint":
        return {
            "prompt": base_prompt,
            "completion": harmony,
            "thinking": thinking,
            "harmony_completion": harmony,
        }
    kto_prompt, kto_completion = build_kto_pair(base_prompt, thinking, final)
    return {
        "prompt": kto_prompt,
        "completion": kto_completion,
        "thinking": thinking,
        "harmony_completion": harmony,
    }


def kto_validation_issues(thought: str, response: str) -> list[str]:
    """KTO 学習から除外する致命的な形式問題。空のみ（実行停止の原因にしない）。"""
    issues: list[str] = []
    if not thought.strip():
        issues.append("empty_thinking")
    if not response.strip():
        issues.append("empty_final")
    return issues


def kto_quality_notes(thought: str, response: str) -> list[str]:
    """品質上の注意点（記録用。KTO 除外や実行停止には使わない）。"""
    notes: list[str] = []
    combined = thought + response
    if _META_PATTERN.search(combined):
        notes.append("meta_pattern")
    if _ENGLISH_FRAGMENT.search(thought):
        notes.append("english_in_thinking")
    return notes


def is_valid_for_kto(thought: str, response: str) -> bool:
    """KTO 学習データとして形式が妥当か。"""
    return not kto_validation_issues(thought, response)


def generation_ended_with_stop(generated: str) -> bool:
    """生成が Harmony 終了トークンで止まったか（打ち切りでないか）。"""
    text = generated.strip()
    return any(text.endswith(stop) for stop in HARMONY_STOP_STRINGS)


def is_likely_truncated(generated: str, *, max_new_tokens: int, token_count: int) -> bool:
    """max_new_tokens 上限で切れた疑いがあるか。"""
    if generation_ended_with_stop(generated):
        return False
    # 停止トークンなしで上限付近なら打ち切りとみなす
    return token_count >= max(1, max_new_tokens - 2)


def build_assistant_raw(prompt_text: str, generated: str) -> str:
    """prefill を含む assistant 応答全文を組み立てる。"""
    assistant_start = prompt_text.rfind("<|start|>assistant")
    assistant_prefix = (
        prompt_text[assistant_start:] if assistant_start >= 0 else ANALYSIS_PREFILL
    )
    return assistant_prefix + generated
