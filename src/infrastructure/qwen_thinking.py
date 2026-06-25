"""Qwen3 / Shisa V2.1 系の構造化出力（【思考】/【回答】）プロンプトとパース。"""

from __future__ import annotations

import re
from typing import Literal

from transformers import PreTrainedTokenizer

from src.infrastructure.harmony import (
    DEVELOPER_PROMPT,
    _ENGLISH_FRAGMENT,
    _META_PATTERN,
    count_text_tokens,
)

QwenKtoFormat = Literal["split", "joint"]

THINKING_HEADER = "【思考】"
ANSWER_HEADER = "【回答】"
THINKING_PREFILL = "方針として、"
# Qwen3 の空 thinking ブロック（enable_thinking=False 時に付く）
THINK_OPEN = "<" + "think>"
THINK_CLOSE = "</" + "think>"

# 最終回答の上限（字）。超えたら KTO 除外
MAX_FINAL_CHARS = 250

QWEN_DEVELOPER_PROMPT = (
    DEVELOPER_PROMPT
    + """

【出力形式】（厳守。見出し・ラベル以外の説明文は書かない）
"""
    + THINKING_HEADER
    + """
（日本語3文以内。受信者に合わせた方針のみ）
"""
    + ANSWER_HEADER
    + """
（最適化した運動促進メッセージ本文のみ。50〜150字。分析・前置きは書かない）
"""
)


def _im_end_token(tokenizer: PreTrainedTokenizer) -> str:
    return tokenizer.eos_token or "<|" + "im_end|>"


def _strip_empty_thinking_block(prompt: str) -> str:
    """chat template が付与する空の  ブロックを除去。"""
    empty = f"{THINK_OPEN}\n\n{THINK_CLOSE}\n\n"
    if prompt.endswith(empty):
        return prompt[: -len(empty)]
    return prompt


def build_prompt_text(tokenizer: PreTrainedTokenizer, instruction: str) -> str:
    """system + user + assistant（【思考】prefill）までのプロンプト。"""
    messages = [
        {"role": "system", "content": QWEN_DEVELOPER_PROMPT},
        {"role": "user", "content": instruction},
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=False,
    )
    prompt_text = _strip_empty_thinking_block(prompt_text)
    return prompt_text + f"{THINKING_HEADER}\n{THINKING_PREFILL}"


def _strip_thinking_prefill(text: str) -> str:
    text = text.strip()
    if text.startswith(THINKING_PREFILL):
        return text[len(THINKING_PREFILL) :].strip()
    return text


def parse_qwen_thinking_output(
    raw: str,
    *,
    im_end: str | None = None,
) -> tuple[str, str]:
    """assistant 応答から thinking / final を抽出。"""
    text = raw.strip()
    if not text:
        return "", ""

    body = text.split(THINKING_HEADER, 1)[-1] if THINKING_HEADER in text else text

    if ANSWER_HEADER not in body:
        thinking = _strip_thinking_prefill(body)
        return thinking, ""

    thinking, final = body.split(ANSWER_HEADER, 1)
    thinking = _strip_thinking_prefill(thinking)
    final = final.strip()
    if im_end and final.endswith(im_end):
        final = final[: -len(im_end)].strip()
    return thinking.strip(), final


def has_answer_marker(text: str) -> bool:
    return ANSWER_HEADER in text


def build_qwen_completion(thinking: str, final: str, *, im_end: str) -> str:
    suffix = thinking.strip()
    if suffix and not suffix.startswith(THINKING_PREFILL):
        suffix = f"{THINKING_PREFILL}{suffix}"
    return f"{suffix}\n{ANSWER_HEADER}\n{final.strip()}{im_end}"


def build_kto_pair(
    base_prompt: str,
    thinking: str,
    final: str,
    *,
    im_end: str,
) -> tuple[str, str]:
    suffix = thinking.strip()
    if suffix.startswith(THINKING_PREFILL):
        suffix = suffix[len(THINKING_PREFILL) :].strip()
    kto_prompt = base_prompt + (suffix if suffix else "") + f"\n{ANSWER_HEADER}\n"
    kto_completion = f"{final.strip()}{im_end}"
    return kto_prompt, kto_completion


def preprocess_qwen_for_kto(
    base_prompt: str,
    thinking: str,
    final: str,
    *,
    im_end: str,
    mode: QwenKtoFormat = "split",
) -> dict[str, str]:
    full = build_qwen_completion(thinking, final, im_end=im_end)
    if mode == "joint":
        return {
            "prompt": base_prompt,
            "completion": full,
            "thinking": thinking,
            "harmony_completion": full,
        }
    kto_prompt, kto_completion = build_kto_pair(
        base_prompt, thinking, final, im_end=im_end
    )
    return {
        "prompt": kto_prompt,
        "completion": kto_completion,
        "thinking": thinking,
        "harmony_completion": full,
    }


def kto_validation_issues(thought: str, response: str) -> list[str]:
    issues: list[str] = []
    if not thought.strip():
        issues.append("empty_thinking")
    if not response.strip():
        issues.append("empty_final")
    if len(response.strip()) > MAX_FINAL_CHARS:
        issues.append(f"final_too_long({len(response.strip())})")
    return issues


def kto_quality_notes(thought: str, response: str) -> list[str]:
    notes: list[str] = []
    combined = thought + response
    if _META_PATTERN.search(combined):
        notes.append("meta_pattern")
    if _ENGLISH_FRAGMENT.search(thought):
        notes.append("english_in_thinking")
    # 分析調の final（受信者説明から始まる等）
    if re.search(r"^(40|50|60)代|受信者|ベースメッセージ|実行期", response.strip()):
        notes.append("analysis_in_final")
    return notes


def generation_ended_with_stop(generated: str, *, im_end: str) -> bool:
    return generated.strip().endswith(im_end)


def is_likely_truncated(
    generated: str,
    *,
    max_new_tokens: int,
    token_count: int,
    im_end: str,
) -> bool:
    if generation_ended_with_stop(generated, im_end=im_end):
        return False
    return token_count >= max(1, max_new_tokens - 2)


def build_assistant_raw(prompt_text: str, generated: str) -> str:
    marker = "<|" + "im_start|>assistant"
    start = prompt_text.rfind(marker)
    prefix = prompt_text[start:] if start >= 0 else ""
    if THINKING_HEADER not in prefix:
        if prefix and not prefix.endswith("\n"):
            prefix += "\n"
        prefix += f"{THINKING_HEADER}\n{THINKING_PREFILL}"
    return prefix + generated
