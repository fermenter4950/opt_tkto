from __future__ import annotations

import re
from dataclasses import replace

from src.application.interfaces.effect_predictor import EffectPredictor
from src.domain import Label, UserCharacteristics
from src.infrastructure.llm.client import LLMSession


class EffectPredictorLLM(EffectPredictor):
    """ローカル LLM で運動促進メッセージを pos/neg 採点する。

    adapter_path を指定すると LoRA 学習済み採点モデルを使う。
    """

    SYSTEM = (
        "あなたは運動促進メッセージを評価するアシスタントです。"
        "メッセージと受信者の特性が与えられたら、その受信者が運動への意識を高め"
        "行動に移したくなるかを positive または negative で分類してください。"
    )
    PROMPT_TEMPLATE = """\
### 運動促進メッセージ ###
'''
{message}
'''

### 受信者の特性 ###
性別: {gender}
年齢: {age_group}
行動変容ステージ: {stage}

### タスク ###
上記のメッセージを、指定された受信者が読んだ場合、運動への意識が高まり行動に移したくなるかを判断し、
positive または negative のいずれか1語のみで回答してください。

### 評価基準 ###
- positive: 受信者の特性に適しており、運動促進の動機付けになる可能性が高い
- negative: 受信者の特性に適していない、または動機付けにならない可能性が高い

回答（positive または negative のみ）:"""

    def __init__(
        self,
        session: LLMSession,
        adapter_path: str | None = None,
        max_new_tokens: int = 16,
        temperature: float = 0.0,
    ):
        self.session = session
        if adapter_path:
            from peft import PeftModel

            self.session = replace(
                session,
                model=PeftModel.from_pretrained(session.model, adapter_path),
            )
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def predict(
        self,
        message: str,
        characteristics: UserCharacteristics,
        threshold: float = 0.5,
    ) -> Label:
        del threshold

        prompt = self.PROMPT_TEMPLATE.format(
            message=message,
            gender=characteristics.gender.value,
            age_group=characteristics.age_group.value,
            stage=characteristics.stage.value,
        )
        chat = [
            {"role": "system", "content": self.SYSTEM},
            {"role": "user", "content": prompt},
        ]
        generated = self.session.generate_from_chat(
            chat,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            do_sample=self.temperature > 0,
        )
        return self._parse_label(generated)

    @staticmethod
    def _parse_label(text: str) -> Label:
        normalized = text.strip().lower()
        if re.search(r"\bnegative\b", normalized):
            return Label.NEGATIVE
        if re.search(r"\bpositive\b", normalized):
            return Label.POSITIVE

        neg_score = normalized.count("negative") + normalized.count("否")
        pos_score = normalized.count("positive") + normalized.count("是")
        if neg_score > pos_score:
            return Label.NEGATIVE
        return Label.POSITIVE
