from openai import OpenAI

from src.application.interfaces.effect_predictor import EffectPredictor
from src.domain import Label, UserCharacteristics


class EffectPredictorImpl(EffectPredictor):
    SYSTEM = "あなたは健康促進メッセージを評価するアシスタントです。健康メッセージと受信者の特性（性別、年代、行動変容ステージ）が与えられた場合、その特性を持つ受信者がそのメッセージを読み健康改善に前向きになるかを 'positive', 'negative' のいずれかで分類してください。"
    PROMPT_TEMPLATE = """\
### 健康促進メッセージ ###
'''
{message}
'''

### 受信者の特性 ###
性別: {sex}
年代: {age}
行動変容ステージ: {stage}

### タスク ###
上記の健康促進メッセージを、指定された受信者が読んだ場合、健康改善に前向きになるかどうかを判断し、以下のいずれかで回答してください：
- positive
- negative

### 注意事項 ###
- 回答は「positive」または「negative」のみとしてください。
- 説明や追加のコメントは不要です。

### 評価基準 ###
- positive: メッセージが受信者の特性に適しており、健康改善への動機付けになる可能性が高い
- negative: メッセージが受信者の特性に適していない、または健康改善への動機付けにならない可能性が高い

### 回答手順 ###
1. メッセージと受信者の特性を慎重に分析してください。
2. 評価基準に基づいて判断を行ってください。
3. 最終的な回答（positiveまたはnegative）のみを出力してください。\
"""

    def __init__(
        self,
        api_key: str,
        model: str,
    ):
        self.model = model
        self.client = OpenAI(api_key)

    def predict(
        self,
        completion: str,
        characteristics: UserCharacteristics,
    ) -> Label:
        prompt = self.PROMPT_TEMPLATE.format(
            message=completion,
            age_group=characteristics.age_group.value,
            gender=characteristics.gender.value,
            stage=characteristics.stage.value,
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.SYSTEM},
                {"role": "user", "content": prompt},
            ],
        )

        content = response.choices[0].message.content

        if "negative" in content:
            return Label.NEGATIVE
        return Label.POSITIVE
