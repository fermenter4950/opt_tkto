from openai import OpenAI

from src.application.interfaces.effect_predictor import EffectPredictor
from src.domain import Label, UserCharacteristics
from src.domain.user_characteristics import age_group
from src.domain.user_characteristics.behavior_stage import BehaviorStage
from src.domain.user_characteristics.gender import Gender


class EffectPredictorImpl(EffectPredictor):
    SYSTEM = "あなたは健康促進メッセージを評価するアシスタントです。健康メッセージと受信者の特性（性別、年代、行動変容ステージ）が与えられた場合、その特性を持つ受信者がそのメッセージを読み健康改善に前向きになるかを 'positive', 'negative' のいずれかで分類してください。"
    PROMPT_TEMPLATE = """\
### 健康促進メッセージ ###
'''
{message}
'''

### 受信者の特性 ###
性別: {gender}
年代: {age_group}
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
        self.client = OpenAI(api_key=api_key)

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
        messages = [
            {"role": "system", "content": self.SYSTEM},
            {"role": "user", "content": prompt},
        ]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=1,
        )

        content = response.choices[0].message.content
        negative_similarity = self._jaccard_similarity(content, "negative")
        positive_similarity = self._jaccard_similarity(content, "positive")

        # 類似度が高い方を返す
        return (
            Label.NEGATIVE
            if negative_similarity > positive_similarity
            else Label.POSITIVE
        )

    @staticmethod
    def _jaccard_similarity(str1, str2):
        a = set(str1.split())
        b = set(str2.split())
        c = a.intersection(b)
        return float(len(c)) / (len(a) + len(b) - len(c))


if __name__ == "__main__":
    import os

    from dotenv import load_dotenv

    load_dotenv(".env")

    effect_predictor = EffectPredictorImpl(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-3.5-turbo",
    )

    completion = "＼✨運動を始めよう✨／\u3000運動を始めるきっかけは人それぞれ。でも、始める前には「何を始めればいいのかわからない」や「面倒くさい」などの不安や迷いはつきものです。\u3000でも、運動を始めることで、ストレスの解消や体重の減少、睡眠の質の改善など、多くのメリットが得られます。\u3000はじめは短い時間や、好きな運動から始めてみてください。徐々に時間や種類を増やし、運動を習慣化することが大切です。\u3000是非、運動を始めて健康な生活を手に入れましょう！#ラビユキ\u3000#Domingo https://t.co/UbtgxgNSCb"
    characteristics = UserCharacteristics(
        gender=Gender.MALE,
        age_group=age_group.AgeGroup.FORTIES_TO_FIFTIES,
        stage=BehaviorStage.CONTEMPLATION_TO_PREPARATION,
    )

    label = effect_predictor.predict(completion, characteristics)
    print(label)
