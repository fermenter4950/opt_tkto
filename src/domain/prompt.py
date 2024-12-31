from src.domain.user_characteristics import UserCharacteristics


class Prompt:
    """
    Represents a prompt for generating a sentence.

    Attributes:
        base_message: The base message of a sentence.
        characteristics: The characteristics of a user.
        content: The content of a sentence.
    """

    PROMPT_TEMPLATE = """
あなたは，"運動促進メッセージ"を最適化するアシスタントです。以下のベースメッセージと受信者情報をもとに、対象の受信者が"運動への意識を高め、実際に行動に移したくなる"ようにメッセージを再構築してください．

### ベースメッセージ ###
{base_message}

### 受信者情報 ###
性別: {gender}
年齢: {age_group}
行動変容ステージ: {stage}

### 出力形式 ###
必ずJSON形式で出力してください
{{
  "thought": <思考> // あなたが受信者情報を踏まえてどのようにメッセージを最適化したか、その思考プロセスをできる限り詳しく記述する。,
  "message": <最適化されたメッセージ> // 実際に最適化されたメッセージ本文。
}}

### 必要条件・留意点 ###
thoughtフィールドには、メッセージを最適化するために考慮した内容（受信者の特徴や行動変容ステージへの配慮点など）を書き出してください。messageフィールドには，実際に受信者が「運動してみよう」と思えるようなメッセージの完成形を書いてください。\
"""

    def __init__(
        self,
        base_message: str,
        characteristics: UserCharacteristics,
    ):
        self.base_message = base_message
        self.characteristics = characteristics
        self.content = self._generate()

    def _generate(self) -> str:
        return self.PROMPT_TEMPLATE.format(
            base_message=self.base_message,
            gender=self.characteristics.gender.value,
            age_group=self.characteristics.age_group.value,
            stage=self.characteristics.stage.value,
        )
