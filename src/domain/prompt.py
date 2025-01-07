from src.domain.user_characteristics import UserCharacteristics


class Instruction:
    """
    Represents a prompt for generating a sentence.

    Attributes:
        base_message: The base message of a sentence.
        characteristics: The characteristics of a user.
        content: The content of a sentence.
    """

    PROMPT_TEMPLATE = """\
あなたは、"運動促進メッセージ"を最適化するアシスタントです。
以下のベースメッセージと受信者情報をもとに、対象の受信者が"運動への意識を高め、実際に行動に移したくなる"ようにメッセージを再構築してください。

###ベースメッセージ###
{base_message}

###受信者情報###
性別: {gender}
年齢: {age_group}
行動変容ステージ: {stage}\
"""

    def __init__(
        self,
        base_message: str,
        characteristics: UserCharacteristics,
        use_direct_prompt: bool = False,
    ):
        self.base_message = base_message
        self.characteristics = characteristics
        self.content = self._generate(use_direct_prompt)

    def _generate(self, use_direct_prompt: bool):
        template = self.PROMPT_TEMPLATE

        prompt = template.format(
            base_message=self.base_message,
            gender=self.characteristics.gender.value,
            age_group=self.characteristics.age_group.value,
            stage=self.characteristics.stage.value,
        )

        return prompt
