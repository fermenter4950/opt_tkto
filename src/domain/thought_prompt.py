import json
import re
from typing import Tuple


class ThoughtPrompt:
    """
    Represents a prompt for generating a sentence.

    Attributes:
        base_message: The base message of a sentence.
        characteristics: The characteristics of a user.
        content: The content of a sentence.
    """

    TEMPLATE = """\
以下のようにタスクに対して包括的で詳細な回答をしてください。回答する前に思考プロセスを書き出し、その後に回答を記述してください。
必ず##出力形式##に従ってJSON形式で出力してください。
    
##タスク##
{instruction}

##出力形式##
必ずJSON形式で以下の形式のJsonオブジェクトのみを出力してください。
{{
  "thought": "思考",
  "response": "回答"
}}
"""

    def __init__(
        self,
        instruction: str,
    ):
        self.content = self._generate(instruction)

    def _generate(self, instruction: str) -> str:
        prompt = self.TEMPLATE.format(
            instruction=instruction,
        )

        return prompt

    @staticmethod
    def json_decoder(content: str) -> Tuple[str, str]:
        pattern = r'"([^"]*)"'  # ダブルクォートで囲まれた部分を検出

        def replace_newlines(match):
            return '"' + match.group(1).replace("\n", "\\n") + '"'

        try:
            content_escaped = re.sub(pattern, replace_newlines, content)
            data = json.loads(content_escaped)
        except json.JSONDecodeError as e:
            raise ValueError("JSON形式のデータが見つかりません") from e

        if "thought" not in data:
            raise ValueError("thought フィールドが見つかりません")
        if "response" not in data:
            raise ValueError("response フィールドが見つかりません")

        thought: str = data["thought"]
        response: str = data["response"]
        return thought, response
