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
必ず##出力形式##に従って出力してください。
    
##タスク##
{instruction}

##出力形式##
必ず以下の形式のJsonオブジェクトのみを出力してください。
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

        content = re.sub(pattern, replace_newlines, content)

        # validation
        is_valid, error_message = _validate_response(content)
        if not is_valid:
            raise ValueError(error_message)

        # JSON文字列をパース
        data = json.loads(content)

        thought: str = data["thought"]
        response: str = data["response"]
        return thought, response


def _validate_response(json_str: str) -> Tuple[bool, str | None]:
    """
    JSON文字列がスキーマに適合するかを検証する関数

    Args:
        json_str (str): 検証するJSON文字列

    Returns:
        Tuple[bool, str | None]: (バリデーション結果, エラーメッセージ)
    """
    try:
        # JSON文字列をパース
        data = json.loads(json_str)

        # dictかどうかの確認
        if not isinstance(data, dict):
            return False, "データはオブジェクト型である必要があります"

        # 必須フィールドの確認
        required_fields = ["thought", "response"]
        for field in required_fields:
            if field not in data:
                return False, f"必須フィールド '{field}' がありません"
            if not isinstance(data[field], str):
                return False, f"フィールド '{field}' は文字列である必要があります"

        return True, None

    except json.JSONDecodeError as e:
        return False, f"不正なJSON形式です: {str(e)}"
    except Exception as e:
        return False, f"予期せぬエラーが発生しました: {str(e)}"
