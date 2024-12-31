import json
import re
from typing import Tuple


class Completion:
    """
    Represents a completion of a sentence.

    Attributes:
        completion: The completion of a sentence.
        thought: The thought of a user.
        response: The response of a user.
    """

    def __init__(
        self,
        content: str,
    ):
        self.content = content
        self.thought, self.response = self._decompose(content)

    def _decompose(self, content: str) -> Tuple[str, str]:
        pattern = r'"([^"]*)"'  # ダブルクォートで囲まれた部分を検出

        def replace_newlines(match):
            return '"' + match.group(1).replace("\n", "\\n") + '"'

        content_escaped = re.sub(pattern, replace_newlines, content)
        data = json.loads(content_escaped)
        thought: str = data["thought"]
        message: str = data["message"]
        return thought, message


if __name__ == "__main__":
    completion = Completion(
        content="""{
   "thought":"20代〜30代の男性は、運動に興味を示す一方で、忙しい生活が続き、運動習慣がついていない人が多く、行動変容ステージは関心期〜準備期と考えられる。したがって、メッセージは、運動の重要性を再確認し、簡単に始められる方法を提案することで、行動に移したくなるように最適化する。",
   "message":"🏃‍♂️運動を始めよう🏃‍♂️
実は、運動は生活習慣病やがんのリスクを減らすだけでなく、軽い気分障害の予防・解消にも効果的です！忙しい日々でも、短時間の運動や家でのストレッチから始めることができます。まずは、5分から始めてみてください！ #ラビユキ #Domingo"
}"""
    )
    print(completion.content)
    print(completion.thought)
    print(completion.response)
