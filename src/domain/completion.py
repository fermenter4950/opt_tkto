import json
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

    def _decompose(content: str) -> Tuple[str, str]:
        data = json.loads(content)
        thought: str = data["thought"]
        message: str = data["message"]
        return thought, message
