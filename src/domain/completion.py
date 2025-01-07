from src.domain.thought_prompt import ThoughtPrompt


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
        self.thought, self.response = ThoughtPrompt.json_decoder(content)
