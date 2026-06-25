from src.infrastructure.harmony import build_harmony_completion


class LlmJpCompletion:
    """Harmony 出力。content は analysis prefill 直後の継続（保存・分析用）。"""

    def __init__(
        self,
        thought: str,
        response: str,
        raw: str = "",
        generated: str = "",
        *,
        generated_tokens: int = 0,
        thinking_tokens: int = 0,
        final_tokens: int = 0,
    ):
        self.thought = thought
        self.response = response
        self.raw = raw
        self.generated_tokens = generated_tokens
        self.thinking_tokens = thinking_tokens
        self.final_tokens = final_tokens
        self.content = generated.strip() or build_harmony_completion(thought, response)
