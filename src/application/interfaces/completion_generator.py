from abc import ABC, abstractmethod
from typing import Any, Literal

KtoFormat = Literal["split", "joint"]


class CompletionGenerator(ABC):
    """Instruction から TKTO 用 completion を生成する。"""

    @abstractmethod
    def build_kto_prompt(self, instruction: str) -> str:
        """KTO の prompt 列に保存する文字列。"""
        raise NotImplementedError()

    @abstractmethod
    def generate(self, instruction: str) -> Any:
        """thought / response / content を持つ completion オブジェクトを返す。"""
        raise NotImplementedError()

    @abstractmethod
    def preprocess_for_kto(
        self,
        base_prompt: str,
        thinking: str,
        final: str,
        *,
        mode: KtoFormat = "split",
    ) -> dict[str, str]:
        """rollout を TRL KTO 用 prompt/completion へ変換する。"""
        raise NotImplementedError()

    @abstractmethod
    def kto_validation_issues(self, thought: str, response: str) -> list[str]:
        """KTO 学習から除外する形式問題。"""
        raise NotImplementedError()

    @abstractmethod
    def kto_quality_notes(self, thought: str, response: str) -> list[str]:
        """品質上の注意点（記録用）。"""
        raise NotImplementedError()
