from abc import ABC
from typing import List

from src.domain import Completion, Prompt


class MessageGenerator(ABC):
    def generate(self, prompt: Prompt, k: int) -> List[Completion]:
        raise NotImplementedError()
