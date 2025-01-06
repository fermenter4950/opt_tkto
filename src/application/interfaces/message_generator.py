from abc import ABC
from typing import List

from src.domain import Completion, Instruction


class MessageGenerator(ABC):
    def generate(self, prompt: Instruction, k: int) -> List[Completion]:
        raise NotImplementedError()
