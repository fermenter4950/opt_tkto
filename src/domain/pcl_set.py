from dataclasses import dataclass

from src.domain.completion import Completion
from src.domain.label import Label
from src.domain.prompt import Prompt


@dataclass
class PCLSet:
    prompt: Prompt
    completion: Completion
    label: Label

    def to_dict(self):
        return {
            "prompt": self.prompt.content,
            "completion": self.completion.content,
            "label": self.label.value,
        }
