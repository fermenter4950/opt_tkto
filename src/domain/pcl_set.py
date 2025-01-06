from dataclasses import dataclass

from src.domain.label import Label


@dataclass
class PCLSet:
    prompt: str
    completion: str
    label: Label

    def to_dict(self):
        return {
            "prompt": self.prompt,
            "completion": self.completion,
            "label": self.label == Label.POSITIVE,
        }
