from dataclasses import dataclass

from src.domain.label import Label


@dataclass
class PCLSet:
    prompt: str
    completion: str
    label: Label
    thinking: str = ""
    harmony_completion: str = ""

    def to_dict(self):
        row = {
            "prompt": self.prompt,
            "completion": self.completion,
            "label": self.label == Label.POSITIVE,
        }
        if self.thinking:
            row["thinking"] = self.thinking
        if self.harmony_completion:
            row["harmony_completion"] = self.harmony_completion
        return row
