from abc import ABC

from src.domain import Label, UserCharacteristics


class EffectPredictor(ABC):
    def predict(
        self,
        completion: str,
        characteristics: UserCharacteristics,
    ) -> Label:
        raise NotImplementedError()
