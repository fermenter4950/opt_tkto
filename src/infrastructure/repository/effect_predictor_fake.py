import random

from src.application.interfaces.effect_predictor import EffectPredictor
from src.domain import Label, UserCharacteristics


class EffectPredictorFake(EffectPredictor):
    def __init__(self):
        pass

    def predict(
        self,
        completion: str,
        characteristics: UserCharacteristics,
    ) -> Label:
        return random.choice([Label.NEGATIVE, Label.NEGATIVE])
