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
        return random.choice([Label.NEGATIVE, Label.POSITIVE])


if __name__ == "__main__":
    from src.domain import AgeGroup, BehaviorStage, Gender

    effect_predictor_fake = EffectPredictorFake()
    for _ in range(5):
        print(
            effect_predictor_fake.predict(
                "Hello",
                UserCharacteristics(
                    gender=Gender.FEMALE,
                    age_group=AgeGroup.FORTIES_TO_FIFTIES,
                    stage=BehaviorStage.ACTION_TO_MAINTENANCE,
                ),
            )
        )
