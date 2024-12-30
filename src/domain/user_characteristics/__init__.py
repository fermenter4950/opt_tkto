from dataclasses import dataclass

from src.domain.user_characteristics.age_group import AgeGroup
from src.domain.user_characteristics.behavior_stage import BehaviorStage
from src.domain.user_characteristics.gender import Gender

__all__ = [
    AgeGroup,
    BehaviorStage,
    Gender,
]


@dataclass
class UserCharacteristics:
    gender: Gender
    age_group: AgeGroup
    stage: BehaviorStage
