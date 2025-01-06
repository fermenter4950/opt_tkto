from src.domain.completion import Completion
from src.domain.ktto_config import KTTOConfig, TrainingResult
from src.domain.label import Label
from src.domain.pcl_set import PCLSet
from src.domain.prompt import Instruction
from src.domain.tkto_config import TKTOConfig
from src.domain.user_characteristics import (
    AgeGroup,
    BehaviorStage,
    Gender,
    UserCharacteristics,
)

__all__ = [
    Completion,
    Label,
    Instruction,
    AgeGroup,
    BehaviorStage,
    Gender,
    UserCharacteristics,
    PCLSet,
    KTTOConfig,
    TrainingResult,
    TKTOConfig,
]

if __name__ == "__main__":
    completion = Completion(
        "私は、プログラミングを勉強しています。",
        lambda x: x.split("、"),
    )
    print(completion.thought)
    print(completion.response)

    prompt = Instruction(
        "あなたの情報を教えてください。",
        UserCharacteristics(
            gender=Gender.FEMALE,
            age_group=AgeGroup.TWENTIES_TO_THIRTIES,
            stage=BehaviorStage.ACTION_TO_MAINTENANCE,
        ),
        template="{base_message}, {gender}, {age_group}, {stage}",
    )
    print(prompt.content)
