from src.domain.completion import Completion
from src.domain.label import Label
from src.domain.pcl_set import PCLSet
from src.domain.prompt import Prompt
from src.domain.user_characteristics import (
    AgeGroup,
    BehaviorStage,
    Gender,
    UserCharacteristics,
)

__all__ = [
    Completion,
    Label,
    Prompt,
    AgeGroup,
    BehaviorStage,
    Gender,
    UserCharacteristics,
    PCLSet,
]

if __name__ == "__main__":
    completion = Completion(
        "私は、プログラミングを勉強しています。",
        lambda x: x.split("、"),
    )
    print(completion.thought)
    print(completion.response)

    prompt = Prompt(
        "あなたの情報を教えてください。",
        UserCharacteristics(
            gender=Gender.FEMALE,
            age_group=AgeGroup.TWENTIES_TO_THIRTIES,
            stage=BehaviorStage.ACTION_TO_MAINTENANCE,
        ),
        template="{base_message}, {gender}, {age_group}, {stage}",
    )
    print(prompt.content)
