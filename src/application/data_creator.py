from typing import Callable, List

from src.application.interfaces import EffectPredictor
from src.domain import PCLSet, Prompt, UserCharacteristics
from src.domain.completion import Completion


class DataCreator:
    def __init__(
        self,
        message_generator: Callable[[Prompt, int], List[Completion]],
        effect_predictor: EffectPredictor,
    ):
        self.message_generator = message_generator
        self.effect_predictor = effect_predictor

    def execute(
        self,
        base_messages: List[str],
        characteristics_list: List[UserCharacteristics],
        k: int,
    ) -> List[PCLSet]:
        pcl_set_list: List[PCLSet] = []
        for base_message in base_messages:
            for characteristics in characteristics_list:
                pcl_set_list.extend(
                    self._generate_pcl_set(base_message, characteristics, k)
                )
        return pcl_set_list

    def _generate_pcl_set(
        self,
        base_message: str,
        characteristic: UserCharacteristics,
        k: int,
    ):
        prompt = Prompt(base_message, characteristic)
        completions = self.message_generator(prompt.content, k)
        pcl_set_list: List[PCLSet] = []
        for completion in completions:
            effect = self.effect_predictor.predict(completion.response, characteristic)
            pcl_set = PCLSet(prompt, completion, effect)
            pcl_set_list.append(pcl_set)
        return pcl_set_list
