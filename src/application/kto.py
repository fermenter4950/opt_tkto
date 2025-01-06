from typing import List

from src.application.interfaces import (
    DatasetRepository,
    EffectPredictor,
    ModelRepository,
)
from src.domain import KTTOConfig, TrainingResult, UserCharacteristics


class KTTOTrainer:
    """
    ユースケースを実装するApplicationサービス
    """

    def __init__(
        self,
        model_repository: ModelRepository,
        dataset_repository: DatasetRepository,
        effect_predictor: EffectPredictor,
        config: KTTOConfig,
    ):
        self.model_repository = model_repository
        self.dataset_repository = dataset_repository
        self.effect_predictor = effect_predictor
        self.config = config

    def train(
        self, base_messages: List[str], characteristics_list: List[UserCharacteristics]
    ) -> List[TrainingResult]:
        results = []
        for i in range(self.config.n_iter):
            pcl_set_list = self._create_training_data(
                base_messages, characteristics_list
            )
            result = self._train_iteration(i, pcl_set_list)
            results.append(result)
        return results
