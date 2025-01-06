from abc import ABC

from datasets import Dataset


class DatasetRepository(ABC):
    """
    データセットの永続化を担当するRepository
    """

    def save_dataset(self, dataset: Dataset, iteration: int):
        raise NotImplementedError()
