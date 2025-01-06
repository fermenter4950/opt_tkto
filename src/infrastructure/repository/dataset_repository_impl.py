import os

from datasets import Dataset

from src.application.interfaces import DatasetRepository


class DatasetRepositoryImpl(DatasetRepository):
    """
    データセットの永続化を担当するRepository
    """

    def __init__(self, output_base_dir: str):
        self.output_base_dir = output_base_dir

    def save_dataset(self, dataset: Dataset, iteration: int):
        output_dir = self._get_output_dir(iteration)
        dataset.save_to_disk(output_dir)

    def _get_output_dir(self, iteration: int) -> str:
        return os.path.join(self.output_base_dir, f"iteration_{iteration}")
