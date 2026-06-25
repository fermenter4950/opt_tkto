import os
from typing import Optional

from peft import PeftModel

from src.application.interfaces import ModelRepository


class ModelRepositoryImpl(ModelRepository):
    """
    モデルの永続化を担当するRepository
    """

    def __init__(self, base_model_path: str, output_base_dir: str):
        self.base_model_path = base_model_path
        self.output_base_dir = output_base_dir

    def save_model(self, model: PeftModel, iteration: int) -> str:
        output_dir = self._get_output_dir(iteration)
        model.save_pretrained(output_dir)
        return output_dir

    def load_model(self, iteration: Optional[int] = None) -> PeftModel:
        if iteration is None:
            return self._load_base_model()
        return self._load_peft_model(iteration)

    def _get_output_dir(self, iteration: int) -> str:
        return os.path.join(self.output_base_dir, f"iteration_{iteration}")
