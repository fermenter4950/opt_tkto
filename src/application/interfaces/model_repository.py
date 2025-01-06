from abc import ABC
from typing import Optional

from peft import PeftModel


class ModelRepository(ABC):
    """
    モデルの永続化を担当するRepository
    """

    def save_model(self, model: PeftModel, iteration: int) -> str:
        raise NotImplementedError()

    def load_model(self, iteration: Optional[int] = None) -> PeftModel:
        raise NotImplementedError()
