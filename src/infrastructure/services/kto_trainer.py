from dataclasses import dataclass
from typing import List

import peft
import trl
from datasets import Dataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase


@dataclass
class KTOConfig:
    output_dir: str
    max_steps: int
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    max_grad_norm: float
    warmup_ratio: float
    weight_decay: float
    save_steps: int
    optim: str
    lr_scheduler_type: str
    report_to: str
    fp16: bool
    bf16: bool
    gradient_checkpointing: bool
    remove_unused_columns: bool
    max_prompt_length: int
    max_length: int


@dataclass
class PeftConfig:
    target_modules: List[str]
    r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


class KTOTrainer:
    def __init__(
        self,
        model: PreTrainedModel,
        args: KTOConfig,
        processing_class: PreTrainedTokenizerBase,
        train_dataset: Dataset,
        peft_config=PeftConfig,
    ):
        self.model = model
        self.args = args
        self.processing_class = processing_class
        self.train_dataset = train_dataset
        self.peft_config = peft_config
        self.trainer = self._create_trainer()

    def _create_trainer(self):
        args = trl.KTOConfig(
            output_dir=self.args.output_dir,
            fp16=self.args.fp16,
            bf16=self.args.bf16,
            max_steps=self.args.max_steps,
            per_device_train_batch_size=self.args.per_device_train_batch_size,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            optim=self.args.optim,
            learning_rate=self.args.learning_rate,
            lr_scheduler_type=self.args.lr_scheduler_type,
            max_grad_norm=self.args.max_grad_norm,
            warmup_ratio=self.args.warmup_ratio,
            weight_decay=self.args.weight_decay,
            save_steps=self.args.save_steps,
            report_to=self.args.report_to,
            gradient_checkpointing=self.args.gradient_checkpointing,
            remove_unused_columns=self.args.remove_unused_columns,
            max_prompt_length=self.args.max_prompt_length,
            max_length=self.args.max_length,
        )

        peft_config = peft.LoraConfig(
            r=self.peft_config.r,
            lora_alpha=self.peft_config.lora_alpha,
            lora_dropout=self.peft_config.lora_dropout,
            bias=self.peft_config.bias,
            task_type=self.peft_config.task_type,
            target_modules=self.peft_config.target_modules,
        )

        trainer = trl.KTOTrainer(
            model=self.model,
            args=args,
            processing_class=self.processing_class,
            train_dataset=self.train_dataset,
            peft_config=peft_config,
        )

        return trainer

    def train(self):
        trainer = self._create_trainer()
        trainer.train()

    def save_model(self, output_dir):
        self.trainer.save_model(output_dir)
