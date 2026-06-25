from dataclasses import dataclass
from typing import Any, List

import peft
import torch
import torch.nn as nn
import trl
from datasets import Dataset
from peft import PeftModel
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from transformers.data.data_collator import DataCollatorMixin
from trl import KTOTrainer as TRLKTOTrainer
from trl.trainer.utils import pad


def assemble_kto_sequence(
    prompt_ids: list[int],
    answer_ids: list[int],
    *,
    max_length: int | None,
) -> tuple[list[int], list[int]]:
    """prompt + completion を組み立てる。長い場合は prompt 先頭を切り、completion を必ず残す。"""
    prompt_ids = list(prompt_ids)
    answer_ids = list(answer_ids)
    if max_length is None:
        full_ids = prompt_ids + answer_ids
        return full_ids, [-100] * len(prompt_ids) + answer_ids

    if len(answer_ids) >= max_length:
        answer_ids = answer_ids[-max_length:]
        prompt_ids = []
    elif len(prompt_ids) + len(answer_ids) > max_length:
        prompt_ids = prompt_ids[-(max_length - len(answer_ids)) :]

    full_ids = prompt_ids + answer_ids
    labels = [-100] * len(prompt_ids) + answer_ids
    return full_ids, labels


@dataclass
class DataCollatorForKTO(DataCollatorMixin):
    """TRL 既定 collator の代替。長い Harmony prompt でも completion の loss 対象を残す。"""

    pad_token_id: int
    max_length: int | None = None
    return_tensors: str = "pt"

    def torch_call(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        batch: dict[str, Any] = {}
        for prefix, ids_key in [("completion", "completion_ids"), ("KL_completion", "KL_completion_ids")]:
            if ids_key not in examples[0]:
                continue

            full_ids_list: list[list[int]] = []
            labels_list: list[list[int]] = []
            for ex in examples:
                full_ids, labels = assemble_kto_sequence(
                    ex["prompt_ids"],
                    ex[ids_key],
                    max_length=self.max_length,
                )
                full_ids_list.append(full_ids)
                labels_list.append(labels)

            batch[f"{prefix}_input_ids"] = pad(
                [torch.tensor(ids, dtype=torch.int64) for ids in full_ids_list],
                padding_value=self.pad_token_id,
                padding_side="right",
            )
            batch[f"{prefix}_attention_mask"] = pad(
                [torch.ones(len(ids), dtype=torch.int64) for ids in full_ids_list],
                padding_value=0,
                padding_side="right",
            )
            batch[f"{prefix}_labels"] = pad(
                [torch.tensor(lbl, dtype=torch.int64) for lbl in labels_list],
                padding_value=-100,
                padding_side="right",
            )

        if "reference_logps" in examples[0]:
            batch["reference_logps"] = torch.tensor([ex["reference_logps"] for ex in examples])
        if "reference_KL_logps" in examples[0]:
            batch["reference_KL_logps"] = torch.tensor([ex["reference_KL_logps"] for ex in examples])
        batch["label"] = [ex["label"] for ex in examples]
        return batch


class SingleDeviceKTOTrainer(TRLKTOTrainer):
    """複数 GPU 環境でも DataParallel を使わず単一デバイスで KTO する。"""

    def _wrap_model(
        self,
        model: nn.Module,
        training: bool = True,
        dataloader=None,
    ) -> nn.Module:
        if self.accelerator.unwrap_model(model, keep_torch_compile=False) is not model:
            return model
        return model


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
    gradient_checkpointing_kwargs: dict[str, Any] | None
    remove_unused_columns: bool
    max_length: int
    beta: float = 0.3
    precompute_ref_log_probs: bool = True
    precompute_ref_batch_size: int = 1


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
            gradient_checkpointing_kwargs=self.args.gradient_checkpointing_kwargs,
            remove_unused_columns=self.args.remove_unused_columns,
            max_length=self.args.max_length,
            beta=self.args.beta,
            precompute_ref_log_probs=self.args.precompute_ref_log_probs,
            precompute_ref_batch_size=self.args.precompute_ref_batch_size,
        )

        peft_config = None
        if not isinstance(self.model, PeftModel):
            peft_config = peft.LoraConfig(
                r=self.peft_config.r,
                lora_alpha=self.peft_config.lora_alpha,
                lora_dropout=self.peft_config.lora_dropout,
                bias=self.peft_config.bias,
                task_type=self.peft_config.task_type,
                target_modules=self.peft_config.target_modules,
            )

        data_collator = DataCollatorForKTO(
            pad_token_id=self.processing_class.pad_token_id,
            max_length=self.args.max_length,
        )
        if args.remove_unused_columns:
            args.remove_unused_columns = False

        trainer = SingleDeviceKTOTrainer(
            model=self.model,
            args=args,
            processing_class=self.processing_class,
            train_dataset=self.train_dataset,
            peft_config=peft_config,
            data_collator=data_collator,
        )

        return trainer

    def train(self):
        self.trainer.train()

    def save_model(self, output_dir):
        self.trainer.save_model(output_dir)
