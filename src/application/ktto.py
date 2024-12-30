import os
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd
import torch
from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.application.data_creator import DataCreator
from src.application.interfaces.effect_predictor import EffectPredictor
from src.domain import PCLSet, UserCharacteristics
from src.infrastructure.services.kto_trainer import KTOConfig, KTOTrainer, PeftConfig


@dataclass
class KTTOConfig:
    n_iter: int = 1
    k: int = 5


class KTTOTrainer:
    def __init__(
        self,
        base_model_path: str,
        base_messages: List[str],
        characteristics_list: List[UserCharacteristics],
        effect_predictor: EffectPredictor,
        output_base_dir: str = "./kto_lora_iterations",
        peft_path: Optional[str] = None,
        config: Optional[KTTOConfig] = None,
        use_4bit: bool = False,
    ):
        self.effect_predictor = effect_predictor
        self.base_model_path = base_model_path
        self.peft_path = peft_path
        self.base_messages = base_messages
        self.characteristics_list = characteristics_list
        self.output_base_dir = output_base_dir
        self.config = config or KTTOConfig()
        self._current_iteration = 0
        self.use_4bit = use_4bit
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path, torch_dtype=torch.bfloat16, device_map={"": 0}
        ).to(self.device)
        if peft_path is None:
            self.model = self.base_model

    def _get_last_iteration(self, path: str) -> int:
        """パスから最後のiteration番号を取得"""
        try:
            return int(path.split("_")[-1])
        except ValueError:
            return 0

    def _get_output_dir(self, iteration: int) -> str:
        """iteration用の出力ディレクトリパスを生成"""
        return os.path.join(self.output_base_dir, f"iteration_{iteration}")

    def _train_iteration(
        self,
        pcl_set_list: List[PCLSet],
    ):
        """1回の追加学習を実行"""
        # トレーナーの設定と学習実行
        output_dir = self._get_output_dir(self._current_iteration)

        self.kto_config = KTOConfig(
            output_dir=output_dir,
            fp16=True,
            bf16=False,
            max_steps=300,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=1,
            optim="paged_adamw_32bit",
            learning_rate=2e-4,
            lr_scheduler_type="cosine",
            max_grad_norm=0.3,
            warmup_ratio=0.03,
            weight_decay=0.001,
            save_steps=50,
            report_to="tensorboard",
            gradient_checkpointing=True,
        )

        self.peft_config = PeftConfig(
            r=64,
            lora_alpha=16,
            lora_dropout=0.1,
            bias="none",
            target_modules=None,
            task_type="CAUSAL_LM",
        )

        pcl_sets = [pcl_set.to_dict() for pcl_set in pcl_set_list]
        pcl_df = pd.DataFrame(pcl_sets)
        dataset = Dataset.from_pandas(pcl_df)
        dataset.save_to_disk(self._get_output_dir(self._current_iteration))

        if self.peft_path is not None:
            self.model = PeftModel.from_pretrained(
                self.base_model,
                self.peft_path,
            ).to(self.device)

        trainer = KTOTrainer(
            model=self.model,
            args=self.kto_config,
            processing_class=self.tokenizer,
            train_dataset=dataset,
            peft_config=self.peft_config,
        )

        trainer.train()
        trainer.save_model(output_dir)
        self.peft_path = output_dir
        # 現在のモデルを更新
        del trainer
        del self.model
        self.model = PeftModel.from_pretrained(
            self.base_model,
            output_dir,
        )
        return output_dir

    def train(self):
        """複数回の追加学習を実行"""

        for _ in range(self.config.n_iter):

            def _message_generator(prompt: Prompt, k: int) -> List[Completion]:
                chat = [{"role": "user", "content": prompt.content}]
                input_ids = self.tokenizer.apply_chat_template(
                    chat, return_tensors="pt"
                ).to(self.device)
                completions = []
                for _ in range(k):
                    output_ids = self.model.generate(
                        input_ids=input_ids,
                        do_sample=True,
                        temperature=0.7,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )
                    content = self.tokenizer.decode(
                        output_ids[0], skip_special_tokens=True
                    )
                    assistant = content.split("assistant")[1].strip()
                    completion = Completion(content=assistant)
                    completions.append(completion)
                return completions

            data_creator = DataCreator(
                message_generator=_message_generator,
                effect_predictor=self.effect_predictor,
            )

            results = []
            # データセットの生成
            pcl_set_list = data_creator.execute(
                base_messages=self.base_messages,
                characteristics_list=self.characteristics_list,
                k=self.config.k,
            )
            output_dir = self._train_iteration(pcl_set_list)
            results.append(output_dir)
            self._current_iteration += 1

        return results


if __name__ == "__main__":
    from src.domain import (
        AgeGroup,
        BehaviorStage,
        Completion,
        Gender,
        Label,
        PCLSet,
        Prompt,
    )

    pcl_set_list = [
        PCLSet(
            prompt=Prompt(
                base_message="Hello, how are you?",
                characteristics=UserCharacteristics(
                    age_group=AgeGroup.FORTIES_TO_FIFTIES,
                    gender=Gender.FEMALE,
                    stage=BehaviorStage.ACTION_TO_MAINTENANCE,
                ),
                template="{base_message}, {age_group}, {gender}, {stage}",
            ),
            completion=Completion(
                content="I'm fine, thank you.", decomposer=lambda x: (x, x)
            ),
            label=Label.NEGATIVE,
        ),
        PCLSet(
            prompt=Prompt(
                base_message="Hello, how are you?",
                characteristics=UserCharacteristics(
                    age_group=AgeGroup.FORTIES_TO_FIFTIES,
                    gender=Gender.FEMALE,
                    stage=BehaviorStage.ACTION_TO_MAINTENANCE,
                ),
                template="{base_message}, {age_group}, {gender}, {stage}",
            ),
            completion=Completion(content="I'm not fine.", decomposer=lambda x: (x, x)),
            label=Label.NEGATIVE,
        ),
    ]
    pcl_sets = [pcl_set.to_dict() for pcl_set in pcl_set_list]
    pcl_df = pd.DataFrame(pcl_sets)
    print(pcl_df)
    dataset = Dataset.from_pandas(pcl_df)
    print(dataset)
