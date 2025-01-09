import os
from typing import Dict, List, Optional

import pandas as pd
from datasets import Dataset, DatasetInfo
from peft import PeftModel
from transformers import PreTrainedModel, PreTrainedTokenizer

from src.application.interfaces import EffectPredictor
from src.domain import TKTOConfig
from src.domain.completion import Completion
from src.domain.label import Label
from src.domain.pcl_set import PCLSet
from src.domain.user_characteristics import UserCharacteristics
from src.infrastructure.services.kto_trainer import KTOConfig, KTOTrainer, PeftConfig


class TKTOTrainer:
    def __init__(
        self,
        prompts: List[str],
        base_model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        effect_predictor: EffectPredictor,
        config: TKTOConfig,
        metadata: Dict[str, any],
        initial_epoch: Optional[int] = None,
        initial_step: Optional[int] = None,
        peft_path: Optional[str] = None,
    ):
        self.prompts = prompts
        self.config = config
        self.base_model = base_model
        self.peft_path = peft_path
        if peft_path is not None:
            self.model = PeftModel.from_pretrained(
                base_model,
                peft_path,
                tokenizer=tokenizer,
                device=base_model.device,
            )
        else:
            self.model = base_model
        self.effect_predictor = effect_predictor
        self.tokenizer = tokenizer
        self.metadata = metadata
        self._steps = initial_step if initial_step is not None else 0
        self._epochs = initial_epoch if initial_epoch is not None else 0

    def _get_output_dir(self) -> str:
        """iteration用の出力ディレクトリパスを生成"""
        return os.path.join(
            self.config.output_dir, f"epoch_{self._epochs}", f"step_{self._steps}"
        )

    def _generate_completion(self, prompt: str):
        while True:
            chat = [{"role": "user", "content": prompt}]
            input_ids = self.tokenizer.apply_chat_template(
                chat, return_tensors="pt"
            ).to(self.model.device)
            try:
                output_ids = self.model.generate(
                    input_ids=input_ids,
                    do_sample=True,
                    temperature=0.7,
                    max_length=2**15,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            except Exception:
                continue
            content = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            assistant = content.split("assistant")[-1]
            try:
                completion = Completion(content=assistant)
                return completion
            except Exception:
                retry_chat = [
                    # {"role": "user", "content": prompt},
                    # {"role": "assistant", "content": assistant},
                    # {
                    #     "role": "user",
                    #     "content": "thought と response フィールドを持つ正しい形式のJsonオブジェクトのみを出力してください。必ず波括弧で囲ってください",
                    # },
                    {
                        "role": "user",
                        "content": f"""
あなたは，指定されたスキーマのJSONオブジェクトを生成するアシスタントです．
以下の###assistant###をもとに，指定されたスキーマに従ったJSONオブジェクトを生成してください．

###assistant###
{assistant}

###スキーマ###
必ず以下のスキーマに従ったJSONオブジェクト作成してください．
絶対にそのJsonオブジェクトのみを出力してください．
{{
  "thought": "string",
  "response": "string"
}}

###注意###
1. 全てのフィールド名と値をダブルクォート(")で囲みなさい
2. JSON全体を波括弧{{}}で囲みなさい
3. 各フィールドがカンマ(,)で区切りなさい
4. 最後の要素の後にカンマをつけるな
""",
                    },
                ]
                input_ids = self.tokenizer.apply_chat_template(
                    retry_chat, return_tensors="pt"
                ).to(self.model.device)
                try:
                    output_ids = self.model.generate(
                        input_ids=input_ids,
                        do_sample=True,
                        temperature=0.7,
                        max_length=2**15,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )
                except Exception:
                    continue
                content = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
                assistant = content.split("assistant")[-1]
                try:
                    completion = Completion(content=assistant)
                    return completion
                except Exception:
                    assistant = assistant.replace("「", '"').replace("」", '"')
                    assistant = assistant + "}"
                    try:
                        completion = Completion(content=assistant)
                        return completion
                    except Exception:
                        continue

    def _generate_k_completions(self, prompt: str) -> List[Completion]:
        completions = []
        for i in range(self.config.num_of_output):
            completion = self._generate_completion(prompt)
            print(f"{i+1} 回目: {completion.content}", flush=True)
            completions.append(completion)
        return completions

    def _kto(self, dataset: Dataset):
        self.kto_config = KTOConfig(
            output_dir=self._get_output_dir(),
            fp16=True,
            bf16=False,
            max_steps=300,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=1,
            learning_rate=2e-4,
            max_grad_norm=0.3,
            warmup_ratio=0.03,
            weight_decay=0.001,
            save_steps=50,
            lr_scheduler_type="cosine",
            report_to="tensorboard",
            optim="paged_adamw_32bit",
            gradient_checkpointing=True,
            remove_unused_columns=True,
            max_prompt_length=1024,
            max_length=2**15,
        )

        self.peft_config = PeftConfig(
            r=64,
            lora_alpha=16,
            lora_dropout=0.1,
            bias="none",
            target_modules=["q_proj", "v_pro"],
            task_type="CAUSAL_LM",
        )

        trainer = KTOTrainer(
            model=self.model,
            args=self.kto_config,
            processing_class=self.tokenizer,
            train_dataset=dataset,
            peft_config=self.peft_config,
        )

        trainer.train()
        trainer.save_model(self._get_output_dir())
        del trainer

    def train_one_step(
        self, selected_prompts: List[str], characteristics: List[UserCharacteristics]
    ):
        # selected_prompts の prompt を key とする、辞書を作成
        prompt_dict = {
            prompt: {
                "completions": [],
                "labels": [],
            }
            for prompt in selected_prompts
        }

        print("出力の生成を開始します。", flush=True)
        # 出力の生成
        for prompt in selected_prompts:
            completions = self._generate_k_completions(prompt)
            prompt_dict[prompt]["completions"].extend(completions)

        print("ラベル付与を開始します。", flush=True)
        # ラベル付与
        labels: Label = []
        for prompt, characteristics in zip(selected_prompts, characteristics):
            completions = prompt_dict[prompt]["completions"]
            for completion in completions:
                label = self.effect_predictor.predict(
                    completion.response,
                    characteristics,
                )
                labels.append(label)
            prompt_dict[prompt]["labels"] = labels

        # PCLSet(prompt, completion, label) として全て保存する
        pcl_sets = []
        for prompt in selected_prompts:
            completions: List[Completion] = prompt_dict[prompt]["completions"]
            labels = prompt_dict[prompt]["labels"]
            for completion, label in zip(completions, labels):
                pcl_set = PCLSet(
                    prompt=prompt,
                    completion=completion.content,
                    label=label,
                )
                pcl_sets.append(pcl_set)

        # データセットへ変換し、中間生成物として保存
        pcl_df = pd.DataFrame([pcl_set.to_dict() for pcl_set in pcl_sets])
        dataset = Dataset.from_pandas(
            pcl_df,
            info=DatasetInfo(
                dataset_name=f"pcl_dataset_{self._epochs}_{self._steps}",
            ),
        )
        dataset.save_to_disk(self._get_output_dir())

        # KTOによるアライメント実行
        self._kto(dataset)

        # モデルの更新
        del self.model
        self.model = PeftModel.from_pretrained(
            self.base_model,
            self._get_output_dir(),
        )

    def train(self):
        while self._epochs < self.config.n_iter:
            while self._steps < len(self.prompts) // self.config.batch_size:
                start = self._steps * self.config.batch_size
                end = (self._steps + 1) * self.config.batch_size
                selected_prompts = self.prompts[
                    start : end if end < len(self.prompts) else len(self.prompts)
                ]
                characteristics = self.metadata["characteristics"][
                    start : end if end < len(self.prompts) else len(self.prompts)
                ]
                self.train_one_step(selected_prompts, characteristics)
                self._steps += 1
            self._steps = 0
            self._epochs += 1
