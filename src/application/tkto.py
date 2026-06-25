import gc
import math
import os
import shutil
from dataclasses import replace
from queue import Queue
from threading import Thread
from typing import Dict, List, Optional

import pandas as pd
import torch
from datasets import Dataset, DatasetInfo
from peft import PeftModel
from transformers import PreTrainedModel, PreTrainedTokenizer

from src.application.interfaces import CompletionGenerator, EffectPredictor
from src.domain import TKTOConfig
from src.domain.label import Label
from src.domain.pcl_set import PCLSet
from src.domain.user_characteristics import UserCharacteristics
from src.infrastructure.harmony import HarmonyKtoFormat
from src.infrastructure.repository.effect_predictor_llm import EffectPredictorLLM
from src.infrastructure.services.kto_trainer import (
    KTOConfig,
    KTOTrainer,
    PeftConfig,
    assemble_kto_sequence,
)

_PIPELINE_QUEUE_MAXSIZE = 8
_PIPELINE_SENTINEL = object()


class TKTOTrainer:
    def __init__(
        self,
        prompts: List[str],
        base_model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        effect_predictor: EffectPredictor,
        completion_generator: CompletionGenerator,
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
                is_trainable=True,
            )
        else:
            self.model = base_model
        self.effect_predictor = effect_predictor
        self.completion_generator = completion_generator
        self.tokenizer = tokenizer
        self.metadata = metadata
        self._steps = initial_step if initial_step is not None else 0
        self._epochs = initial_epoch if initial_epoch is not None else 0
        self._last_kto_max_steps: int | None = None
        self._sync_generator_model()

    def _sync_generator_model(self) -> None:
        """生成に使う LLM を KTO 後の self.model（Peft 含む）と揃える。"""
        self.model.eval()
        self.completion_generator.session = replace(
            self.completion_generator.session,
            model=self.model,
        )

    def _get_output_dir(self) -> str:
        """iteration用の出力ディレクトリパスを生成"""
        return os.path.join(
            self.config.output_dir, f"epoch_{self._epochs}", f"step_{self._steps}"
        )

    def _generate_completion(self, instruction: str):
        return self.completion_generator.generate(instruction)

    def _scorer_worker(
        self,
        queue: Queue,
        prompt_dict: Dict[str, dict],
        *,
        total_items: int,
        scored_count: List[int],
        error_holder: List[BaseException],
    ) -> None:
        while True:
            item = queue.get()
            try:
                if item is _PIPELINE_SENTINEL:
                    return
                prompt, completion, characteristics = item
                label = self.effect_predictor.predict(
                    completion.response,
                    characteristics,
                    threshold=0.5,
                )
                prompt_dict[prompt]["completions"].append(completion)
                prompt_dict[prompt]["labels"].append(label)
                scored_count[0] += 1
                print(
                    f"  [採点 {scored_count[0]}/{total_items}] "
                    f"{label.name.lower()}",
                    flush=True,
                )
            except BaseException as exc:
                error_holder.append(exc)
                return
            finally:
                queue.task_done()

    def _pipeline_generate_and_score(
        self,
        selected_prompts: List[str],
        characteristics: List[UserCharacteristics],
        prompt_dict: Dict[str, dict],
    ) -> None:
        """生成（GPU0）と採点（GPU1）をキューで重ねて実行する。"""
        total_prompts = len(selected_prompts)
        k = self.config.num_of_output
        total_items = total_prompts * k
        queue: Queue = Queue(maxsize=_PIPELINE_QUEUE_MAXSIZE)
        scored_count = [0]
        error_holder: List[BaseException] = []

        worker = Thread(
            target=self._scorer_worker,
            args=(queue, prompt_dict),
            kwargs={
                "total_items": total_items,
                "scored_count": scored_count,
                "error_holder": error_holder,
            },
            daemon=True,
        )
        worker.start()

        try:
            for idx, (prompt, chars) in enumerate(
                zip(selected_prompts, characteristics), start=1
            ):
                print(f"[生成 {idx}/{total_prompts}]", flush=True)
                for i in range(k):
                    print(f"  出力 {i + 1}/{k} 生成中...", flush=True)
                    completion = self._generate_completion(prompt)
                    thought_preview = completion.thought.replace("\n", " ")[:80]
                    preview = completion.response.replace("\n", " ")[:80]
                    print(
                        f"  出力 {i + 1}/{k} 思考 ({completion.thinking_tokens} tok): "
                        f"{thought_preview}...",
                        flush=True,
                    )
                    print(
                        f"  出力 {i + 1}/{k} 完了 "
                        f"(生成 {completion.generated_tokens} tok / "
                        f"final {completion.final_tokens} tok): "
                        f"{preview}...",
                        flush=True,
                    )
                    queue.put((prompt, completion, chars))
        finally:
            queue.put(_PIPELINE_SENTINEL)
            worker.join()

        if error_holder:
            raise error_holder[0]

    def _ensure_default_adapter_trainable(self, model=None) -> int:
        """LoRA を trainable に戻し、学習可能パラメータ数を返す。

        KTO 保存時の inference_mode=True だと requires_grad が全て False になり、
        rewards/chosen=0・grad_norm=0 のまま学習が止まる。
        """
        model = self.model if model is None else model
        if isinstance(model, PeftModel):
            for cfg in model.peft_config.values():
                cfg.inference_mode = False

        trainable = 0
        for name, param in model.named_parameters():
            if ".ref." in name:
                param.requires_grad_(False)
                continue
            if "lora_" in name or ".default." in name:
                param.requires_grad_(True)
                trainable += param.numel()
        return trainable

    def _prepare_model_for_kto(self) -> None:
        """KTO 学習前にモデルを train モードへ整える（2 epoch 目以降の Peft 継続学習用）。"""
        self._ensure_default_adapter_trainable()
        self.model.train()
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False},
            )
        elif hasattr(self.model, "enable_input_require_grads"):
            self.model.enable_input_require_grads()

    def _release_scorer_gpu(self) -> None:
        """KTO 前に採点モデルを GPU から外し、GPU1 の VRAM を空ける。"""
        if not isinstance(self.effect_predictor, EffectPredictorLLM):
            return
        scorer_model = self.effect_predictor.session.model
        if scorer_model is not None:
            scorer_model.to("cpu")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _restore_scorer_gpu(self) -> None:
        """次 step の採点用に scorer を GPU1 へ戻す。"""
        if not isinstance(self.effect_predictor, EffectPredictorLLM):
            return
        scorer_model = self.effect_predictor.session.model
        if scorer_model is not None and torch.cuda.device_count() >= 2:
            scorer_model.to("cuda:1")
        elif scorer_model is not None:
            scorer_model.to("cuda:0")

    def _log_gpu_memory(self, label: str) -> None:
        if not torch.cuda.is_available():
            return
        parts = []
        for index in range(torch.cuda.device_count()):
            free, total = torch.cuda.mem_get_info(index)
            parts.append(f"cuda:{index} {free / 2**30:.1f}/{total / 2**30:.1f} GiB free")
        print(f"VRAM [{label}] " + ", ".join(parts), flush=True)

    def _evict_models_from_gpu(self) -> None:
        """KTO 前に生成・採点モデルを GPU から完全に退避する。

        PeftModel.cpu() だけでは generator 側の参照が残り、
        HF Trainer が cuda:0 に載せようとして OOM になることがある。
        """
        self._release_scorer_gpu()

        gen_model = self.completion_generator.session.model
        if gen_model is not None:
            gen_model.to("cpu")

        if self.model is not None:
            self.model.to("cpu")
        if self.base_model is not None:
            self.base_model.to("cpu")

        self.completion_generator.session = replace(
            self.completion_generator.session,
            model=self.base_model,
        )

        gc.collect()
        if torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                with torch.cuda.device(index):
                    torch.cuda.empty_cache()

    def _assert_kto_vram(self, device: torch.device, *, min_free_gib: float = 18.0) -> None:
        if not torch.cuda.is_available():
            return
        index = device.index if device.index is not None else 0
        free, total = torch.cuda.mem_get_info(index)
        free_gib = free / 2**30
        if free_gib < min_free_gib:
            print(
                f"警告: KTO 用 GPU cuda:{index} の空き VRAM が少なめです "
                f"({free_gib:.1f}/{total / 2**30:.1f} GiB free)。KTO を試行します。",
                flush=True,
            )

    def _kto_device(self) -> torch.device:
        """KTO 学習デバイス。

        HuggingFace Trainer / Accelerate は常に cuda:0 を使うため、
        生成モデルを GPU から退避したうえで cuda:0 で KTO する。
        （cuda:1 に載せても Trainer 初期化時に cuda:0 へ戻り二重配置で OOM しやすい）
        """
        if not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device("cuda:0")

    def _prepare_for_kto(self) -> None:
        """KTO 前: scorer / 生成モデルを CPU へ退避し、空き GPU を KTO 専用にする。"""
        self._log_gpu_memory("KTO前（解放前）")
        self._evict_models_from_gpu()

        kto_device = self._kto_device()
        self._assert_kto_vram(kto_device)
        torch.cuda.set_device(kto_device)
        self.model = self.model.to(kto_device)
        self._prepare_model_for_kto()
        self._log_gpu_memory(f"KTO前（配置後 {kto_device}）")

    def _use_gradient_checkpointing(self) -> bool:
        """KTO 中の活性化メモリを抑える（Peft 継続学習でも有効）。"""
        return True

    def _resolve_kto_max_steps(self, dataset_size: int, per_device_batch_size: int = 2) -> int:
        """KTO step 数を決める。one_epoch 時はデータ 1 周を上限とする。"""
        configured = self.config.kto_max_steps
        if dataset_size <= 0:
            return configured
        one_epoch_steps = max(1, math.ceil(dataset_size / per_device_batch_size))
        if self.config.kto_one_epoch:
            return min(configured, one_epoch_steps)
        return configured

    def _resolve_kto_adapter_path(self) -> str:
        output_dir = self._get_output_dir()
        max_steps = self._last_kto_max_steps or self.config.kto_max_steps
        checkpoint_dir = os.path.join(output_dir, f"checkpoint-{max_steps}")
        if os.path.isdir(checkpoint_dir):
            return checkpoint_dir
        return output_dir

    def _reload_model_after_kto(self) -> None:
        """KTO 後に学習済み adapter を読み直し、GPU メモリを整理する。"""
        adapter_path = self._resolve_kto_adapter_path()
        self.completion_generator.session = replace(
            self.completion_generator.session,
            model=self.base_model,
        )
        del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        gen_device = torch.device("cuda:0")
        if torch.cuda.is_available():
            torch.cuda.set_device(gen_device)
        if self.base_model.device != gen_device:
            self.base_model.to(gen_device)
        self.model = PeftModel.from_pretrained(
            self.base_model,
            adapter_path,
            is_trainable=True,
        )
        self._ensure_default_adapter_trainable()
        if hasattr(self.model, "gradient_checkpointing_disable"):
            self.model.gradient_checkpointing_disable()
        self._sync_generator_model()
        self._restore_scorer_gpu()

    def _should_run_kto(
        self,
        pcl_sets: List[PCLSet],
        *,
        total_candidates: int,
    ) -> bool:
        """KTO を実行できるか（TRL の最低要件のみ）。"""
        if len(pcl_sets) < 2:
            return False
        positive = sum(1 for item in pcl_sets if item.label == Label.POSITIVE)
        negative = len(pcl_sets) - positive
        if positive < 1 or negative < 1:
            return False
        if total_candidates <= 0:
            return False
        if self.config.kto_min_valid_ratio <= 0:
            return True
        valid_ratio = len(pcl_sets) / total_candidates
        return valid_ratio >= self.config.kto_min_valid_ratio

    def _backup_adapter_before_kto(self) -> str | None:
        """KTO 前の adapter を退避（ロールバック用）。"""
        if not isinstance(self.model, PeftModel):
            return None
        backup_dir = os.path.join(self._get_output_dir(), "pre_kto_adapter")
        if os.path.isdir(backup_dir):
            shutil.rmtree(backup_dir)
        os.makedirs(backup_dir, exist_ok=True)
        self.model.save_pretrained(backup_dir)
        return backup_dir

    def _rollback_adapter(self, backup_dir: str | None) -> None:
        """直前の adapter 退避から復元する。"""
        if backup_dir is None or not os.path.isdir(backup_dir):
            return
        self.completion_generator.session = replace(
            self.completion_generator.session,
            model=self.base_model,
        )
        del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        gen_device = torch.device("cuda:0")
        if torch.cuda.is_available():
            torch.cuda.set_device(gen_device)
        if self.base_model.device != gen_device:
            self.base_model.to(gen_device)
        self.model = PeftModel.from_pretrained(
            self.base_model,
            backup_dir,
            is_trainable=True,
        )
        self._ensure_default_adapter_trainable()
        self._sync_generator_model()

    def _validate_kto_dataset(self, dataset: Dataset) -> bool:
        """max_length 切り詰め後も completion に loss 対象トークンが残るか確認する。"""
        sample = dataset[0]
        if "prompt_ids" not in sample or "completion_ids" not in sample:
            return True
        _, labels = assemble_kto_sequence(
            sample["prompt_ids"],
            sample["completion_ids"],
            max_length=self.config.kto_max_length,
        )
        loss_tokens = sum(1 for label in labels if label != -100)
        if loss_tokens == 0:
            print(
                "警告: KTO をスキップします。"
                f" max_length={self.config.kto_max_length} で completion がすべて切り詰められています。",
                flush=True,
            )
            return False
        return True

    def _kto(self, dataset: Dataset) -> bool:
        per_device_batch_size = 2
        max_steps = self._resolve_kto_max_steps(len(dataset), per_device_batch_size)
        self._last_kto_max_steps = max_steps
        save_steps = self.config.kto_save_steps or max(max_steps // 6, 1)
        print(
            f"KTO max_steps={max_steps} "
            f"(データ {len(dataset)} 件, batch={per_device_batch_size}, "
            f"one_epoch={self.config.kto_one_epoch})",
            flush=True,
        )
        use_gradient_checkpointing = self._use_gradient_checkpointing()
        self.kto_config = KTOConfig(
            output_dir=self._get_output_dir(),
            fp16=False,
            bf16=True,
            max_steps=max_steps,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=1,
            learning_rate=self.config.kto_learning_rate,
            max_grad_norm=0.3,
            warmup_ratio=0.03,
            weight_decay=0.001,
            save_steps=save_steps,
            lr_scheduler_type="cosine",
            report_to="tensorboard",
            optim="paged_adamw_32bit",
            gradient_checkpointing=use_gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False}
            if use_gradient_checkpointing
            else None,
            remove_unused_columns=True,
            max_length=self.config.kto_max_length,
            beta=self.config.kto_beta,
        )

        self.peft_config = PeftConfig(
            r=32,
            lora_alpha=16,
            lora_dropout=0.1,
            bias="none",
            target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM",
        )

        self._prepare_for_kto()
        kto_trainer = KTOTrainer(
            model=self.model,
            args=self.kto_config,
            processing_class=self.tokenizer,
            train_dataset=dataset,
            peft_config=self.peft_config,
        )
        self.model = kto_trainer.trainer.model
        if not self._validate_kto_dataset(kto_trainer.trainer.train_dataset):
            return False
        trainable = self._ensure_default_adapter_trainable(self.model)
        print(f"KTO trainable LoRA params: {trainable:,}", flush=True)
        if trainable == 0:
            print(
                "警告: KTO をスキップします（学習可能な LoRA パラメータが 0）。",
                flush=True,
            )
            return False

        self._log_gpu_memory("KTO学習直前")
        kto_trainer.train()
        kto_trainer.save_model(self._get_output_dir())
        del kto_trainer.trainer
        del kto_trainer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return True

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

        total_prompts = len(selected_prompts)
        print(
            f"=== epoch {self._epochs} step {self._steps} ===",
            flush=True,
        )
        print(
            f"出力の生成と採点をパイプライン実行します。"
            f"（{total_prompts} プロンプト × {self.config.num_of_output} 出力）",
            flush=True,
        )
        self._pipeline_generate_and_score(
            selected_prompts, characteristics, prompt_dict
        )

        # Harmony KTO: split（thinking→prompt） or joint（全文 completion）
        kto_format: HarmonyKtoFormat = self.config.kto_harmony_format  # type: ignore[assignment]
        if kto_format not in ("split", "joint"):
            raise ValueError(
                f"未知の kto_harmony_format: {self.config.kto_harmony_format!r} "
                "（split または joint）"
            )
        print(f"Harmony KTO format: {kto_format}", flush=True)

        pcl_sets: List[PCLSet] = []
        candidate_rows: list[dict] = []
        skipped_format = 0
        for instruction in selected_prompts:
            completions: List = prompt_dict[instruction]["completions"]
            labels = prompt_dict[instruction]["labels"]
            base_prompt = self.completion_generator.build_kto_prompt(instruction)
            for completion, label in zip(completions, labels):
                issues = self.completion_generator.kto_validation_issues(
                    completion.thought, completion.response
                )
                notes = self.completion_generator.kto_quality_notes(
                    completion.thought, completion.response
                )
                candidate_rows.append(
                    {
                        "instruction": instruction,
                        "thinking": completion.thought,
                        "completion": completion.response,
                        "label": label == Label.POSITIVE,
                        "generated_tokens": completion.generated_tokens,
                        "thinking_tokens": completion.thinking_tokens,
                        "final_tokens": completion.final_tokens,
                        "kto_valid": not issues,
                        "kto_invalid_reasons": ";".join(issues),
                        "kto_quality_notes": ";".join(notes),
                    }
                )
                if issues:
                    skipped_format += 1
                    continue
                kto_row = self.completion_generator.preprocess_for_kto(
                    base_prompt,
                    completion.thought,
                    completion.response,
                    mode=kto_format,
                )
                pcl_sets.append(
                    PCLSet(
                        prompt=kto_row["prompt"],
                        completion=kto_row["completion"],
                        label=label,
                        thinking=kto_row["thinking"],
                        harmony_completion=kto_row["harmony_completion"],
                    )
                )

        if skipped_format:
            print(
                f"KTO 対象外（形式不正）: {skipped_format} 件を除外",
                flush=True,
            )

        output_dir = self._get_output_dir()
        os.makedirs(output_dir, exist_ok=True)
        pd.DataFrame(candidate_rows).to_csv(
            os.path.join(output_dir, "candidates.csv"),
            index=False,
        )

        positive = sum(1 for item in pcl_sets if item.label == Label.POSITIVE)
        print(
            f"KTO 候補: {len(pcl_sets)} 件 "
            f"(positive={positive}, negative={len(pcl_sets) - positive})",
            flush=True,
        )

        if not self._should_run_kto(
            pcl_sets,
            total_candidates=len(candidate_rows),
        ):
            valid_ratio = (
                len(pcl_sets) / len(candidate_rows) if candidate_rows else 0.0
            )
            print(
                f"KTO をスキップします（学習可能 {len(pcl_sets)}/{len(candidate_rows)} 件、"
                f"valid率 {valid_ratio:.1%} < 閾値 {self.config.kto_min_valid_ratio:.0%} "
                f"または pos/neg 不足）。",
                flush=True,
            )
            return

        print(
            f"KTO 学習: {len(pcl_sets)} 件 "
            f"(positive={positive}, negative={len(pcl_sets) - positive}、全件使用)",
            flush=True,
        )

        pcl_df = pd.DataFrame([pcl_set.to_dict() for pcl_set in pcl_sets])
        dataset = Dataset.from_pandas(
            pcl_df,
            info=DatasetInfo(
                dataset_name=f"pcl_dataset_{self._epochs}_{self._steps}",
            ),
        )
        dataset.save_to_disk(self._get_output_dir())

        backup_dir = self._backup_adapter_before_kto()
        kto_ok = False
        try:
            kto_ok = self._kto(dataset)
        except Exception as exc:
            print(f"警告: KTO 失敗（スキップして続行）: {exc}", flush=True)
        if not kto_ok:
            self._rollback_adapter(backup_dir)
            return

        # モデルの更新
        self._reload_model_after_kto()

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
