import argparse
import logging
import os
import shutil
import warnings
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from src.application.tkto import TKTOTrainer
from src.domain import (
    AgeGroup,
    BehaviorStage,
    Gender,
    UserCharacteristics,
)
from src.domain.prompt import Instruction
from src.domain.run_config import RunConfig, load_run_config, resolve_model_output_dir
from src.domain.tkto_config import TKTOConfig
from src.infrastructure.llm import create_completion_generator, list_model_names, load_llm
from src.infrastructure.repository.effect_predictor_bert import EffectPredictorBERT
from src.infrastructure.repository.effect_predictor_llm import EffectPredictorLLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TKTO 学習パイプライン")
    parser.add_argument(
        "--config",
        default=os.getenv("TKTO_CONFIG", "configs/llmjp_elyza.yaml"),
        help="実行設定 YAML のパス",
    )
    parser.add_argument(
        "--model",
        help="optimizer.model を上書き（YAML より優先）",
    )
    parser.add_argument(
        "--effect-model",
        help="scorer.model を上書き（scorer.type=llm 時）",
    )
    return parser.parse_args()


def apply_cli_overrides(config: RunConfig, args: argparse.Namespace) -> RunConfig:
    if args.model:
        config.optimizer.model = args.model
    if args.effect_model:
        config.scorer.model = args.effect_model
    env_output = os.getenv("OUTPUT_DIR")
    if env_output:
        config.output_dir = env_output
    return config


def load_messages(path: str, *, column: str | None = None) -> list[str]:
    data_path = Path(path)
    sep = "\t" if data_path.suffix.lower() == ".tsv" else ","
    df = pd.read_csv(path, sep=sep)
    if column is not None:
        message_column = column
    elif "メッセージ" in df.columns:
        message_column = "メッセージ"
    elif "message" in df.columns:
        message_column = "message"
    else:
        raise ValueError(
            "メッセージ列が見つかりません。"
            f"利用可能: {list(df.columns)}"
        )
    return df[message_column].dropna().astype(str).tolist()


def build_prompts(messages: list[str]) -> tuple[list[str], dict]:
    characteristics_list = [
        UserCharacteristics(gender=g, age_group=a, stage=s)
        for g in Gender._member_map_.values()
        for a in AgeGroup._member_map_.values()
        for s in BehaviorStage._member_map_.values()
    ]

    prompts: list[str] = []
    metadata: dict = {"characteristics": []}
    for message in messages:
        for characteristics in characteristics_list:
            instruction = Instruction(
                base_message=message,
                characteristics=characteristics,
            )
            prompts.append(instruction.content)
            metadata["characteristics"].append(characteristics)
    return prompts, metadata


def build_effect_predictor(config: RunConfig, gen_session):
    scorer_type = config.scorer.type.lower()
    if scorer_type == "bert":
        print(f"効果予測 (BERT): {config.scorer.model_path}", flush=True)
        return EffectPredictorBERT(model_path=config.scorer.model_path)
    if scorer_type == "llm":
        scorer_session = load_llm(config.scorer.model, for_scoring=True)
        print(
            f"効果予測 (LLM): {scorer_session.profile.name} "
            f"[最適化: {gen_session.profile.name} とは別重み]",
            flush=True,
        )
        return EffectPredictorLLM(
            session=scorer_session,
            adapter_path=config.scorer.adapter_path,
        )
    raise ValueError(f"未知の scorer.type: {config.scorer.type!r}（llm または bert）")


def suppress_noisy_warnings() -> None:
    """ライブラリ由来の警告を一旦抑制する（smoke / 本番ログ整理用）。"""
    warnings.filterwarnings("ignore")
    for name in ("transformers", "trl", "peft", "datasets", "accelerate"):
        logging.getLogger(name).setLevel(logging.ERROR)


if __name__ == "__main__":
    suppress_noisy_warnings()
    load_dotenv(".env")
    args = parse_args()

    config = apply_cli_overrides(load_run_config(args.config), args)
    print(f"設定: {Path(args.config).resolve()}", flush=True)
    print(
        f"最適化 LLM: {config.optimizer.model} "
        f"(登録済み: {', '.join(list_model_names())})",
        flush=True,
    )

    gen_session = load_llm(
        config.optimizer.model,
        reasoning_effort=config.optimizer.reasoning_effort,
    )
    completion_generator = create_completion_generator(
        gen_session,
        max_new_tokens=config.optimizer.max_new_tokens,
        temperature=config.optimizer.temperature,
        do_sample=config.optimizer.do_sample,
        length_penalty=config.optimizer.length_penalty,
        num_beams=config.optimizer.num_beams,
    )
    if config.optimizer.num_beams > 1:
        print(
            f"生成: beam search (num_beams={config.optimizer.num_beams}, "
            f"length_penalty={config.optimizer.length_penalty})",
            flush=True,
        )

    messages = load_messages(
        config.data.messages_csv,
        column=config.data.message_column,
    )
    print(
        f"データ: {Path(config.data.messages_csv).resolve()} "
        f"({len(messages)} 件)",
        flush=True,
    )
    prompts, metadata = build_prompts(messages)

    run_output_dir = resolve_model_output_dir(
        config.output_dir,
        gen_session.profile.name,
        timestamp_dir=config.timestamp_dir,
        run_id=config.run_id,
    )
    Path(run_output_dir).mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(args.config).resolve(), Path(run_output_dir) / "run_config.yaml")
    print(f"出力: {Path(run_output_dir).resolve()}", flush=True)

    tkto_config = TKTOConfig(
        n_iter=config.tkto.n_iter,
        batch_size=config.tkto.batch_size,
        num_of_output=config.tkto.num_of_output,
        output_dir=run_output_dir,
        kto_max_steps=config.kto.max_steps,
        kto_save_steps=config.kto.save_steps,
        kto_learning_rate=config.kto.learning_rate,
        kto_beta=config.kto.beta,
        kto_harmony_format=config.kto.harmony_format,
        kto_max_length=config.kto.max_length,
        kto_min_valid_ratio=config.kto.min_valid_ratio,
        kto_one_epoch=config.kto.one_epoch,
    )

    trainer = TKTOTrainer(
        prompts=prompts,
        config=tkto_config,
        base_model=gen_session.model,
        tokenizer=gen_session.tokenizer,
        effect_predictor=build_effect_predictor(config, gen_session),
        completion_generator=completion_generator,
        metadata=metadata,
        initial_epoch=config.tkto.initial_epoch,
        initial_step=config.tkto.initial_step,
        peft_path=config.tkto.peft_path,
    )

    trainer.train()
