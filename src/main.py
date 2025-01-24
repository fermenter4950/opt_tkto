import os

import pandas as pd
import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.application.interfaces import effect_predictor
from src.application.tkto import TKTOTrainer
from src.domain import (
    AgeGroup,
    BehaviorStage,
    Gender,
    UserCharacteristics,
    thought_prompt,
)
from src.domain.prompt import Instruction
from src.domain.tkto_config import TKTOConfig
from src.infrastructure.repository.effect_predictor_impl import EffectPredictorImpl

if __name__ == "__main__":
    load_dotenv(".env")
    output_dir = os.getenv("OUTPUT_DIR")

    effect_predictor = EffectPredictorImpl(
        api_key=os.getenv("OPENAI_API_KEY"),
        model=os.getenv("OPENAI_MODEL_ID"),
    )

    messages = pd.read_csv("data/messages_sorted.csv")["message"].tolist()
    characteristics_list = []
    for gender in Gender._member_map_.values():
        for age_group in AgeGroup._member_map_.values():
            for stage in BehaviorStage._member_map_.values():
                characteristics_list.append(
                    UserCharacteristics(
                        gender=gender,
                        age_group=age_group,
                        stage=stage,
                    )
                )

    prompts = []
    metadata = {
        "characteristics": [],
    }
    for message in messages:
        for characteristics in characteristics_list:
            instruction = Instruction(
                base_message=message,
                characteristics=characteristics,
            )
            prompt = thought_prompt.ThoughtPrompt(
                instruction=instruction.content,
            )
            metadata["characteristics"].append(characteristics)
            prompts.append(prompt.content)

    config = TKTOConfig(
        n_iter=5,
        batch_size=12 * 15,
        num_of_output=3,
        output_dir=os.path.join(output_dir, "2024-01-14"),
    )

    base_model_path = "elyza/Llama-3-ELYZA-JP-8B"
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.pad_token = tokenizer.eos_token
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        raise ValueError("GPU is not available.")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype=torch.bfloat16, device_map="auto"
    ).to(device)

    trainer = TKTOTrainer(
        prompts=prompts,
        config=config,
        base_model=base_model,
        tokenizer=tokenizer,
        effect_predictor=effect_predictor,
        metadata=metadata,
        initial_epoch=1,
        initial_step=1,
    )

    trainer.train()
