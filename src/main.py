import os

import pandas as pd
from dotenv import load_dotenv

from src.application.ktto import KTTOConfig, KTTOTrainer
from src.domain import AgeGroup, BehaviorStage, Gender, UserCharacteristics
from src.infrastructure.repository.effect_predictor_impl import EffectPredictorImpl

if __name__ == "__main__":
    load_dotenv(".env")
    messages = pd.read_csv("data/messages.csv")["message"].tolist()
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
    trainer = KTTOTrainer(
        base_model_path="elyza/Llama-3-ELYZA-JP-8B",
        base_messages=messages,
        effect_predictor=EffectPredictorImpl(
            api_key=os.getenv("OPENAI_API_KEY"),
            model=os.getenv("OPENAI_MODEL_ID"),
        ),
        characteristics_list=characteristics_list,
        config=KTTOConfig(
            n_iter=5,
        ),
    )
    trainer.train()
