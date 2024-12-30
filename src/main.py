from src.application.ktto import KTTOConfig, KTTOTrainer
from src.domain import AgeGroup, BehaviorStage, Gender, UserCharacteristics
from src.infrastructure.repository.effect_predictor_fake import EffectPredictorFake

if __name__ == "__main__":
    trainer = KTTOTrainer(
        base_model_path="elyza/Llama-3-ELYZA-JP-8B",
        base_messages=[
            "あなたは何歳ですか？",
            "あなたの性別を教えてください。",
            "あなたの行動ステージを教えてください。",
        ],
        effect_predictor=EffectPredictorFake(),
        characteristics_list=[
            UserCharacteristics(
                gender=Gender.FEMALE,
                age_group=AgeGroup.FORTIES_TO_FIFTIES,
                stage=BehaviorStage.ACTION_TO_MAINTENANCE,
            ),
        ],
        config=KTTOConfig(
            n_iter=2,
        ),
    )
    trainer.train()
