import os

import pandas as pd
import torch
import torch.nn as nn
import tqdm
from sklearn.model_selection import train_test_split
from transformers import BertModel, BertTokenizer

from src.application.interfaces.effect_predictor import EffectPredictor
from src.domain import UserCharacteristics
from src.domain.label import Label
from src.domain.user_characteristics.age_group import AgeGroup
from src.domain.user_characteristics.behavior_stage import BehaviorStage
from src.domain.user_characteristics.gender import Gender


class HealthMessageBERT(nn.Module):
    def __init__(self, bert_model_name):
        super(HealthMessageBERT, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(0.1)
        self.regressor = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size + 5, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, input_ids, attention_mask, sex, age, stage):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        dropped_bert = self.dropout(pooled_output)

        # Reshape the characteristic tensors to match dimensions
        sex = sex.unsqueeze(1) if sex.dim() == 1 else sex
        age = age.unsqueeze(0) if age.dim() == 1 else age
        stage = stage.unsqueeze(1) if stage.dim() == 1 else stage

        combined = torch.cat((dropped_bert, sex, age, stage), dim=1)
        return self.regressor(combined)


class EffectPredictorBERT(EffectPredictor):
    def __init__(self, model_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        bert_model = os.getenv(
            "BERT_BASE_MODEL", "cl-tohoku/bert-base-japanese-v3"
        )
        self.model = HealthMessageBERT(bert_model)

        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

    def _age_to_tensor(self, age_group):
        age_mapping = {
            "20代〜30代": torch.tensor([1, 0, 0], dtype=torch.float),
            "40代〜50代": torch.tensor([0, 1, 0], dtype=torch.float),
            "60代以上": torch.tensor([0, 0, 1], dtype=torch.float),
        }
        return age_mapping[age_group]

    def _stage_to_tensor(self, stage):
        stage_mapping = {
            "関心期〜準備期": torch.tensor([0], dtype=torch.float),
            "実行期〜維持期": torch.tensor([1], dtype=torch.float),
        }
        return stage_mapping[stage]

    def predict(
        self,
        message: str,
        characteristics: UserCharacteristics,
        threshold: float = 0.5,
    ) -> Label:
        self.model.eval()

        encoding = self.model.tokenizer(
            message,
            add_special_tokens=True,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        sex_tensor = torch.tensor(
            [0 if characteristics.gender.value.lower() == "male" else 1],
            dtype=torch.float,
        ).to(self.device)

        age_tensor = self._age_to_tensor(characteristics.age_group.value).to(
            self.device
        )
        stage_tensor = self._stage_to_tensor(characteristics.stage.value).to(
            self.device
        )

        with torch.no_grad():
            output = self.model(
                input_ids, attention_mask, sex_tensor, age_tensor, stage_tensor
            )
            prediction = output.squeeze().item()
        if prediction >= threshold:
            return Label.POSITIVE
        else:
            return Label.NEGATIVE


class CustomDataset(torch.utils.data.Dataset):
    def __init__(
        self, texts, genders, ages, stages, labels=None, tokenizer=None, max_length=512
    ):
        self.texts = texts
        self.genders = genders
        self.ages = ages
        self.stages = stages
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        item = {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "gender": torch.tensor(self.genders[idx], dtype=torch.float),
            "age": torch.tensor(self.ages[idx], dtype=torch.float),
            "stage": torch.tensor(self.stages[idx], dtype=torch.float),
        }

        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)

        return item


def train_model(
    model,
    train_loader,
    val_loader,
    num_epochs=10,
    learning_rate=2e-5,
    save_path="model.pth",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    best_val_loss = float("inf")

    for epoch in tqdm.tqdm(range(num_epochs)):
        model.train()
        total_loss = 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            gender = batch["gender"].to(device)
            age = batch["age"].to(device)
            stage = batch["stage"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask, gender, age, stage)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                gender = batch["gender"].to(device)
                age = batch["age"].to(device)
                stage = batch["stage"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids, attention_mask, gender, age, stage)
                val_loss += criterion(outputs.squeeze(), labels).item()

        current_val_loss = val_loss / len(val_loader)
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_loss": best_val_loss,
                },
                save_path,
            )
            print(f"Model saved at epoch {epoch+1}")

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Average training loss: {total_loss/len(train_loader):.4f}")
        print(f"Average validation loss: {current_val_loss:.4f}")


def main():
    df = pd.read_csv("data/message.csv")
    # データの準備
    messages = df["message"].tolist()
    genders = df["gender"].tolist()
    ages = pd.get_dummies(df["age_group"]).astype(int).values.tolist()
    stages = df["stage"].tolist()
    labels = df["score"].tolist()

    # データを訓練用とテスト用に分割
    (
        train_messages,
        val_messages,
        train_genders,
        val_genders,
        train_ages,
        val_ages,
        train_stages,
        val_stages,
        train_labels,
        val_labels,
    ) = train_test_split(
        messages, genders, ages, stages, labels, test_size=0.2, random_state=42
    )

    # モデルとトークナイザーの初期化
    model = HealthMessageBERT("cl-tohoku/bert-base-japanese-v3")

    # データセットとデータローダーの作成
    train_dataset = CustomDataset(
        texts=train_messages,
        genders=train_genders,
        ages=train_ages,
        stages=train_stages,
        labels=train_labels,
        tokenizer=model.tokenizer,
    )

    val_dataset = CustomDataset(
        texts=val_messages,
        genders=val_genders,
        ages=val_ages,
        stages=val_stages,
        labels=val_labels,
        tokenizer=model.tokenizer,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=2, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=2, shuffle=False)

    # モデルの学習
    train_model(
        model,
        train_loader,
        val_loader,
        num_epochs=10,
        save_path=os.getenv(
            "EFFECT_PREDICTOR_MODEL_PATH", "models/health_message_model.pth"
        ),
    )
    predictor = EffectPredictorBERT(
        os.getenv("EFFECT_PREDICTOR_MODEL_PATH", "models/health_message_model.pth")
    )
    prediction = predictor.predict(
        "運動は健康に良い影響を与えます",
        UserCharacteristics(
            gender=Gender.MALE,
            age_group=AgeGroup.FORTIES_TO_FIFTIES,
            stage=BehaviorStage.ACTION_TO_MAINTENANCE,
        ),
    )
    print(f"Prediction: {prediction}")


if __name__ == "__main__":
    main()
