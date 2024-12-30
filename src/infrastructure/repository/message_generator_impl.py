from typing import List

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.application.interfaces.message_generator import MessageGenerator
from src.domain import Completion, Prompt


class MessageGeneratorImpl(MessageGenerator):
    def self(self, base_model_path: str, peft_path: str, use_4bit: bool = True):
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path, torch_dtype=torch.bfloat16, device_map={"": 0}
        ).to(self.device)

        self.model = PeftModel.from_pretrained(
            base_model,
            peft_path,
        )

    def generate(self, prompt: Prompt, k: int) -> List[Completion]:
        chat = [{"role": "user", "content": prompt}]
        input_ids = self.tokenizer.apply_chat_template(
            chat,
            return_tensors="pt",
        ).to(self.device)

        completions: Completion = []
        for _ in range(k):
            output_ids = self.model.generate(
                input_ids=input_ids,
                do_sample=True,
                temperature=0.7,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            content = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            assistant = content.split("assistant")[1].strip()
            completion = Completion(content=assistant)
            completions.append(completion)

        return completions
