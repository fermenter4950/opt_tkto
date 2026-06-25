"""LLM の読み込みと generate 呼び出し。"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from transformers.generation.stopping_criteria import StopStringCriteria, StoppingCriteriaList

from src.infrastructure.llm.profile import LLMProfile, resolve_profile


@dataclass
class LLMSession:
    """読み込み済み LLM + tokenizer + プロファイル。"""

    profile: LLMProfile
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizer

    @property
    def device(self):
        return self.model.device

    def generate_from_text(
        self,
        prompt_text: str,
        *,
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.95,
        do_sample: bool = True,
        stop_strings: list[str] | None = None,
        length_penalty: float | None = None,
        num_beams: int = 1,
    ) -> str:
        """生テキストプロンプトから生成し、新規トークン部分の文字列を返す。"""
        inputs = self.tokenizer(prompt_text, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        use_beam = num_beams > 1 or (
            length_penalty is not None and length_penalty != 1.0
        )
        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if use_beam:
            generate_kwargs["num_beams"] = max(num_beams, 2)
            generate_kwargs["do_sample"] = False
            generate_kwargs["early_stopping"] = True
            if length_penalty is not None:
                generate_kwargs["length_penalty"] = length_penalty
        elif do_sample:
            generate_kwargs["do_sample"] = True
            generate_kwargs["temperature"] = temperature
            generate_kwargs["top_p"] = top_p
        else:
            generate_kwargs["do_sample"] = False
        if stop_strings:
            generate_kwargs["stopping_criteria"] = StoppingCriteriaList(
                [StopStringCriteria(self.tokenizer, stop_strings)]
            )

        self.model.eval()
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **generate_kwargs)

        prompt_len = inputs["input_ids"].shape[-1]
        return self.tokenizer.decode(
            output_ids[0, prompt_len:],
            skip_special_tokens=False,
        )

    def _encode_chat(
        self,
        messages: list[dict[str, str]],
        *,
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, torch.Tensor]:
        """chat template 結果を generate 用の tensor dict に正規化する。"""
        template_kwargs = chat_template_kwargs or {}
        encoded = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            **template_kwargs,
        )
        if isinstance(encoded, torch.Tensor):
            return {"input_ids": encoded.to(self.device)}
        return {key: value.to(self.device) for key, value in encoded.items()}

    def generate_from_chat(
        self,
        messages: list[dict[str, str]],
        *,
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        do_sample: bool = True,
        skip_special_tokens: bool = True,
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> str:
        """chat メッセージリストから生成し、assistant 部分の文字列を返す。"""
        inputs = self._encode_chat(messages, chat_template_kwargs=chat_template_kwargs)

        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.eos_token_id,
            "do_sample": do_sample,
        }
        if do_sample:
            generate_kwargs["temperature"] = temperature
        else:
            generate_kwargs["do_sample"] = False

        self.model.eval()
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **generate_kwargs)

        prompt_len = inputs["input_ids"].shape[-1]
        return self.tokenizer.decode(
            output_ids[0, prompt_len:],
            skip_special_tokens=skip_special_tokens,
        )


def load_llm(
    model_name: str,
    *,
    for_scoring: bool = False,
    reasoning_effort: str | None = None,
) -> LLMSession:
    """モデル名を指定して LLM を読み込む。

    for_scoring=True: 採点専用（推論なし chat モデルも可）。
    reasoning_effort: Harmony 用。指定時はプロファイル値を上書き。
    """
    profile = resolve_profile(model_name, for_scoring=for_scoring)
    if reasoning_effort is not None:
        profile = replace(profile, reasoning_effort=reasoning_effort)
    if not torch.cuda.is_available():
        raise ValueError("GPU is not available.")

    kwargs: dict[str, Any] = {}
    if profile.trust_remote_code:
        kwargs["trust_remote_code"] = True

    role = "採点" if for_scoring else "最適化"
    print(f"{role} LLM: {profile.name} ({profile.backend})", flush=True)
    print(f"パス/ID: {profile.model_id}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(profile.model_id, **kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # device_map だと .to(cuda:1) が効かず KTO 時に GPU0 に重みが残り OOM しやすい
    device = torch.device("cuda:1" if for_scoring else "cuda:0")
    model = AutoModelForCausalLM.from_pretrained(
        profile.model_id,
        dtype=torch.bfloat16,
        **kwargs,
    )
    model.to(device)
    print(f"{role} LLM の配置先: {device}", flush=True)
    return LLMSession(profile=profile, model=model, tokenizer=tokenizer)
