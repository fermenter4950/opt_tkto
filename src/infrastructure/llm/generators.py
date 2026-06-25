"""推論付きモデル向け completion 生成。"""

from __future__ import annotations

from src.application.interfaces.completion_generator import CompletionGenerator, KtoFormat
from src.domain.llmjp_completion import LlmJpCompletion
from src.infrastructure.harmony import (
    HARMONY_STOP_STRINGS,
    build_assistant_raw as build_harmony_assistant_raw,
    build_prompt_text as build_harmony_prompt_text,
    count_text_tokens,
    kto_quality_notes as harmony_kto_quality_notes,
    kto_validation_issues as harmony_kto_validation_issues,
    parse_harmony_output,
    preprocess_harmony_for_kto,
)
from src.infrastructure.llm.client import LLMSession
from src.infrastructure.llm.profile import Backend
from src.infrastructure import qwen_thinking


class HarmonyCompletionGenerator(CompletionGenerator):
    def __init__(
        self,
        session: LLMSession,
        max_new_tokens: int = 1024,
        temperature: float = 0.0,
        do_sample: bool = False,
        max_attempts: int = 3,
        length_penalty: float | None = 1.2,
        num_beams: int = 4,
    ):
        self.session = session
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.do_sample = do_sample
        self.max_attempts = max_attempts
        self.length_penalty = length_penalty
        self.num_beams = num_beams
        self.reasoning_effort = session.profile.reasoning_effort

    def build_kto_prompt(self, instruction: str) -> str:
        return build_harmony_prompt_text(
            self.session.tokenizer,
            instruction,
            reasoning_effort=self.reasoning_effort,
        )

    def preprocess_for_kto(
        self,
        base_prompt: str,
        thinking: str,
        final: str,
        *,
        mode: KtoFormat = "split",
    ) -> dict[str, str]:
        return preprocess_harmony_for_kto(
            base_prompt,
            thinking,
            final,
            mode=mode,  # type: ignore[arg-type]
        )

    def kto_validation_issues(self, thought: str, response: str) -> list[str]:
        return harmony_kto_validation_issues(thought, response)

    def kto_quality_notes(self, thought: str, response: str) -> list[str]:
        return harmony_kto_quality_notes(thought, response)

    def _make_completion(
        self,
        analysis: str,
        final: str,
        *,
        raw: str,
        generated: str,
        token_count: int,
    ) -> LlmJpCompletion:
        tokenizer = self.session.tokenizer
        return LlmJpCompletion(
            analysis,
            final,
            raw=raw,
            generated=generated,
            generated_tokens=token_count,
            thinking_tokens=count_text_tokens(tokenizer, analysis),
            final_tokens=count_text_tokens(tokenizer, final),
        )

    def generate(self, instruction: str) -> LlmJpCompletion:
        """Harmony 生成。失敗しても例外にせず、可能な範囲で出力して続行する。"""
        prompt_text = self.build_kto_prompt(instruction)
        best: LlmJpCompletion | None = None

        for attempt in range(self.max_attempts):
            try:
                generated = self.session.generate_from_text(
                    prompt_text,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    do_sample=self.do_sample,
                    stop_strings=list(HARMONY_STOP_STRINGS),
                    length_penalty=self.length_penalty,
                    num_beams=self.num_beams,
                )
            except Exception as exc:
                print(
                    f"警告: 生成エラー ({attempt + 1}/{self.max_attempts}): {exc}",
                    flush=True,
                )
                continue

            token_count = count_text_tokens(self.session.tokenizer, generated)
            raw = build_harmony_assistant_raw(prompt_text, generated)
            analysis, final = parse_harmony_output(raw)
            if not final.strip() and generated.strip():
                final = generated.strip()

            candidate = self._make_completion(
                analysis,
                final,
                raw=raw,
                generated=generated,
                token_count=token_count,
            )
            best = candidate
            if final.strip():
                return candidate

        if best is not None:
            print(
                "警告: final が空のため最善出力を採用して続行します。",
                flush=True,
            )
            return best

        print("警告: 生成に失敗したため空の completion で続行します。", flush=True)
        return LlmJpCompletion("", "", raw="", generated="")


class QwenThinkingCompletionGenerator(CompletionGenerator):
    """Qwen3 / Shisa V2.1 系の thinking + final 生成。"""

    def __init__(
        self,
        session: LLMSession,
        max_new_tokens: int = 1024,
        temperature: float = 0.0,
        do_sample: bool = False,
        max_attempts: int = 3,
        length_penalty: float | None = 1.2,
        num_beams: int = 4,
    ):
        self.session = session
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.do_sample = do_sample
        self.max_attempts = max_attempts
        self.length_penalty = length_penalty
        self.num_beams = num_beams
        self.im_end = qwen_thinking._im_end_token(session.tokenizer)

    def build_kto_prompt(self, instruction: str) -> str:
        return qwen_thinking.build_prompt_text(self.session.tokenizer, instruction)

    def preprocess_for_kto(
        self,
        base_prompt: str,
        thinking: str,
        final: str,
        *,
        mode: KtoFormat = "split",
    ) -> dict[str, str]:
        return qwen_thinking.preprocess_qwen_for_kto(
            base_prompt,
            thinking,
            final,
            im_end=self.im_end,
            mode=mode,  # type: ignore[arg-type]
        )

    def kto_validation_issues(self, thought: str, response: str) -> list[str]:
        return qwen_thinking.kto_validation_issues(thought, response)

    def kto_quality_notes(self, thought: str, response: str) -> list[str]:
        return qwen_thinking.kto_quality_notes(thought, response)

    def _stop_strings(self) -> list[str]:
        return [self.im_end]

    def _make_completion(
        self,
        thinking: str,
        final: str,
        *,
        raw: str,
        generated: str,
        token_count: int,
    ) -> LlmJpCompletion:
        tokenizer = self.session.tokenizer
        return LlmJpCompletion(
            thinking,
            final,
            raw=raw,
            generated=generated,
            generated_tokens=token_count,
            thinking_tokens=count_text_tokens(tokenizer, thinking),
            final_tokens=count_text_tokens(tokenizer, final),
        )

    def generate(self, instruction: str) -> LlmJpCompletion:
        prompt_text = self.build_kto_prompt(instruction)
        best: LlmJpCompletion | None = None

        for attempt in range(self.max_attempts):
            try:
                generated = self.session.generate_from_text(
                    prompt_text,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    do_sample=self.do_sample,
                    stop_strings=self._stop_strings(),
                    length_penalty=self.length_penalty,
                    num_beams=self.num_beams,
                )
            except Exception as exc:
                print(
                    f"警告: 生成エラー ({attempt + 1}/{self.max_attempts}): {exc}",
                    flush=True,
                )
                continue

            if not qwen_thinking.has_answer_marker(generated):
                print(
                    f"警告: 【回答】なし ({attempt + 1}/{self.max_attempts})、再生成します。",
                    flush=True,
                )
                continue

            token_count = count_text_tokens(self.session.tokenizer, generated)
            raw = qwen_thinking.build_assistant_raw(prompt_text, generated)
            thinking, final = qwen_thinking.parse_qwen_thinking_output(
                raw, im_end=self.im_end
            )

            candidate = self._make_completion(
                thinking,
                final,
                raw=raw,
                generated=generated,
                token_count=token_count,
            )
            best = candidate
            if thinking.strip() and final.strip():
                return candidate

        if best is not None:
            print(
                "警告: final が空のため最善出力を採用して続行します。",
                flush=True,
            )
            return best

        print("警告: 生成に失敗したため空の completion で続行します。", flush=True)
        return LlmJpCompletion("", "", raw="", generated="")


def create_completion_generator(
    session: LLMSession,
    *,
    max_new_tokens: int | None = None,
    temperature: float = 0.0,
    do_sample: bool = False,
    length_penalty: float | None = 1.2,
    num_beams: int = 4,
) -> CompletionGenerator:
    backend: Backend = session.profile.backend
    common = dict(
        max_new_tokens=max_new_tokens or 1024,
        temperature=temperature,
        do_sample=do_sample,
        length_penalty=length_penalty,
        num_beams=num_beams,
    )
    if backend == "harmony":
        return HarmonyCompletionGenerator(session, **common)
    if backend == "qwen_thinking":
        return QwenThinkingCompletionGenerator(session, **common)
    raise ValueError(f"未対応バックエンド: {backend}")
