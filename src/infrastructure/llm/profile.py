"""LLM プロファイル定義とモデル名解決（推論付きモデルのみ）。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

# 推論付きバックエンド: harmony（LLM-jp）/ qwen_thinking（Shisa V2.1 等）
Backend = Literal["harmony", "qwen_thinking"]


@dataclass(frozen=True)
class LLMProfile:
    name: str
    model_id: str
    backend: Backend
    trust_remote_code: bool = False
    reasoning_effort: str = "low"


def _elyza_model_id() -> str:
    path = os.getenv("BASE_MODEL_PATH", "/data/LLM/Llama-3-ELYZA-JP-8B")
    if os.path.isdir(path):
        return path
    if "/" in path and not path.startswith(("/", ".", "~")):
        return path
    return os.getenv("BASE_MODEL_HF_ID", "elyza/Llama-3-ELYZA-JP-8B")


def _elyza_scorer_profile(name: str) -> LLMProfile:
    return LLMProfile(
        name=name,
        model_id=_elyza_model_id(),
        backend="harmony",
        trust_remote_code=False,
    )


def _llmjp_model_id() -> str:
    path = os.getenv("LLMJP_MODEL_PATH", "llm-jp/llm-jp-4-8b-thinking")
    if os.path.isdir(path):
        return path
    if "/" in path and not path.startswith(("/", ".", "~")):
        return path
    return os.getenv("LLMJP_MODEL_HF_ID", "llm-jp/llm-jp-4-8b-thinking")


def _llmjp_profile(name: str) -> LLMProfile:
    return LLMProfile(
        name=name,
        model_id=_llmjp_model_id(),
        backend="harmony",
        trust_remote_code=True,
        reasoning_effort=os.getenv("LLMJP_REASONING_EFFORT", "low"),
    )


def _shisa_model_id() -> str:
    path = os.getenv("SHISA_MODEL_PATH", "shisa-ai/shisa-v2.1-qwen3-8b")
    if os.path.isdir(path):
        return path
    if "/" in path and not path.startswith(("/", ".", "~")):
        return path
    return os.getenv("SHISA_MODEL_HF_ID", "shisa-ai/shisa-v2.1-qwen3-8b")


def _shisa_profile(name: str) -> LLMProfile:
    return LLMProfile(
        name=name,
        model_id=_shisa_model_id(),
        backend="qwen_thinking",
        trust_remote_code=True,
    )


# モデル名（CLI / MODEL_NAME）→ プロファイル
# 推論なしモデル（Llama3-Elyza 等）は最適化 LLM として登録しない
MODEL_REGISTRY: dict[str, LLMProfile] = {
    "llm-jp-4-8b-thinking": _llmjp_profile("llm-jp-4-8b-thinking"),
    "llm-jp": _llmjp_profile("llm-jp"),
    "shisa-v2.1-qwen3-8b": _shisa_profile("shisa-v2.1-qwen3-8b"),
    "shisa-v2.1": _shisa_profile("shisa-v2.1"),
}

# 採点専用（推論なし chat モデル可）。最適化 LLM には登録しない
SCORER_REGISTRY: dict[str, LLMProfile] = {
    "elyza": _elyza_scorer_profile("elyza"),
    "llama-3-elyza-jp-8b": _elyza_scorer_profile("llama-3-elyza-jp-8b"),
}


def _is_reasoning_model(model_id: str) -> bool:
    lower = model_id.lower()
    return (
        "thinking" in lower
        or "llm-jp" in lower
        or "shisa" in lower
        or "qwen3" in lower
    )


def resolve_profile(model_name: str, *, for_scoring: bool = False) -> LLMProfile:
    """モデル名・パス・HF ID から LLMProfile を解決する。

    for_scoring=True のときは採点専用: 推論付きでなくても任意の chat モデルを許可。
    """
    key = model_name.strip()

    if for_scoring and key in SCORER_REGISTRY:
        registered = SCORER_REGISTRY[key]
        if key in {"elyza", "llama-3-elyza-jp-8b"}:
            return _elyza_scorer_profile(registered.name)
        return registered

    if key in MODEL_REGISTRY:
        registered = MODEL_REGISTRY[key]
        if registered.backend == "harmony" and key in {"llm-jp", "llm-jp-4-8b-thinking"}:
            return _llmjp_profile(registered.name)
        if registered.backend == "qwen_thinking" and key in {
            "shisa-v2.1-qwen3-8b",
            "shisa-v2.1",
        }:
            return _shisa_profile(registered.name)
        return registered

    if os.path.isdir(key) or (
        "/" in key and not key.startswith(("/", ".", "~"))
    ):
        if not for_scoring and not _is_reasoning_model(key):
            raise ValueError(
                f"最適化 LLM は推論付きモデルのみ対応: {model_name!r}。"
                f"登録済み: {', '.join(list_model_names())}"
            )
        basename = (
            os.path.basename(key.rstrip("/")) if os.path.isdir(key) else key.split("/")[-1]
        )
        trust = _is_reasoning_model(key) or "llm-jp" in key.lower()
        lower = key.lower()
        if "shisa" in lower or "qwen3" in lower:
            backend: Backend = "qwen_thinking"
        else:
            backend = "harmony"
        return LLMProfile(
            name=basename,
            model_id=key,
            backend=backend,
            trust_remote_code=trust,
            reasoning_effort=os.getenv("LLMJP_REASONING_EFFORT", "low"),
        )

    registries = (
        {**MODEL_REGISTRY, **SCORER_REGISTRY} if for_scoring else MODEL_REGISTRY
    )
    available = ", ".join(sorted(registries))
    role = "採点" if for_scoring else "最適化"
    raise ValueError(
        f"未知の{role}モデル名: {model_name!r}。"
        f"登録済み: {available}。"
        f"または HF ID / ローカルパスを直接指定できます。"
    )


def list_model_names() -> list[str]:
    return sorted(MODEL_REGISTRY.keys())


def list_scorer_model_names() -> list[str]:
    return sorted({**MODEL_REGISTRY, **SCORER_REGISTRY}.keys())
