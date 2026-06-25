"""TKTO 実行設定（YAML）。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class OptimizerConfig:
    model: str = "llm-jp-4-8b-thinking"
    reasoning_effort: str = "low"
    max_new_tokens: int = 1024
    temperature: float = 0.0
    do_sample: bool = False
    # beam search 時のみ有効。> 1.0 で短い出力を優先（score /= length**penalty）
    length_penalty: float = 1.2
    num_beams: int = 4


@dataclass
class ScorerConfig:
    type: str = "llm"  # llm | bert
    model: str = "llm-jp-4-8b-thinking"
    model_path: str = "models/health_message_model.pth"
    adapter_path: str | None = None


@dataclass
class TktoRunConfig:
    n_iter: int = 5
    batch_size: int = 180
    num_of_output: int = 3
    initial_epoch: int = 0
    initial_step: int = 0
    peft_path: str | None = None


@dataclass
class KtoConfig:
    """1 TKTO step あたりの KTO（LoRA）学習設定。"""

    max_steps: int = 50
    save_steps: int | None = None
    learning_rate: float = 5e-5
    beta: float = 0.3
    harmony_format: str = "split"
    max_length: int = 4096
    min_valid_ratio: float = 0.0
    one_epoch: bool = True


@dataclass
class DataConfig:
    # .tsv（推奨: message.tsv）または .csv。拡張子で区切り文字を自動判定
    messages_csv: str = "data/message.tsv"
    message_column: str | None = "メッセージ"


@dataclass
class RunConfig:
    output_dir: str = "result"
    timestamp_dir: bool = True
    run_id: str | None = None
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scorer: ScorerConfig = field(default_factory=ScorerConfig)
    tkto: TktoRunConfig = field(default_factory=TktoRunConfig)
    kto: KtoConfig = field(default_factory=KtoConfig)
    data: DataConfig = field(default_factory=DataConfig)


def _merge(section: dict[str, Any] | None, cls: type):
    if not section:
        return cls()
    valid = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    return cls(**{k: v for k, v in section.items() if k in valid})


def load_run_config(path: str | Path) -> RunConfig:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"設定ファイルが見つかりません: {config_path}")

    with config_path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    run_section = raw.get("run", {})
    output_dir = run_section.get("output_dir", "result")
    run_id = run_section.get("run_id")
    timestamp_dir = run_section.get("timestamp_dir", True)

    return RunConfig(
        output_dir=str(output_dir),
        timestamp_dir=bool(timestamp_dir),
        run_id=str(run_id) if run_id is not None else None,
        optimizer=_merge(raw.get("optimizer"), OptimizerConfig),
        scorer=_merge(raw.get("scorer"), ScorerConfig),
        tkto=_merge(raw.get("tkto"), TktoRunConfig),
        kto=_merge(raw.get("kto"), KtoConfig),
        data=_merge(raw.get("data"), DataConfig),
    )


def resolve_model_output_dir(
    output_dir: str,
    model_name: str,
    *,
    timestamp_dir: bool = True,
    run_id: str | None = None,
) -> str:
    """モデル別 root の下に run ディレクトリを決める。

    run_id 指定時はそのディレクトリを再利用（再開用）。
    timestamp_dir=True かつ run_id 未指定時は YYYYMMDD_HHMMSS を付与。
    """
    from datetime import datetime

    base = Path(output_dir) / model_name
    if run_id:
        return str(base / run_id)
    if timestamp_dir:
        return str(base / datetime.now().strftime("%Y%m%d_%H%M%S"))
    return str(base)
