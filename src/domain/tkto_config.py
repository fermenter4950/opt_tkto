from dataclasses import dataclass


@dataclass
class TKTOConfig:
    # 1イテレーションあたりの学習に利用するプロンプトの数
    # 原稿では $n$ と表記されている
    batch_size: int

    # イテレーション回数
    # 原稿では $t$ と表記されている
    n_iter: int

    # 1つのプロンプトに対する出力の数
    # 原稿では $k$ と表記されている
    num_of_output: int

    # 中間生成物や最終生成物の保存先ディレクトリ
    output_dir: str

    # 1 TKTO step あたりの KTO optimizer step 数（上限。one_epoch=true 時はデータ量でさらに制限）
    kto_max_steps: int = 50

    # None のとき max_steps に応じて自動設定
    kto_save_steps: int | None = None

    kto_learning_rate: float = 5e-5
    kto_beta: float = 0.3
    kto_harmony_format: str = "split"
    kto_max_length: int = 4096
    # 形式 OK サンプルがこの割合未満なら KTO をスキップ（壊れた LoRA 更新を防ぐ）
    kto_min_valid_ratio: float = 0.0
    # true のとき max_steps を ceil(n_samples / batch_size) 以下に制限（過学習防止）
    kto_one_epoch: bool = True
