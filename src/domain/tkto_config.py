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
