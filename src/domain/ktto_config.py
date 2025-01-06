class KTTOConfig:
    """
    設定値を表すValueObjectとしてdomain層に移動
    """

    def __init__(self, n_iter: int = 1, k: int = 5):
        self.n_iter = n_iter
        self.k = k

    def validate(self):
        """ビジネスルールのバリデーション"""
        if self.n_iter < 1:
            raise ValueError("n_iter must be greater than 0")
        if self.k < 1:
            raise ValueError("k must be greater than 0")


class TrainingResult:
    """
    学習結果を表すValueObjectとしてdomain層に移動
    """

    def __init__(self, iteration: int, output_dir: str):
        self.iteration = iteration
        self.output_dir = output_dir
