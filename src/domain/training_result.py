class TrainingResult:
    """
    学習結果を表すValueObjectとしてdomain層に移動
    """

    def __init__(self, iteration: int, output_dir: str):
        self.iteration = iteration
        self.output_dir = output_dir
