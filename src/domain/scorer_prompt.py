"""pos/neg スコアラー用の共通プロンプト（学習・推論で同一）。"""

from __future__ import annotations

SCORER_SYSTEM_PROMPT = (
    "あなたは健康促進メッセージを評価するアシスタントです。"
    "健康メッセージと受信者の特性（性別、年代、行動変容ステージ）が与えられた場合、"
    "その特性を持つ受信者がそのメッセージを読み健康改善に前向きになるかを "
    "'positive', 'negative' のいずれかで分類してください。"
)

SCORER_BASELINE_PROMPT_TEMPLATE = """\
### 健康促進メッセージ ###
'''
{message}
'''

### 受信者の特性 ###
性別: {sex}
年代: {age}
行動変容ステージ: {stage}

### タスク ###
上記の健康促進メッセージを，指定された受信者が読んだ場合，健康改善に前向きになるかどうかを判断し，以下のいずれかで回答してください:
- positive
- negative

### 注意事項 ###
- 回答は「positive」または「negative」のみとしてください.
- 説明や追加のコメントは不要です.

### 評価基準 ###
- positive: メッセージが受信者の特性に適しており，健康改善への動機付けになる可能性が高い
- negative: メッセージが受信者の特性に適していない，または健康改善への動機付けにならない可能性が高い

### 回答手順 ###
1. メッセージと受信者の特性を慎重に分析してください.
2. 評価基準に基づいて判断を行ってください.
3. 最終的な回答(positive または negative)のみを出力してください."""


def format_scorer_prompt(
    *,
    message: str,
    sex: str,
    age: str,
    stage: str,
) -> str:
    """ベースラインプロンプトにメッセージと受信者属性を埋め込む。"""
    return SCORER_BASELINE_PROMPT_TEMPLATE.format(
        message=str(message).strip(),
        sex=sex,
        age=age,
        stage=stage,
    )
