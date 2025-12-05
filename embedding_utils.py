"""
埋め込み（Embedding）・類似度計算ユーティリティモジュール

このモジュールは、テキストをベクトル（数値の配列）に変換し、
ベクトル同士の類似度を計算する機能を提供します。

【仕組みの説明】
- 「埋め込み」とは、テキストを数百次元の数値ベクトルに変換することです
- 似た意味のテキストは似たベクトルになります
- ベクトル同士の「コサイン類似度」を計算することで、
  テキストの意味的な近さを数値化できます
"""

from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer

# 使用する埋め込みモデル
# paraphrase-multilingual-MiniLM-L12-v2 は多言語対応で、日本語もそこそこ扱えます
# モデルは初回実行時に自動でダウンロードされます
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

# モデルのキャッシュ（一度読み込んだら再利用）
_model_cache: Optional[SentenceTransformer] = None


def get_embedding_model() -> SentenceTransformer:
    """
    埋め込みモデルを取得する関数
    
    一度読み込んだモデルはキャッシュして再利用します。
    これにより、2回目以降の呼び出しが高速になります。
    
    Returns:
        SentenceTransformerモデル
    """
    global _model_cache
    
    if _model_cache is None:
        print(f"埋め込みモデル '{MODEL_NAME}' を読み込み中...")
        _model_cache = SentenceTransformer(MODEL_NAME)
        print("モデルの読み込みが完了しました。")
    
    return _model_cache


def create_embedding(text: str) -> np.ndarray:
    """
    テキストを埋め込みベクトルに変換する関数
    
    Args:
        text: ベクトル化したいテキスト
    
    Returns:
        テキストの埋め込みベクトル（numpy配列）
    """
    model = get_embedding_model()
    embedding = model.encode(text, convert_to_numpy=True)
    return embedding


def create_embeddings_batch(texts: List[str]) -> List[np.ndarray]:
    """
    複数のテキストをまとめて埋め込みベクトルに変換する関数
    
    一度に複数のテキストを処理するため、
    1つずつ処理するより効率的です。
    
    Args:
        texts: ベクトル化したいテキストのリスト
    
    Returns:
        埋め込みベクトルのリスト
    """
    model = get_embedding_model()
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    return list(embeddings)


def calculate_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    2つのベクトル間のコサイン類似度を計算する関数
    
    コサイン類似度は -1 から 1 の値を取り、
    1に近いほど似ている、-1に近いほど反対の意味を持ちます。
    
    Args:
        vec1: 1つ目のベクトル
        vec2: 2つ目のベクトル
    
    Returns:
        コサイン類似度（-1.0 〜 1.0）
    """
    # ベクトルのノルム（長さ）を計算
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    # ゼロ除算を防ぐ
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    # コサイン類似度 = (ベクトルの内積) / (ノルムの積)
    similarity = np.dot(vec1, vec2) / (norm1 * norm2)
    
    return float(similarity)


def search_similar_chunks(
    query: str,
    chunks: List[Dict[str, Any]],
    top_k: int = 5,
    threshold: float = 0.3
) -> List[Dict[str, Any]]:
    """
    質問文に類似したチャンクを検索する関数
    
    処理の流れ：
    1. 質問文を埋め込みベクトルに変換
    2. 各チャンクの埋め込みベクトルとコサイン類似度を計算
    3. 類似度が高い上位K件を返す
    
    Args:
        query: 質問文
        chunks: チャンク情報のリスト（各チャンクはembeddingを持つ）
        top_k: 返す結果の最大件数（デフォルト: 5件）
        threshold: 類似度の閾値（これより低いものは除外、デフォルト: 0.3）
    
    Returns:
        類似度の高いチャンクのリスト（類似度スコア付き）
    """
    # 質問文を埋め込みベクトルに変換
    query_embedding = create_embedding(query)
    
    # 各チャンクとの類似度を計算
    results = []
    for chunk in chunks:
        if chunk["embedding"] is None:
            continue
        
        similarity = calculate_cosine_similarity(query_embedding, chunk["embedding"])
        
        # 閾値以上のもののみ追加
        if similarity >= threshold:
            result = {
                "pdf_name": chunk["pdf_name"],
                "page_number": chunk["page_number"],
                "text": chunk["text"],
                "similarity": similarity
            }
            results.append(result)
    
    # 類似度の高い順にソート
    results.sort(key=lambda x: x["similarity"], reverse=True)
    
    # 上位K件を返す
    return results[:top_k]


def add_embeddings_to_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    チャンクリストに埋め込みベクトルを追加する関数
    
    Args:
        chunks: チャンク情報のリスト
    
    Returns:
        埋め込みベクトルが追加されたチャンクリスト
    """
    # チャンクのテキストを抽出
    texts = [chunk["text"] for chunk in chunks]
    
    # バッチ処理で埋め込みを生成
    embeddings = create_embeddings_batch(texts)
    
    # 各チャンクに埋め込みを設定
    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding
    
    return chunks


# ===== オプション: LLM を使った回答生成（モック） =====
# 以下はLLMを使って自然な回答を生成する場合のモック関数です。
# 実際に使用する場合は、OpenAI APIなどを呼び出すように実装してください。

def generate_answer_with_llm(query: str, context_chunks: List[Dict[str, Any]]) -> str:
    """
    【モック関数】LLMを使って質問に対する回答を生成する
    
    注意: これはモック実装です。
    実際に使用する場合は、以下のように実装してください：
    
    1. 環境変数から API キーを取得
       import os
       api_key = os.getenv("OPENAI_API_KEY")
    
    2. OpenAI API などを呼び出し
       from openai import OpenAI
       client = OpenAI(api_key=api_key)
       response = client.chat.completions.create(...)
    
    Args:
        query: ユーザーの質問
        context_chunks: 関連するチャンクのリスト
    
    Returns:
        生成された回答文（モックでは固定メッセージ）
    """
    # モック実装: 実際のLLM呼び出しの代わりに固定メッセージを返す
    return (
        "【この機能は未実装です】\n"
        "LLMを使った自然な回答生成を行うには、\n"
        "OpenAI APIなどの設定が必要です。\n"
        "現在は、関連するテキストの抜粋のみを表示しています。"
    )

