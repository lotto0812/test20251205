"""
PDF処理ユーティリティモジュール

このモジュールは、PDFファイルからテキストを抽出し、
チャンク（小さな文章の塊）に分割する機能を提供します。
"""

from typing import List, Dict, Any
import pdfplumber


def extract_text_from_pdf(pdf_file) -> List[Dict[str, Any]]:
    """
    PDFファイルから各ページのテキストを抽出する関数
    
    Args:
        pdf_file: Streamlitでアップロードされたファイルオブジェクト
    
    Returns:
        各ページの情報を含む辞書のリスト
        [{"page_number": 1, "text": "ページのテキスト..."}, ...]
    """
    pages_data = []
    
    try:
        # pdfplumberでPDFを開く
        with pdfplumber.open(pdf_file) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                # ページからテキストを抽出
                text = page.extract_text()
                
                # テキストが取れた場合のみ追加
                if text and text.strip():
                    pages_data.append({
                        "page_number": page_num,
                        "text": text.strip()
                    })
    except Exception as e:
        raise Exception(f"PDFの読み込みに失敗しました: {str(e)}")
    
    return pages_data


def chunk_text(text: str, max_chars: int = 800, overlap: int = 100) -> List[str]:
    """
    長いテキストを小さなチャンク（塊）に分割する関数
    
    文章を一定の文字数ごとに分割します。
    オーバーラップを設けることで、文の途中で切れた場合も
    前後のチャンクで補完できるようにしています。
    
    Args:
        text: 分割したいテキスト
        max_chars: 1チャンクの最大文字数（デフォルト: 800文字）
        overlap: チャンク間で重複させる文字数（デフォルト: 100文字）
    
    Returns:
        分割されたテキストのリスト
    """
    # テキストが短い場合はそのまま返す
    if len(text) <= max_chars:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        # 終了位置を計算
        end = start + max_chars
        
        # テキストの末尾を超えないように調整
        if end >= len(text):
            chunks.append(text[start:])
            break
        
        # なるべく文の区切りで分割する（。や\nを探す）
        # 区切り文字を後ろから探して、見つかればそこで切る
        split_pos = end
        for delimiter in ["。", "\n", "、", " "]:
            pos = text.rfind(delimiter, start, end)
            if pos > start:
                split_pos = pos + 1
                break
        
        chunks.append(text[start:split_pos])
        
        # 次の開始位置（オーバーラップを考慮）
        start = split_pos - overlap if split_pos - overlap > start else split_pos
    
    return chunks


def process_pdf(pdf_file, pdf_name: str, max_chunk_chars: int = 800) -> List[Dict[str, Any]]:
    """
    PDFファイルを処理して、チャンク情報のリストを返す関数
    
    この関数は以下の処理を行います：
    1. PDFから各ページのテキストを抽出
    2. 各ページのテキストをチャンクに分割
    3. 各チャンクにPDF名とページ番号の情報を付与
    
    Args:
        pdf_file: Streamlitでアップロードされたファイルオブジェクト
        pdf_name: PDFのファイル名
        max_chunk_chars: 1チャンクの最大文字数
    
    Returns:
        チャンク情報のリスト。各チャンクは以下の情報を持つ：
        - pdf_name: PDFファイル名
        - page_number: ページ番号
        - text: チャンクのテキスト
        - embedding: 埋め込みベクトル（後で設定される、初期値はNone）
    """
    # PDFからページごとのテキストを抽出
    pages_data = extract_text_from_pdf(pdf_file)
    
    if not pages_data:
        raise Exception(f"{pdf_name} からテキストを抽出できませんでした。")
    
    all_chunks = []
    
    # 各ページのテキストをチャンクに分割
    for page_info in pages_data:
        page_number = page_info["page_number"]
        page_text = page_info["text"]
        
        # テキストをチャンクに分割
        text_chunks = chunk_text(page_text, max_chars=max_chunk_chars)
        
        # 各チャンクに情報を付与
        for chunk_text_content in text_chunks:
            chunk_info = {
                "pdf_name": pdf_name,
                "page_number": page_number,
                "text": chunk_text_content,
                "embedding": None  # 後で埋め込みベクトルを設定
            }
            all_chunks.append(chunk_info)
    
    return all_chunks

