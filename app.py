"""
PDF QA PoC（複数PDF対応）- Streamlit アプリケーション

このアプリケーションは、複数のPDFファイルをアップロードし、
その内容に基づいて質問に答えるPoCツールです。

【使い方】
1. 左サイドバーでPDFファイルをアップロード
2. 「インデックス作成」ボタンをクリック
3. 質問を入力して「検索」ボタンをクリック

【起動方法】
streamlit run app.py
"""

import streamlit as st
from typing import List, Dict, Any

# 自作モジュールをインポート
from pdf_utils import process_pdf
from embedding_utils import add_embeddings_to_chunks, search_similar_chunks


# ===== ページ設定 =====
st.set_page_config(
    page_title="PDF QA PoC",
    page_icon="📚",
    layout="wide"
)


# ===== セッション状態の初期化 =====
# session_state を使って、ページ更新後もデータを保持します
def initialize_session_state():
    """セッション状態を初期化する関数"""
    if "chunks" not in st.session_state:
        st.session_state.chunks = []  # チャンクのリスト
    if "index_created" not in st.session_state:
        st.session_state.index_created = False  # インデックス作成済みフラグ
    if "uploaded_file_names" not in st.session_state:
        st.session_state.uploaded_file_names = []  # アップロード済みファイル名


initialize_session_state()


# ===== サイドバー: PDFアップロードとインデックス作成 =====
def render_sidebar():
    """サイドバーを描画する関数"""
    
    with st.sidebar:
        st.header("📁 PDFアップロード")
        st.markdown("複数のPDFファイルをアップロードできます。")
        
        # ファイルアップローダー（複数ファイル対応）
        uploaded_files = st.file_uploader(
            "PDFファイルを選択",
            type=["pdf"],
            accept_multiple_files=True,
            help="複数のPDFファイルを選択できます"
        )
        
        # アップロードされたファイル数を表示
        if uploaded_files:
            st.info(f"📄 {len(uploaded_files)} 個のファイルが選択されています")
            for file in uploaded_files:
                st.write(f"- {file.name}")
        
        st.markdown("---")
        
        # インデックス作成ボタン
        if st.button("🔨 インデックス作成", type="primary", use_container_width=True):
            create_index(uploaded_files)
        
        # インデックス状態の表示
        if st.session_state.index_created:
            st.success("✅ インデックス作成済み")
            st.write(f"チャンク数: {len(st.session_state.chunks)}")
            st.write("対象ファイル:")
            for name in st.session_state.uploaded_file_names:
                st.write(f"- {name}")
        
        st.markdown("---")
        
        # インデックスクリアボタン
        if st.session_state.index_created:
            if st.button("🗑️ インデックスをクリア", use_container_width=True):
                clear_index()


def create_index(uploaded_files) -> None:
    """
    アップロードされたPDFからインデックスを作成する関数
    
    処理の流れ：
    1. PDFファイルの存在チェック
    2. 各PDFからテキスト抽出・チャンク化
    3. 全チャンクの埋め込みベクトルを生成
    4. session_state に保存
    """
    # ファイルが選択されているかチェック
    if not uploaded_files:
        st.sidebar.error("❌ PDFファイルをアップロードしてください")
        return
    
    try:
        all_chunks = []
        file_names = []
        
        # 進捗表示用
        progress_bar = st.sidebar.progress(0)
        status_text = st.sidebar.empty()
        
        total_files = len(uploaded_files)
        
        # 各PDFを処理
        for i, pdf_file in enumerate(uploaded_files):
            status_text.text(f"📖 処理中: {pdf_file.name}")
            
            try:
                # PDFからチャンクを抽出
                chunks = process_pdf(pdf_file, pdf_file.name)
                all_chunks.extend(chunks)
                file_names.append(pdf_file.name)
                
            except Exception as e:
                st.sidebar.warning(f"⚠️ {pdf_file.name} の処理でエラー: {str(e)}")
                continue
            
            # 進捗更新
            progress_bar.progress((i + 1) / total_files * 0.5)  # 前半50%
        
        # チャンクが取得できたかチェック
        if not all_chunks:
            st.sidebar.error("❌ PDFからテキストを抽出できませんでした")
            progress_bar.empty()
            status_text.empty()
            return
        
        # 埋め込みベクトルを生成
        status_text.text("🧮 埋め込みベクトルを生成中...")
        
        with st.spinner("埋め込みモデルを読み込み中...（初回は時間がかかります）"):
            all_chunks = add_embeddings_to_chunks(all_chunks)
        
        progress_bar.progress(1.0)  # 完了
        
        # session_state に保存
        st.session_state.chunks = all_chunks
        st.session_state.index_created = True
        st.session_state.uploaded_file_names = file_names
        
        status_text.text("✅ 完了！")
        st.sidebar.success(f"✅ インデックス作成完了！({len(all_chunks)} チャンク)")
        
        # 少し待ってから表示をクリア
        import time
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
        
        # 画面を更新
        st.rerun()
        
    except Exception as e:
        st.sidebar.error(f"❌ エラーが発生しました: {str(e)}")


def clear_index() -> None:
    """インデックスをクリアする関数"""
    st.session_state.chunks = []
    st.session_state.index_created = False
    st.session_state.uploaded_file_names = []
    st.rerun()


# ===== メインエリア: タイトルと説明 =====
def render_header():
    """ヘッダー部分を描画する関数"""
    st.title("📚 PDF QA PoC（複数PDF対応）")
    st.markdown("""
    **複数のPDFファイルの内容に基づいて質問できるツールです。**
    
    1. 左のサイドバーでPDFファイルをアップロード
    2. 「インデックス作成」ボタンをクリック
    3. 下の入力欄で質問を入力して検索
    
    ---
    """)


# ===== メインエリア: 質問入力と検索結果 =====
def render_search_area():
    """質問入力と検索結果エリアを描画する関数"""
    
    # インデックスが作成されていない場合
    if not st.session_state.index_created:
        st.info("👆 まず、左のサイドバーでPDFをアップロードし、インデックスを作成してください。")
        return
    
    st.subheader("🔍 質問を入力")
    
    # 質問入力フォーム
    with st.form(key="search_form"):
        query = st.text_input(
            "質問を入力してください",
            placeholder="例: この文書の主なポイントは何ですか？",
            help="日本語で質問できます"
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            top_k = st.selectbox("表示件数", options=[3, 5, 10], index=1)
        with col2:
            threshold = st.slider(
                "類似度の閾値",
                min_value=0.0,
                max_value=1.0,
                value=0.3,
                step=0.05,
                help="この値より低い類似度の結果は表示されません"
            )
        
        search_button = st.form_submit_button("🔍 検索", type="primary")
    
    # 検索実行
    if search_button:
        if not query.strip():
            st.warning("⚠️ 質問を入力してください")
            return
        
        perform_search(query, top_k, threshold)


def perform_search(query: str, top_k: int, threshold: float) -> None:
    """
    検索を実行して結果を表示する関数
    
    Args:
        query: 質問文
        top_k: 表示する結果の最大件数
        threshold: 類似度の閾値
    """
    with st.spinner("🔍 検索中..."):
        try:
            # 類似チャンクを検索
            results = search_similar_chunks(
                query=query,
                chunks=st.session_state.chunks,
                top_k=top_k,
                threshold=threshold
            )
            
            # 結果を表示
            display_search_results(query, results)
            
        except Exception as e:
            st.error(f"❌ 検索中にエラーが発生しました: {str(e)}")


def display_search_results(query: str, results: List[Dict[str, Any]]) -> None:
    """
    検索結果を表示する関数
    
    Args:
        query: 質問文
        results: 検索結果のリスト
    """
    st.markdown("---")
    st.subheader("📋 検索結果")
    
    # 結果がない場合
    if not results:
        st.warning("😕 該当する情報が見つかりませんでした。別の質問を試してみてください。")
        return
    
    st.write(f"**質問:** {query}")
    st.write(f"**{len(results)} 件の関連情報が見つかりました**")
    
    # 各結果を表示
    for i, result in enumerate(results, start=1):
        with st.expander(
            f"📄 {i}. {result['pdf_name']} - {result['page_number']}ページ "
            f"(類似度: {result['similarity']:.2%})",
            expanded=(i <= 3)  # 上位3件は展開表示
        ):
            # メタ情報
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("ファイル名", result["pdf_name"])
            with col2:
                st.metric("ページ番号", f"{result['page_number']} ページ")
            with col3:
                st.metric("類似度スコア", f"{result['similarity']:.2%}")
            
            # テキスト内容
            st.markdown("**関連テキスト:**")
            
            # テキストを見やすく表示（長い場合は省略）
            text = result["text"]
            if len(text) > 1000:
                st.text_area(
                    "テキスト内容",
                    value=text,
                    height=200,
                    disabled=True,
                    label_visibility="collapsed"
                )
            else:
                st.markdown(f"```\n{text}\n```")


# ===== フッター =====
def render_footer():
    """フッターを描画する関数"""
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: gray; font-size: 0.8em;">
        PDF QA PoC - Streamlit Application<br>
        Powered by sentence-transformers & pdfplumber
    </div>
    """, unsafe_allow_html=True)


# ===== メイン処理 =====
def main():
    """アプリケーションのメイン関数"""
    render_sidebar()
    render_header()
    render_search_area()
    render_footer()


if __name__ == "__main__":
    main()

