"""
YouTube AI 助手

這是一個 Streamlit 應用程序，可以解析 YouTube 視頻字幕，生成內容摘要，
提供字幕翻譯，並支持與視頻內容的智能對話互動。
"""

import logging
import os
import time
import traceback

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv
from tenacity import (retry, retry_if_exception_type, stop_after_attempt,
                      wait_random_exponential)

# 載入環境變數
load_dotenv()

# 設置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from modules.chat_agent import (chat_with_video, create_chat_agent,
                                generate_thread_id)
from modules.summarizer import (generate_advanced_summary,
                                generate_video_summary)
from modules.text_processor import (load_vector_store,
                                    split_and_vectorize_transcript)
from modules.translator import translate_segments
# 導入自定義模塊
from modules.youtube_utils import (extract_video_id, format_timestamp,
                                   format_transcript, get_video_info,
                                   get_video_transcript)


def main():
    """主函數"""
    # 設置頁面配置
    st.set_page_config(
        page_title="YouTube AI 助手",
        page_icon="🎬",
        layout="wide"
    )

    # 標題
    st.title("YouTube AI 助手")
    st.subheader("解析、翻譯和互動探索 YouTube 視頻內容")

    # 初始化會話狀態
    if "video_processed" not in st.session_state:
        st.session_state.video_processed = False
    if "video_id" not in st.session_state:
        st.session_state.video_id = ""
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "chat_agent" not in st.session_state:
        st.session_state.chat_agent = None
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = generate_thread_id()
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "error_message" not in st.session_state:
        st.session_state.error_message = None
    if "advanced_summary" not in st.session_state:
        st.session_state.advanced_summary = {}

    # 側邊欄 - 視頻輸入
    with st.sidebar:
        st.header("輸入 YouTube 視頻")
        youtube_url = st.text_input("輸入 YouTube 視頻 URL")

        # 目標語言選擇
        st.subheader("選項")
        target_language = st.selectbox(
            "字幕翻譯語言",
            ["中文", "英文", "日文", "韓文", "法文", "德文", "西班牙文"],
            index=0
        )

        # 摘要選項
        summary_type = st.radio(
            "摘要類型",
            ["標準摘要", "高級摘要（包含多個部分）"],
            index=0
        )

        # OpenAI API Key 輸入
        api_key = st.text_input("OpenAI API Key (可選，留空則使用環境變數)", type="password")
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key

        process_button = st.button("處理視頻")

        # 處理按鈕提示
        st.info("注意：處理較長視頻可能需要幾分鐘時間，請耐心等待。")

        # 如果用戶點擊處理按鈕
        if process_button and youtube_url:
            try:
                with st.spinner("處理視頻中..."):
                    logger.info(f"開始處理視頻: {youtube_url}")

                    # 提取視頻 ID
                    video_id = extract_video_id(youtube_url)
                    st.session_state.video_id = video_id
                    logger.info(f"提取到視頻 ID: {video_id}")

                    # 獲取視頻信息
                    st.text("正在獲取視頻信息...")
                    try:
                        video_info = get_video_info_with_retry(video_id)
                        st.session_state.video_info = video_info
                        logger.info(f"獲取到視頻信息: {video_info['title']}")
                    except Exception as e:
                        logger.error(f"獲取視頻信息出錯: {str(e)}")
                        st.error(f"獲取視頻信息時出錯: {str(e)}")
                        st.warning("嘗試使用基本信息繼續處理...")
                        # 創建基本視頻信息以繼續處理
                        video_info = {
                            "title": f"YouTube 視頻 ({video_id})",
                            "author": "未知作者",
                            "thumbnail_url": f"https://i.ytimg.com/vi/{video_id}/maxresdefault.jpg",
                            "length": 0,
                            "publish_date": None,
                            "views": 0
                        }
                        st.session_state.video_info = video_info

                    # 獲取視頻字幕
                    progress_bar = st.progress(0)
                    st.text("正在提取視頻字幕...")
                    try:
                        transcript = get_video_transcript_with_retry(video_id)
                        formatted_transcript = format_transcript(transcript)
                        st.session_state.transcript = formatted_transcript
                        progress_bar.progress(25)
                        logger.info(f"提取到字幕，總字數: {len(formatted_transcript['full_text'].split())}")
                    except Exception as e:
                        logger.error(f"獲取視頻字幕出錯: {str(e)}")
                        st.error(f"獲取視頻字幕時出錯: {str(e)}")
                        st.session_state.error_message = f"無法獲取視頻字幕: {str(e)}"
                        st.warning("無法繼續處理，請確保視頻有可用的字幕。")
                        st.info("提示: 有些視頻沒有字幕或者字幕受限制。請嘗試其他視頻。")
                        progress_bar.progress(100)
                        return

                    # 處理字幕並向量化
                    st.text("正在處理字幕並向量化...")
                    vector_store = split_and_vectorize_transcript(
                        formatted_transcript["full_text"],
                        video_id
                    )
                    st.session_state.vector_store = vector_store
                    progress_bar.progress(50)
                    logger.info("完成字幕向量化")

                    # 翻譯字幕
                    st.text(f"正在將字幕翻譯為{target_language}...")
                    translated_segments = translate_segments(formatted_transcript["segments"], target_language)
                    st.session_state.translated_segments = translated_segments
                    progress_bar.progress(75)
                    logger.info(f"完成字幕翻譯為{target_language}")

                    # 生成視頻摘要
                    st.text("正在生成視頻摘要...")
                    if summary_type == "標準摘要":
                        summary = generate_video_summary(vector_store)
                        st.session_state.summary = summary
                        st.session_state.advanced_summary = {}
                    else:
                        # 高級摘要
                        advanced_summary = generate_advanced_summary(vector_store, video_info)
                        st.session_state.advanced_summary = advanced_summary
                        st.session_state.summary = advanced_summary.get("overview", "摘要生成失敗")

                    logger.info("完成視頻摘要生成")

                    # 創建聊天代理
                    st.text("正在初始化 AI 對話助手...")
                    chat_agent = create_chat_agent(video_info, vector_store)
                    st.session_state.chat_agent = chat_agent
                    progress_bar.progress(100)
                    logger.info("完成 AI 對話助手初始化")

                    # 標記處理完成
                    st.session_state.video_processed = True

                    # 重置聊天歷史和錯誤信息
                    st.session_state.chat_history = []
                    st.session_state.error_message = None
                    st.session_state.thread_id = generate_thread_id()

                st.success("視頻處理完成！")
            except Exception as e:
                logger.error(f"處理視頻時出錯: {str(e)}")
                logger.error(traceback.format_exc())
                st.session_state.error_message = str(e)
                st.error(f"處理視頻時出錯: {str(e)}")

        if st.session_state.video_processed:
            # 顯示視頻信息
            st.subheader("視頻信息")
            st.write(f"標題: {st.session_state.video_info['title']}")
            st.write(f"作者: {st.session_state.video_info['author']}")
            st.write(f"時長: {format_timestamp(st.session_state.video_info['length'])}")
            st.write(f"觀看數: {st.session_state.video_info['views']}")

            # 顯示縮略圖
            st.image(st.session_state.video_info['thumbnail_url'], width=250)

            # 提供視頻連結
            video_url = f"https://www.youtube.com/watch?v={st.session_state.video_id}"
            st.markdown(f"[在 YouTube 上觀看]({video_url})")

    # 錯誤信息顯示
    if st.session_state.error_message:
        st.error(f"錯誤: {st.session_state.error_message}")

    # 主頁面 - 在視頻處理完成後顯示
    if st.session_state.video_processed:
        # 創建三個標籤
        tab1, tab2, tab3 = st.tabs(["視頻摘要", "字幕與翻譯", "AI 助手對話"])

        # 標籤 1: 視頻摘要
        with tab1:
            st.header("視頻摘要")

            # 如果是高級摘要，顯示多個部分
            if st.session_state.advanced_summary:
                # 顯示概述
                st.subheader("概述")
                st.write(st.session_state.advanced_summary.get("overview", "未能生成概述"))

                # 顯示關鍵點
                st.subheader("關鍵點")
                st.write(st.session_state.advanced_summary.get("key_points", "未能生成關鍵點"))

                # 顯示時間線
                st.subheader("內容時間線")
                st.write(st.session_state.advanced_summary.get("timeline", "未能生成時間線"))

                # 顯示結論
                st.subheader("結論與見解")
                st.write(st.session_state.advanced_summary.get("conclusion", "未能生成結論"))
            else:
                # 顯示標準摘要
                st.write(st.session_state.summary)

        # 標籤 2: 字幕與翻譯
        with tab2:
            st.header("字幕與翻譯")

            # 過濾選項
            search_term = st.text_input("搜索字幕", placeholder="輸入關鍵字搜索字幕...")

            # 創建 DataFrame 顯示字幕和翻譯
            subtitles_data = []
            for segment in st.session_state.translated_segments:
                subtitles_data.append({
                    "時間": f"{segment['start_formatted']} - {segment['end_formatted']}",
                    "原文": segment["text"],
                    "翻譯": segment["translation"],
                    "開始時間": segment["start"]  # 用於排序，不顯示
                })

            df = pd.DataFrame(subtitles_data)

            # 根據搜索詞過濾
            if search_term:
                filtered_df = df[df["原文"].str.contains(search_term, case=False) |
                                df["翻譯"].str.contains(search_term, case=False)]
                if len(filtered_df) > 0:
                    st.write(f"找到 {len(filtered_df)} 個匹配結果")
                    st.dataframe(filtered_df[["時間", "原文", "翻譯"]], use_container_width=True)
                else:
                    st.info("沒有找到匹配的字幕")
                    st.dataframe(df[["時間", "原文", "翻譯"]], use_container_width=True)
            else:
                # 按時間排序
                df = df.sort_values("開始時間")
                st.dataframe(df[["時間", "原文", "翻譯"]], use_container_width=True)

            # 提供下載選項
            csv = df[["時間", "原文", "翻譯"]].to_csv(index=False).encode('utf-8')
            st.download_button(
                label="下載字幕與翻譯 (CSV)",
                data=csv,
                file_name=f'subtitles_{st.session_state.video_id}.csv',
                mime='text/csv',
            )

        # 標籤 3: AI 助手對話
        with tab3:
            st.header("與視頻內容對話")

            # 顯示指導信息
            st.info("您可以向 AI 助手詢問關於視頻內容的問題。助手將根據視頻內容回答您的問題。")

            # 顯示對話歷史
            for message in st.session_state.chat_history:
                if message["role"] == "user":
                    st.chat_message("user").write(message["content"])
                else:
                    st.chat_message("assistant").write(message["content"])

            # 用戶輸入
            user_query = st.chat_input("問我關於視頻的問題...")

            if user_query:
                # 添加用戶消息到歷史
                st.session_state.chat_history.append({"role": "user", "content": user_query})
                st.chat_message("user").write(user_query)

                # 獲取 AI 回應
                with st.spinner("思考中..."):
                    try:
                        response = chat_with_video(
                            st.session_state.chat_agent,
                            user_query,
                            st.session_state.thread_id,
                            st.session_state.video_id
                        )
                    except Exception as e:
                        logger.error(f"生成回應時出錯: {str(e)}")
                        logger.error(traceback.format_exc())
                        response = f"處理您的問題時出錯: {str(e)}"

                # 添加 AI 回應到歷史
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                st.chat_message("assistant").write(response)
    else:
        # 未處理視頻時顯示說明
        st.info("請在左側輸入 YouTube 視頻 URL 並點擊 '處理視頻' 按鈕開始")
        st.write("""
        ### 使用說明

        1. 在左側輸入一個 YouTube 視頻的 URL
        2. 選擇翻譯目標語言和摘要類型
        3. 點擊 '處理視頻' 按鈕開始分析
        4. 系統將提取視頻字幕、生成摘要、翻譯內容並建立對話助手
        5. 在 '視頻摘要' 標籤查看內容概述
        6. 在 '字幕與翻譯' 標籤查看原文與翻譯對照
        7. 在 'AI 助手對話' 標籤與基於視頻內容的 AI 進行對話

        ### 示例視頻

        試試這些視頻 URL:

        - TED 演講: https://www.youtube.com/watch?v=8jPQjjsBbIc
        - 教學視頻: https://www.youtube.com/watch?v=5KnFcsSIzbg
        - 科學解釋: https://www.youtube.com/watch?v=nIVq1upU0eI
        """)

        # 介紹特色功能
        st.subheader("特色功能")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("#### 🔍 智能摘要")
            st.write("使用 AI 自動提取視頻的主要內容、關鍵點和結構，讓您快速了解視頻內容。")

        with col2:
            st.markdown("#### 🌐 字幕翻譯")
            st.write("將視頻字幕翻譯成您選擇的語言，提供原文與譯文對照，幫助您理解內容。")

        with col3:
            st.markdown("#### 💬 智能對話")
            st.write("與基於視頻內容的 AI 助手對話，詢問特定問題，深入探索視頻主題。")

        # 在側邊欄顯示一些使用技巧
        st.sidebar.header("使用技巧")
        tips = st.sidebar.expander("點擊查看使用技巧和常見問題")
        with tips:
            st.markdown("""
            ## 使用技巧

            1. **選擇合適的視頻**:
               - 確保視頻有字幕
               - 避免處理極長視頻 (> 1 小時)
               - 公開視頻比私人視頻更容易處理

            2. **摘要選擇**:
               - 標準摘要: 適合簡短視頻和快速瀏覽
               - 高級摘要: 提供更詳細的結構化內容，適合長視頻

            3. **AI 對話技巧**:
               - 使用具體問題獲得更準確的回答
               - 引用視頻中的特定段落進行提問
               - 可以詢問時間軸上的特定內容

            ## 常見問題

            1. **視頻無法處理**:
               - 確認 URL 格式正確
               - 確保視頻有可用字幕
               - 檢查視頻是否有地區限制

            2. **處理時間過長**:
               - 長視頻需要更多處理時間
               - 網絡連接問題可能導致延遲
               - 嘗試較短的視頻以測試功能

            3. **摘要或翻譯不精確**:
               - AI 生成內容有時可能不完美
               - 技術性或專業內容可能效果較差
               - 提供反饋以幫助改進系統
            """)

# 添加重試裝飾器函數，用於處理可重試的 API 錯誤
@retry(
    stop=stop_after_attempt(3),
    wait=wait_random_exponential(multiplier=1, max=10),
    retry=retry_if_exception_type((requests.exceptions.RequestException, TimeoutError))
)
def get_video_info_with_retry(video_id):
    """
    帶有重試機制的視頻信息獲取
    """
    return get_video_info(video_id)

@retry(
    stop=stop_after_attempt(3),
    wait=wait_random_exponential(multiplier=1, max=10),
    retry=retry_if_exception_type((requests.exceptions.RequestException, TimeoutError))
)
def get_video_transcript_with_retry(video_id):
    """
    帶有重試機制的視頻字幕獲取
    """
    return get_video_transcript(video_id)

if __name__ == "__main__":
    main()