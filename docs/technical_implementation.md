# YouTube AI 助手技術實現詳細說明

本文檔詳細介紹各個系統模塊的技術實現方法，包括所用工具、代碼組織、關鍵算法和整合方式。

## 環境配置

### 依賴項管理

```python
# requirements.txt
streamlit==1.43.1
langgraph==0.3.21
langchain==0.2.10
langchain-openai==0.0.5
langchain-community==0.0.13
youtube-transcript-api==0.6.1
faiss-cpu==1.7.4
python-dotenv==1.0.0
pytube==15.0.0
requests==2.31.0
pydantic==2.7.1
```

### 配置文件

我們使用 `.env` 文件來存儲敏感信息，如 API 密鑰：

```
# .env
OPENAI_API_KEY=your_openai_api_key
# 可選的其他 LLM API 密鑰
ANTHROPIC_API_KEY=your_anthropic_api_key
```

## 1. YouTube 字幕提取模塊

### 核心功能實現

```python
from youtube_transcript_api import YouTubeTranscriptApi
from pytube import YouTube
import re

def extract_video_id(youtube_url: str) -> str:
    """從 YouTube URL 中提取視頻 ID"""
    # 支持多種 YouTube URL 格式
    patterns = [
        r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/watch\?v=([^&\s]+)',
        r'(?:https?:\/\/)?(?:www\.)?youtu\.be\/([^\?\s]+)',
        r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/embed\/([^\?\s]+)'
    ]

    for pattern in patterns:
        match = re.search(pattern, youtube_url)
        if match:
            return match.group(1)

    raise ValueError("無法從提供的 URL 中提取視頻 ID")

def get_video_info(video_id: str) -> dict:
    """獲取視頻的基本信息"""
    try:
        yt = YouTube(f"https://www.youtube.com/watch?v={video_id}")
        return {
            "title": yt.title,
            "author": yt.author,
            "length": yt.length,
            "publish_date": yt.publish_date,
            "thumbnail_url": yt.thumbnail_url,
            "views": yt.views
        }
    except Exception as e:
        raise Exception(f"獲取視頻信息時出錯: {str(e)}")

def get_video_transcript(video_id: str, language: str = "en") -> list:
    """獲取視頻字幕"""
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

        # 嘗試獲取指定語言的字幕
        try:
            transcript = transcript_list.find_transcript([language])
        except:
            # 如果失敗，嘗試獲取可用的字幕並翻譯
            transcript = transcript_list.find_transcript(['en'])
            if language != 'en':
                transcript = transcript.translate(language)

        return transcript.fetch()
    except Exception as e:
        raise Exception(f"獲取視頻字幕時出錯: {str(e)}")

def format_transcript(transcript: list) -> dict:
    """格式化字幕數據"""
    formatted_text = ""
    segments = []

    for segment in transcript:
        text = segment['text']
        start = segment['start']
        duration = segment['duration']

        # 格式化時間戳
        start_time = format_timestamp(start)
        end_time = format_timestamp(start + duration)

        # 添加到完整文本
        formatted_text += f"{text} "

        # 保存分段信息
        segments.append({
            "text": text,
            "start": start,
            "start_formatted": start_time,
            "end": start + duration,
            "end_formatted": end_time
        })

    return {
        "full_text": formatted_text.strip(),
        "segments": segments
    }

def format_timestamp(seconds: float) -> str:
    """將秒數轉換為時:分:秒格式"""
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)

    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    else:
        return f"{minutes:02d}:{seconds:02d}"
```

## 2. 內容處理模塊

### 文本分割和向量化

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.docstore.document import Document

def split_and_vectorize_transcript(transcript_text: str, video_id: str, chunk_size: int = 1000, chunk_overlap: int = 100) -> FAISS:
    """將字幕文本分割並向量化"""
    # 創建文本分割器
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    # 將文本分割成文檔
    docs = text_splitter.create_documents([transcript_text])

    # 為每個文檔添加元數據
    for doc in docs:
        doc.metadata = {"source": f"youtube_video_{video_id}"}

    # 創建嵌入模型
    embeddings = OpenAIEmbeddings()

    # 創建向量存儲
    vector_store = FAISS.from_documents(docs, embeddings)

    return vector_store
```

## 3. 翻譯模塊

### 使用 LLM 進行翻譯

```python
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

def translate_segments(segments: list, target_language: str = "中文") -> list:
    """翻譯字幕分段"""
    # 創建 LLM
    llm = OpenAI(temperature=0.1)

    # 創建翻譯提示模板
    prompt_template = PromptTemplate(
        input_variables=["text"],
        template=f"將以下文本準確地翻譯成{target_language}，保持原意。文本：{{text}}"
    )

    # 創建翻譯鏈
    translate_chain = LLMChain(llm=llm, prompt=prompt_template)

    # 翻譯每個分段
    translated_segments = []
    for segment in segments:
        translation = translate_chain.run(segment["text"]).strip()

        translated_segment = segment.copy()
        translated_segment["translation"] = translation
        translated_segments.append(translated_segment)

    return translated_segments
```

## 4. 內容總結模塊

### 使用 LangGraph 生成摘要

```python
from langgraph.graph import StateGraph, END
from langchain.prompts.chat import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.chat_models import ChatOpenAI
from typing import TypedDict, List, Optional
from langchain.schema import Document

# 定義狀態類型
class SummaryState(TypedDict):
    documents: List[Document]
    query: str
    summary: Optional[str]

# 獲取視頻摘要節點
def get_summary(state: SummaryState) -> SummaryState:
    """基於文檔生成視頻摘要"""
    # 組合所有文檔的內容
    text_content = "\n\n".join([doc.page_content for doc in state["documents"]])

    # 創建摘要提示
    prompt = ChatPromptTemplate.from_template(
        """你是一個專業的視頻內容分析師。根據以下視頻字幕內容，生成一個全面的摘要。
        摘要應該包括：
        1. 視頻的主要主題和目的
        2. 關鍵點和主要論點
        3. 重要的細節和例子
        4. 視頻的整體結構和流程

        視頻字幕內容:
        {text}

        請提供一個結構化的摘要，使用標題和子標題組織內容。"""
    )

    # 創建模型和鏈
    model = ChatOpenAI(temperature=0)
    chain = prompt | model

    # 生成摘要
    summary = chain.invoke({"text": text_content})

    # 更新狀態
    state["summary"] = summary.content
    return state

# 創建摘要工作流圖
def create_summary_graph():
    """創建摘要生成工作流圖"""
    # 初始化圖
    workflow = StateGraph(SummaryState)

    # 添加節點
    workflow.add_node("get_summary", get_summary)

    # 設置邊
    workflow.set_entry_point("get_summary")
    workflow.add_edge("get_summary", END)

    # 編譯圖
    return workflow.compile()

def generate_video_summary(vector_store, query: str = "總結這個視頻的內容") -> str:
    """生成視頻摘要"""
    # 獲取相關文檔
    docs = vector_store.similarity_search(query, k=10)

    # 創建初始狀態
    initial_state = {
        "documents": docs,
        "query": query,
        "summary": None
    }

    # 執行摘要圖
    summary_graph = create_summary_graph()
    result = summary_graph.invoke(initial_state)

    return result["summary"]
```

## 5. 對話助手模塊

### 使用 LangGraph 構建對話代理

```python
from typing import List, Dict, TypedDict, Optional, Annotated, Union
from langchain.schema.messages import (
    AIMessage, HumanMessage, SystemMessage, FunctionMessage
)
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain.tools import BaseTool, StructuredTool, tool
from langchain.chat_models import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver

# 定義對話狀態
class ChatState(TypedDict):
    messages: Annotated[List[Union[AIMessage, HumanMessage, SystemMessage, FunctionMessage]], add_messages]
    video_id: str

# 創建搜索工具
@tool
def search_video_content(query: str, vector_store) -> str:
    """從視頻內容中搜索相關信息"""
    docs = vector_store.similarity_search(query, k=3)
    if not docs:
        return "找不到相關信息。"

    content = "\n\n".join([doc.page_content for doc in docs])
    return content

# 構建 LLM 代理
def create_chat_agent(video_info: dict, vector_store):
    """創建基於視頻內容的對話代理"""
    # 創建系統提示
    system_prompt = f"""你是一個友好、專業的 AI 助手，專門解答關於以下視頻的問題：
    標題: {video_info['title']}
    作者: {video_info['author']}

    你可以使用搜索工具查找視頻中的相關內容來回答問題。
    如果用戶問的問題與視頻無關，請禮貌地提醒他們你的專長是討論這個特定視頻的內容。
    總是提供準確、有幫助、相關的信息，並引用視頻中的具體內容。
    """

    # 創建模型
    model = ChatOpenAI(temperature=0.7)

    # 創建搜索工具
    search_tool = StructuredTool.from_function(
        func=lambda query: search_video_content(query, vector_store),
        name="search_video_content",
        description="從視頻內容中搜索相關信息"
    )

    # 定義代理節點
    def agent_node(state):
        """LLM 代理節點"""
        messages = state["messages"]

        # 如果是第一條消息，添加系統提示
        if len(messages) == 1 and isinstance(messages[0], HumanMessage):
            messages = [SystemMessage(content=system_prompt)] + messages

        # 調用模型
        response = model.invoke(messages)
        return {"messages": [response]}

    # 定義工具節點
    tools_node = ToolNode([search_tool])

    # 創建圖
    workflow = StateGraph(ChatState)

    # 添加節點
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tools_node)

    # 添加邊
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent",
        tools_condition,
        {
            "tools": "tools",
            END: END
        }
    )
    workflow.add_edge("tools", "agent")

    # 添加記憶體支持
    memory = MemorySaver()

    # 編譯圖
    return workflow.compile(checkpointer=memory)

# 使用代理進行對話
def chat_with_video(chat_agent, query: str, thread_id: str, video_id: str):
    """與視頻內容對話"""
    # 設置配置
    config = {"configurable": {"thread_id": thread_id}}

    # 創建消息
    messages = [HumanMessage(content=query)]

    # 初始狀態
    state = {
        "messages": messages,
        "video_id": video_id
    }

    # 調用代理
    result = chat_agent.invoke(state, config=config)

    # 返回結果
    return result["messages"][-1].content
```

## 6. Streamlit 前端界面

### 主界面實現

```python
import streamlit as st
import json
import time

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
        st.session_state.thread_id = str(int(time.time()))
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # 側邊欄 - 視頻輸入
    with st.sidebar:
        st.header("輸入 YouTube 視頻")
        youtube_url = st.text_input("輸入 YouTube 視頻 URL")
        process_button = st.button("處理視頻")

        # 如果用戶點擊處理按鈕
        if process_button and youtube_url:
            try:
                with st.spinner("處理視頻中..."):
                    # 提取視頻 ID
                    video_id = extract_video_id(youtube_url)
                    st.session_state.video_id = video_id

                    # 獲取視頻信息
                    video_info = get_video_info(video_id)
                    st.session_state.video_info = video_info

                    # 獲取視頻字幕
                    transcript = get_video_transcript(video_id)
                    formatted_transcript = format_transcript(transcript)
                    st.session_state.transcript = formatted_transcript

                    # 處理字幕並向量化
                    vector_store = split_and_vectorize_transcript(
                        formatted_transcript["full_text"],
                        video_id
                    )
                    st.session_state.vector_store = vector_store

                    # 翻譯字幕
                    translated_segments = translate_segments(formatted_transcript["segments"])
                    st.session_state.translated_segments = translated_segments

                    # 生成視頻摘要
                    summary = generate_video_summary(vector_store)
                    st.session_state.summary = summary

                    # 創建聊天代理
                    chat_agent = create_chat_agent(video_info, vector_store)
                    st.session_state.chat_agent = chat_agent

                    # 標記處理完成
                    st.session_state.video_processed = True

                    # 重置聊天歷史
                    st.session_state.chat_history = []

                st.success("視頻處理完成！")
            except Exception as e:
                st.error(f"處理視頻時出錯: {str(e)}")

        if st.session_state.video_processed:
            # 顯示視頻信息
            st.subheader("視頻信息")
            st.write(f"標題: {st.session_state.video_info['title']}")
            st.write(f"作者: {st.session_state.video_info['author']}")
            st.write(f"時長: {format_timestamp(st.session_state.video_info['length'])}")
            st.write(f"觀看數: {st.session_state.video_info['views']}")

    # 主頁面 - 在視頻處理完成後顯示
    if st.session_state.video_processed:
        # 創建三個標籤
        tab1, tab2, tab3 = st.tabs(["視頻摘要", "字幕與翻譯", "AI 助手對話"])

        # 標籤 1: 視頻摘要
        with tab1:
            st.header("視頻摘要")
            st.write(st.session_state.summary)

        # 標籤 2: 字幕與翻譯
        with tab2:
            st.header("字幕與翻譯")

            # 創建 DataFrame 顯示字幕和翻譯
            import pandas as pd

            subtitles_data = []
            for segment in st.session_state.translated_segments:
                subtitles_data.append({
                    "時間": f"{segment['start_formatted']} - {segment['end_formatted']}",
                    "原文": segment["text"],
                    "翻譯": segment["translation"]
                })

            df = pd.DataFrame(subtitles_data)
            st.dataframe(df, use_container_width=True)

        # 標籤 3: AI 助手對話
        with tab3:
            st.header("與視頻內容對話")

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
                    response = chat_with_video(
                        st.session_state.chat_agent,
                        user_query,
                        st.session_state.thread_id,
                        st.session_state.video_id
                    )

                # 添加 AI 回應到歷史
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                st.chat_message("assistant").write(response)
    else:
        # 未處理視頻時顯示說明
        st.info("請在左側輸入 YouTube 視頻 URL 並點擊 '處理視頻' 按鈕開始")
        st.write("""
        ### 使用說明

        1. 在左側輸入一個 YouTube 視頻的 URL
        2. 點擊 '處理視頻' 按鈕開始分析
        3. 系統將提取視頻字幕、生成摘要、翻譯內容並建立對話助手
        4. 在 '視頻摘要' 標籤查看內容概述
        5. 在 '字幕與翻譯' 標籤查看原文與翻譯對照
        6. 在 'AI 助手對話' 標籤與基於視頻內容的 AI 進行對話
        """)

if __name__ == "__main__":
    main()
```

## 7. 數據存儲和緩存實現

### 使用 FAISS 存儲向量數據

```python
def create_persistent_vector_store(video_id: str, documents: list, embeddings):
    """創建持久化向量存儲"""
    # 使用 FAISS 創建向量存儲
    vector_store = FAISS.from_documents(documents, embeddings)

    # 保存向量存儲
    vector_store.save_local(f"vector_stores/{video_id}")

    return vector_store

def load_vector_store(video_id: str, embeddings):
    """加載持久化向量存儲"""
    try:
        # 嘗試加載現有的向量存儲
        vector_store = FAISS.load_local(f"vector_stores/{video_id}", embeddings)
        return vector_store
    except:
        return None
```

### 使用 LangGraph 的 MemorySaver 實現對話記憶

```python
from langgraph.checkpoint.memory import MemorySaver

# 創建記憶體保存器
memory = MemorySaver()

# 在圖編譯時使用
graph = workflow.compile(checkpointer=memory)

# 在調用時提供 thread_id
config = {"configurable": {"thread_id": thread_id}}
result = graph.invoke(state, config=config)
```

## 項目結構

```
youtube_ai_assistant/
├── app.py                   # Streamlit 應用主入口
├── requirements.txt         # 依賴項
├── .env                     # 環境變量
├── modules/
│   ├── __init__.py
│   ├── youtube_utils.py     # YouTube 相關功能
│   ├── text_processor.py    # 文本處理功能
│   ├── translator.py        # 翻譯功能
│   ├── summarizer.py        # 摘要生成功能
│   └── chat_agent.py        # 對話代理功能
├── data/
│   └── vector_stores/       # 向量存儲
└── docs/
    ├── system_architecture.md
    └── technical_implementation.md
```
