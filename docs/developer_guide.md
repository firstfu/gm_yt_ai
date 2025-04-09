# YouTube AI 助手開發指南

本文檔為開發者提供關於系統架構、代碼組織和擴展方法的詳細信息。無論您是想要修改現有功能、添加新功能，還是僅僅理解系統的工作原理，本指南都將幫助您熟悉項目的技術細節。

## 目錄

1. [技術架構概述](#技術架構概述)
2. [開發環境設置](#開發環境設置)
3. [代碼組織](#代碼組織)
4. [核心模塊](#核心模塊)
5. [擴展指南](#擴展指南)
6. [測試策略](#測試策略)
7. [部署](#部署)
8. [貢獻指南](#貢獻指南)

## 技術架構概述

YouTube AI 助手採用模塊化架構，使用以下主要技術：

- **Streamlit**: 用於前端界面
- **LangGraph**: 用於構建對話代理和工作流
- **LangChain**: 用於處理文本和與 LLM 的交互
- **YouTube Transcript API**: 用於提取視頻字幕
- **FAISS**: 用於向量存儲和相似性搜索
- **大型語言模型 (LLM)**: 用於生成摘要、翻譯和對話回應

系統的主要數據流如下：

1. 用戶輸入 YouTube URL
2. 系統提取視頻字幕
3. 字幕被處理並存儲在向量數據庫中
4. LLM 生成摘要和翻譯
5. 用戶與基於視頻內容的 AI 助手進行對話
6. 對話代理使用向量搜索查找相關信息並生成回應

## 開發環境設置

### 先決條件

- Python 3.9+
- 虛擬環境工具 (如 venv 或 conda)
- Git
- OpenAI API 密鑰或其他 LLM API 密鑰

### 設置步驟

1. **克隆存儲庫**

```bash
git clone https://github.com/yourusername/youtube-ai-assistant.git
cd youtube-ai-assistant
```

2. **創建虛擬環境**

```bash
python -m venv venv
source venv/bin/activate  # 在 Windows 上使用 venv\Scripts\activate
```

3. **安裝依賴項**

```bash
pip install -r requirements.txt
```

4. **設置環境變量**

創建 `.env` 文件：

```
OPENAI_API_KEY=your_openai_api_key
# 可選的其他 LLM API 密鑰
ANTHROPIC_API_KEY=your_anthropic_api_key
```

5. **啟動開發服務器**

```bash
streamlit run app.py
```

## 代碼組織

項目代碼組織如下：

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
    ├── technical_implementation.md
    ├── user_guide.md
    └── developer_guide.md
```

## 核心模塊

### YouTube 工具模塊 (`youtube_utils.py`)

這個模塊處理與 YouTube 相關的功能：

- 從 URL 提取視頻 ID
- 獲取視頻元數據
- 提取視頻字幕
- 格式化字幕數據

關鍵函數：

- `extract_video_id(youtube_url)`: 從 URL 中提取視頻 ID
- `get_video_info(video_id)`: 獲取視頻基本信息
- `get_video_transcript(video_id, language)`: 獲取視頻字幕
- `format_transcript(transcript)`: 格式化字幕數據

### 文本處理模塊 (`text_processor.py`)

該模塊處理文本分割和向量化：

- 將字幕文本分割成較小的塊
- 創建和管理向量存儲
- 處理相似性搜索

關鍵函數：

- `split_and_vectorize_transcript(transcript_text, video_id)`: 分割並向量化字幕
- `create_persistent_vector_store(video_id, documents, embeddings)`: 創建持久化向量存儲
- `load_vector_store(video_id, embeddings)`: 加載向量存儲

### 翻譯模塊 (`translator.py`)

該模塊處理字幕翻譯：

- 使用 LLM 進行高質量翻譯
- 保持時間戳對應關係

關鍵函數：

- `translate_segments(segments, target_language)`: 翻譯字幕分段

### 摘要模塊 (`summarizer.py`)

這個模塊使用 LangGraph 生成視頻摘要：

- 定義摘要工作流圖
- 處理文檔摘要節點

關鍵函數和類：

- `SummaryState`: 定義摘要狀態類型
- `get_summary(state)`: 生成摘要的節點函數
- `create_summary_graph()`: 創建摘要工作流圖
- `generate_video_summary(vector_store, query)`: 生成視頻摘要的主函數

### 對話代理模塊 (`chat_agent.py`)

這個模塊使用 LangGraph 實現對話代理：

- 定義對話狀態和工具
- 構建代理工作流圖
- 管理對話記憶

關鍵函數和類：

- `ChatState`: 定義對話狀態類型
- `search_video_content(query, vector_store)`: 搜索工具
- `create_chat_agent(video_info, vector_store)`: 創建對話代理
- `chat_with_video(chat_agent, query, thread_id, video_id)`: 與視頻內容對話

## 擴展指南

### 添加新的語言模型

要添加新的語言模型，修改 `chat_agent.py` 和 `summarizer.py` 中的相關函數：

```python
# 在 chat_agent.py 中
def create_chat_agent(video_info, vector_store, model_name="openai"):
    # 選擇合適的模型
    if model_name == "openai":
        model = ChatOpenAI(temperature=0.7)
    elif model_name == "anthropic":
        from langchain.chat_models import ChatAnthropic
        model = ChatAnthropic(temperature=0.7)
    elif model_name == "local":
        from langchain.llms import LlamaCpp
        model = LlamaCpp(model_path="path/to/your/model.bin")
    else:
        raise ValueError(f"不支持的模型類型: {model_name}")

    # 其餘代碼保持不變
    # ...
```

### 添加新工具

要向代理添加新工具，在 `chat_agent.py` 中定義工具函數並添加到工具列表：

```python
@tool
def get_video_statistics(video_id: str) -> str:
    """獲取視頻的統計信息，如觀看次數、點贊數等"""
    # 實現功能...
    return statistics

# 在 create_chat_agent 函數中
def create_chat_agent(video_info, vector_store):
    # ...

    # 創建工具列表
    tools = [
        StructuredTool.from_function(
            func=lambda query: search_video_content(query, vector_store),
            name="search_video_content",
            description="從視頻內容中搜索相關信息"
        ),
        StructuredTool.from_function(
            func=lambda vid_id: get_video_statistics(vid_id),
            name="get_video_statistics",
            description="獲取視頻的統計信息"
        )
    ]

    # 定義工具節點
    tools_node = ToolNode(tools)

    # ...
```

### 自定義摘要格式

要修改摘要格式，更新 `summarizer.py` 中的提示模板：

```python
def get_summary(state: SummaryState) -> SummaryState:
    # ...

    # 創建自定義摘要提示
    prompt = ChatPromptTemplate.from_template(
        """你是一個專業的視頻內容分析師。根據以下視頻字幕內容，生成一個全面的摘要。
        摘要應該包括：
        1. 視頻的主要主題和目的 (用一句話概括)
        2. 視頻內容的時間線 (按發生順序列出關鍵事件或點)
        3. 關鍵見解和重要報價 (使用引號標記直接引用)
        4. 適合此視頻的受眾

        以大綱格式呈現，使用項目符號和子項目符號以提高可讀性。

        視頻字幕內容:
        {text}

        摘要:"""
    )

    # ...
```

### 添加新的前端功能

要擴展 Streamlit 前端，修改 `app.py`：

```python
# 添加新標籤或頁面
tab1, tab2, tab3, tab4 = st.tabs(["視頻摘要", "字幕與翻譯", "AI 助手對話", "視頻分析"])

# 在新標籤中添加自定義功能
with tab4:
    st.header("視頻分析")

    # 添加視覺化或其他功能
    if st.session_state.video_processed:
        # 顯示視頻統計圖表
        import matplotlib.pyplot as plt
        import numpy as np

        # 生成示例數據
        word_counts = {}
        for segment in st.session_state.translated_segments:
            for word in segment["text"].split():
                word = word.lower().strip(".,!?")
                if word:
                    word_counts[word] = word_counts.get(word, 0) + 1

        # 顯示前 10 個最常用詞
        top_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        words, counts = zip(*top_words)

        fig, ax = plt.subplots()
        ax.bar(words, counts)
        ax.set_title("視頻中最常用的詞")
        ax.set_ylabel("出現次數")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        st.pyplot(fig)
```

## 測試策略

### 單元測試

使用 `pytest` 為每個模塊創建單元測試：

```python
# test_youtube_utils.py
import pytest
from modules.youtube_utils import extract_video_id, format_timestamp

def test_extract_video_id():
    # 標準 URL
    assert extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    # 短 URL
    assert extract_video_id("https://youtu.be/dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    # 帶參數的 URL
    assert extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=10s") == "dQw4w9WgXcQ"

    # 無效 URL
    with pytest.raises(ValueError):
        extract_video_id("https://www.example.com")

def test_format_timestamp():
    assert format_timestamp(65) == "01:05"
    assert format_timestamp(3661) == "01:01:01"
    assert format_timestamp(0) == "00:00"
```

### 整合測試

創建整合測試來驗證模塊之間的交互：

```python
# test_integration.py
import pytest
from modules.youtube_utils import get_video_transcript, format_transcript
from modules.text_processor import split_and_vectorize_transcript
from modules.summarizer import generate_video_summary

@pytest.mark.integration
def test_end_to_end_processing():
    # 使用已知的視頻 ID
    video_id = "dQw4w9WgXcQ"

    # 提取字幕
    transcript = get_video_transcript(video_id)
    assert transcript is not None
    assert len(transcript) > 0

    # 格式化字幕
    formatted = format_transcript(transcript)
    assert "full_text" in formatted
    assert "segments" in formatted

    # 向量化文本 (使用模擬嵌入)
    from unittest.mock import MagicMock
    mock_embeddings = MagicMock()
    mock_embeddings.embed_documents.return_value = [[0.1] * 10] * len(formatted["segments"])

    vector_store = split_and_vectorize_transcript(
        formatted["full_text"],
        video_id,
        embedding_model=mock_embeddings
    )

    # 驗證向量存儲
    assert vector_store is not None
```

## 部署

### 本地部署

1. 確保安裝了所有依賴項：

```bash
pip install -r requirements.txt
```

2. 設置環境變量：

```bash
export OPENAI_API_KEY=your_key  # Linux/Mac
# 或
set OPENAI_API_KEY=your_key  # Windows
```

3. 運行應用：

```bash
streamlit run app.py
```

### Docker 部署

1. 創建 Dockerfile：

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]
```

2. 構建和運行 Docker 映像：

```bash
docker build -t youtube-ai-assistant .
docker run -p 8501:8501 -e OPENAI_API_KEY=your_key youtube-ai-assistant
```

### 雲部署

#### Streamlit Cloud

1. 將代碼推送到 GitHub 存儲庫
2. 在 [Streamlit Cloud](https://streamlit.io/cloud) 上創建新的應用
3. 連接到您的 GitHub 存儲庫
4. 添加必要的密鑰作為密鑰環境變量

#### Heroku

1. 創建 `Procfile`：

```
web: streamlit run app.py
```

2. 部署到 Heroku：

```bash
heroku create
git push heroku main
heroku config:set OPENAI_API_KEY=your_key
```

## 貢獻指南

### 提交 Pull Request

1. Fork 存儲庫
2. 創建功能分支：`git checkout -b feature/amazing-feature`
3. 提交更改：`git commit -m 'Add amazing feature'`
4. 推送到分支：`git push origin feature/amazing-feature`
5. 打開 Pull Request

### 代碼風格

- 遵循 PEP 8 標準
- 使用類型提示增強代碼可讀性
- 為所有函數和類添加 docstring
- 使用有意義的變量和函數名

### 報告問題

- 使用 GitHub Issues 報告錯誤或請求功能
- 提供重現錯誤的明確步驟
- 包括系統信息和依賴項版本

---

如果您有任何問題或需要進一步的指導，請參考文檔或通過 GitHub Issues 與我們聯繫。祝您在開發中取得成功！
