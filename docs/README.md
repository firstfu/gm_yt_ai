# YouTube AI 助手

## 項目概述

YouTube AI 助手是一個強大的 AI 系統，專為處理 YouTube 視頻內容而設計。該系統能夠解析 YouTube 視頻字幕，生成內容摘要，提供字幕翻譯，並支持與視頻內容的智能對話互動。

### 主要功能

- **字幕提取**: 從任何 YouTube 視頻中提取字幕
- **內容摘要**: 自動生成視頻內容的結構化摘要
- **字幕翻譯**: 將字幕翻譯成中文，並提供原文對照
- **智能問答**: 基於視頻內容與 AI 助手進行對話
- **用戶友好界面**: 使用 Streamlit 提供直觀的用戶體驗

## 技術堆棧

- **Python 3.9+**: 主要開發語言
- **LangGraph 0.3.21**: 用於構建對話代理和知識圖譜
- **Streamlit 1.43.1**: 用於構建用戶界面
- **YouTube Transcript API**: 用於提取 YouTube 視頻字幕
- **FAISS/Chroma**: 向量數據庫，用於高效相似性搜索
- **大型語言模型** (例如 OpenAI GPT、Anthropic Claude 等): 用於內容理解、總結和對話生成

## 文檔結構

本項目的文檔分為以下幾個部分：

1. [系統架構](system_architecture.md) - 詳細描述系統的設計和組件
2. [技術實現詳細說明](technical_implementation.md) - 提供每個模塊的具體實現方法
3. [用戶指南](user_guide.md) - 指導用戶如何使用系統
4. [開發指南](developer_guide.md) - 開發者修改和擴展系統的指南

## 快速開始

### 安裝依賴項

```bash
pip install -r requirements.txt
```

### 配置 API 密鑰

創建一個 `.env` 文件在項目根目錄下：

```
OPENAI_API_KEY=your_openai_api_key
# 可選
ANTHROPIC_API_KEY=your_anthropic_api_key
```

### 運行應用

```bash
streamlit run app.py
```

### 使用流程

1. 在界面中輸入 YouTube 視頻 URL
2. 點擊 "處理視頻" 按鈕
3. 等待系統處理視頻內容
4. 在標籤頁中查看摘要、字幕翻譯或與 AI 助手對話

## 貢獻

歡迎對本項目進行貢獻！請參閱[開發指南](developer_guide.md)了解如何開始。

## 許可證

本項目採用 MIT 許可證 - 詳情請參閱 LICENSE 文件。
