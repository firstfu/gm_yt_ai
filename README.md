# YouTube AI 助手

## 項目概述

YouTube AI 助手是一個強大的 AI 系統，專為處理 YouTube 視頻內容而設計。該系統能夠解析 YouTube 視頻字幕，生成內容摘要，提供字幕翻譯，並支持與視頻內容的智能對話互動。

### 主要功能

- **字幕提取**: 從任何 YouTube 視頻中提取字幕
- **內容摘要**: 自動生成視頻內容的結構化摘要
- **字幕翻譯**: 將字幕翻譯成多種語言，並提供原文對照
- **智能問答**: 基於視頻內容與 AI 助手進行對話
- **用戶友好界面**: 使用 Streamlit 提供直觀的用戶體驗

## 技術堆棧

- **Python 3.9+**: 主要開發語言
- **LangGraph 0.3.21**: 用於構建對話代理和知識圖譜
- **Streamlit 1.43.1**: 用於構建用戶界面
- **YouTube Transcript API**: 用於提取 YouTube 視頻字幕
- **FAISS**: 向量數據庫，用於高效相似性搜索
- **大型語言模型** (例如 OpenAI GPT): 用於內容理解、總結和對話生成

## 快速開始

### 安裝依賴項

```bash
pip install -r requirements.txt
```

### 配置 API 密鑰

1. 創建一個 `.env` 文件在項目根目錄下，基於 `.env.example` 模板:

```
OPENAI_API_KEY=your_openai_api_key
```

或者在應用程序界面中直接輸入 API 密鑰。

### 運行應用

```bash
streamlit run app.py
```

### 使用流程

1. 在界面中輸入 YouTube 視頻 URL
2. 選擇字幕翻譯語言和摘要類型
3. 點擊 "處理視頻" 按鈕
4. 等待系統處理視頻內容
5. 在標籤頁中查看摘要、字幕翻譯或與 AI 助手對話

## 系統架構

系統由以下模塊組成：

- **YouTube 字幕提取模塊**: 提取和格式化字幕
- **文本處理模塊**: 文本分割和向量化
- **翻譯模塊**: 將字幕翻譯為多種語言
- **摘要模塊**: 生成視頻內容的結構化摘要
- **對話代理模塊**: 基於視頻內容的智能問答系統
- **Streamlit 界面**: 用戶友好的前端界面

## 目錄結構

```
youtube_ai_assistant/
├── app.py                   # Streamlit 應用主入口
├── requirements.txt         # 依賴項
├── .env.example             # 環境變量示例
├── modules/
│   ├── __init__.py
│   ├── youtube_utils.py     # YouTube 相關功能
│   ├── text_processor.py    # 文本處理功能
│   ├── translator.py        # 翻譯功能
│   ├── summarizer.py        # 摘要生成功能
│   └── chat_agent.py        # 對話代理功能
└── data/
    └── vector_stores/       # 向量存儲
```

## 錯誤處理與故障排除

本系統設計有多層次的錯誤處理機制，以確保即使在某些 API 限制或網絡問題的情況下仍能正常運行。

### 常見問題與解決方案

1. **YouTube API 限制**

   - **症狀**: HTTP 400 錯誤或「無法獲取視頻信息」
   - **解決方案**:
     - 系統會自動嘗試多種方法獲取視頻信息
     - 檢查視頻是否可公開訪問
     - 部分私人視頻或受限視頻可能無法處理

2. **字幕不可用**

   - **症狀**: 「找不到字幕」錯誤
   - **解決方案**:
     - 確保視頻有字幕（自動或手動）
     - 如果視頻沒有字幕，本工具將無法提取內容

3. **OpenAI API 限制**

   - **症狀**: 模型調用錯誤或超時
   - **解決方案**:
     - 確保 API 密鑰有效且有足夠的使用額度
     - 對於較長視頻，可能需要等待更長時間

4. **記憶體或 CPU 使用過高**

   - **症狀**: 應用運行緩慢或崩潰
   - **解決方案**:
     - 避免處理極長的視頻（>1 小時）
     - 增加系統記憶體
     - 使用更高效的硬體

5. **LangGraph 相容性問題**
   - **症狀**: 匯入或初始化 LangGraph 時出錯
   - **解決方案**:
     - 確保 LangGraph 版本為 0.1.0 或更高
     - 檢查所有依賴項是否正確安裝

### 進階故障排除

若遇到複雜問題，請嘗試以下步驟：

1. 檢查日誌文件獲取詳細錯誤信息
2. 更新所有相關套件到最新版本
3. 檢查 YouTube 服務狀態
4. 檢查 OpenAI API 服務狀態
5. 在不同視頻上測試，排除特定視頻問題

## 效能優化

- 對於較長視頻，系統會自動分割文本以提高處理效率
- 向量存儲使用 FAISS 提供高效語義搜尋
- 系統會緩存已處理的視頻信息，避免重複處理
- 使用非同步處理某些任務以提高響應速度

## 注意事項

- 需要有效的 OpenAI API 密鑰才能使用所有功能
- 處理較長的視頻可能需要更多時間和資源
- 系統依賴於 YouTube 視頻必須有可用的字幕
- 系統具有多層容錯機制，但無法處理所有可能的錯誤情況

## 授權

本項目採用 MIT 許可證 - 詳情請參閱 LICENSE 文件。

## 贊助

如果您發現這個工具有用，請考慮給專案一個星標，或向作者捐贈以支持持續開發。
