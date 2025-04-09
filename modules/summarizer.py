"""
摘要模塊

該模塊使用 LangGraph 生成視頻摘要：
- 定義摘要工作流圖
- 處理文檔摘要節點
"""

import logging
from typing import Any, Dict, List, Optional, TypedDict

from langchain.prompts.chat import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

# 設置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 定義狀態類型
class SummaryState(TypedDict):
    documents: List[Document]
    query: str
    summary: Optional[str]

# 獲取視頻摘要節點
def get_summary(state: SummaryState) -> SummaryState:
    """基於文檔生成視頻摘要

    Args:
        state (SummaryState): 包含文檔和查詢的狀態

    Returns:
        SummaryState: 更新後的狀態，包含摘要
    """
    logger.info("正在生成視頻摘要")

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

        請提供一個結構化的摘要，使用標題和子標題組織內容。使用繁體中文回覆。"""
    )

    # 創建模型和鏈
    logger.info("使用 ChatOpenAI 模型生成摘要")
    model = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k")
    chain = prompt | model

    # 生成摘要
    try:
        summary = chain.invoke({"text": text_content})
        logger.info("摘要生成成功")

        # 更新狀態
        state["summary"] = summary.content
    except Exception as e:
        logger.error(f"生成摘要時出錯: {str(e)}")
        state["summary"] = f"摘要生成失敗: {str(e)}"

    return state

# 創建摘要工作流圖
def create_summary_graph():
    """創建摘要生成工作流圖

    Returns:
        StateGraph: 編譯後的工作流圖
    """
    logger.info("創建摘要工作流圖")

    # 初始化圖
    workflow = StateGraph(SummaryState)

    # 添加節點
    workflow.add_node("get_summary", get_summary)

    # 設置邊
    workflow.set_entry_point("get_summary")
    workflow.add_edge("get_summary", END)

    # 編譯圖
    logger.info("編譯摘要工作流圖")
    return workflow.compile()

def generate_video_summary(vector_store, query: str = "總結這個視頻的內容") -> str:
    """生成視頻摘要

    Args:
        vector_store: 向量存儲對象
        query (str, optional): 摘要查詢，默認為 "總結這個視頻的內容"

    Returns:
        str: 生成的摘要
    """
    logger.info(f"開始使用查詢 '{query}' 生成視頻摘要")

    # 獲取相關文檔
    try:
        docs = vector_store.similarity_search(query, k=10)
        logger.info(f"從向量存儲中檢索到 {len(docs)} 個相關文檔")
    except Exception as e:
        logger.error(f"檢索文檔時出錯: {str(e)}")
        return f"無法生成摘要: {str(e)}"

    # 創建初始狀態
    initial_state = {
        "documents": docs,
        "query": query,
        "summary": None
    }

    # 執行摘要圖
    try:
        logger.info("執行摘要工作流")
        summary_graph = create_summary_graph()
        result = summary_graph.invoke(initial_state)

        if result["summary"]:
            logger.info("成功生成摘要")
            return result["summary"]
        else:
            logger.warning("未能生成摘要")
            return "無法生成摘要，請稍後再試。"
    except Exception as e:
        logger.error(f"執行摘要工作流時出錯: {str(e)}")
        return f"生成摘要時發生錯誤: {str(e)}"

# 高級摘要生成
def generate_advanced_summary(vector_store, video_info: Dict[str, Any]) -> Dict[str, str]:
    """生成高級視頻摘要，包括多個部分

    Args:
        vector_store: 向量存儲對象
        video_info (Dict[str, Any]): 視頻信息

    Returns:
        Dict[str, str]: 包含各部分摘要的字典
    """
    logger.info("開始生成高級視頻摘要")

    # 定義不同的摘要部分
    summary_parts = {
        "overview": "提供一個簡短的視頻概述，不超過 3-5 句話",
        "key_points": "列出視頻中討論的主要要點和論點",
        "timeline": "提供視頻內容的時間線，按順序列出主要事件或主題",
        "conclusion": "總結視頻的主要結論或見解"
    }

    results = {}

    # 為每個部分生成摘要
    for part_name, part_query in summary_parts.items():
        try:
            logger.info(f"生成 '{part_name}' 部分的摘要")

            # 構建針對性查詢
            full_query = f"{part_query}。視頻標題: {video_info.get('title', '未知')}"

            # 獲取相關文檔
            docs = vector_store.similarity_search(full_query, k=5)

            # 創建初始狀態
            initial_state = {
                "documents": docs,
                "query": full_query,
                "summary": None
            }

            # 執行摘要圖
            summary_graph = create_summary_graph()
            result = summary_graph.invoke(initial_state)

            # 保存結果
            if result["summary"]:
                results[part_name] = result["summary"]
            else:
                results[part_name] = f"未能生成 {part_name} 摘要"

        except Exception as e:
            logger.error(f"生成 '{part_name}' 摘要時出錯: {str(e)}")
            results[part_name] = f"生成 {part_name} 摘要時出錯: {str(e)}"

    logger.info("高級摘要生成完成")
    return results