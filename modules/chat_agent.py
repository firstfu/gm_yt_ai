"""
對話代理模塊

該模塊使用 LangGraph 實現對話代理：
- 定義對話狀態和工具
- 構建代理工作流圖
- 管理對話記憶
"""

import logging
import time
import uuid
from typing import Annotated, Any, Dict, List, Optional, TypedDict, Union

from langchain.schema.messages import (AIMessage, FunctionMessage,
                                       HumanMessage, SystemMessage)
from langchain.tools import BaseTool, StructuredTool, tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

# 嘗試以多種方式匯入，以適應不同版本的 LangGraph API
try:
    # 新版本的 langgraph API
    from langgraph.prebuilt import ToolNode, tools_condition
except ImportError:
    try:
        # 嘗試替代匯入路徑
        from langgraph.prebuilt.tool_executor import ToolNode, tools_condition
    except ImportError:
        # 再嘗試其他可能的匯入路徑
        try:
            from langgraph.prebuilt.node import ToolNode, tools_condition
        except ImportError:
            # 最後提供一個備用實現
            logging.warning("找不到 LangGraph 的 ToolNode 和 tools_condition，使用自定義實現")

            class ToolNode:
                def __init__(self, tools):
                    self.tools = tools

                def __call__(self, state):
                    # 簡單的工具調用處理
                    tool_call = state["messages"][-1].additional_kwargs.get("tool_calls", [])
                    if not tool_call:
                        return {"messages": []}

                    tool_name = tool_call[0].get("function", {}).get("name")
                    tool_args = tool_call[0].get("function", {}).get("arguments", "{}")

                    for tool in self.tools:
                        if tool.name == tool_name:
                            result = tool(**eval(tool_args))
                            return {"messages": [FunctionMessage(content=str(result), name=tool_name)]}
                    return {"messages": []}

            def tools_condition(state):
                messages = state["messages"]
                last_message = messages[-1]
                if hasattr(last_message, "additional_kwargs") and "tool_calls" in last_message.additional_kwargs:
                    return "tools"
                return END

# 設置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 定義對話狀態
class ChatState(TypedDict):
    messages: Annotated[List[Union[AIMessage, HumanMessage, SystemMessage, FunctionMessage]], add_messages]
    video_id: str

# 創建搜索工具
@tool
def search_video_content(query: str, vector_store) -> str:
    """從視頻內容中搜索相關信息

    Args:
        query (str): 搜索查詢
        vector_store: 向量存儲對象

    Returns:
        str: 相關信息
    """
    logger.info(f"搜索視頻內容: '{query}'")
    try:
        docs = vector_store.similarity_search(query, k=3)
        if not docs:
            logger.warning("找不到相關信息")
            return "找不到相關信息。"

        content = "\n\n".join([f"段落 {i+1}: {doc.page_content}" for i, doc in enumerate(docs)])
        logger.info(f"找到 {len(docs)} 個相關文檔")
        return content
    except Exception as e:
        logger.error(f"搜索視頻內容時出錯: {str(e)}")
        return f"搜索出錯: {str(e)}"

# 構建 LLM 代理
def create_chat_agent(video_info: Dict[str, Any], vector_store):
    """創建基於視頻內容的對話代理

    Args:
        video_info (Dict[str, Any]): 視頻信息
        vector_store: 向量存儲對象

    Returns:
        StateGraph: 編譯後的對話代理圖
    """
    logger.info(f"為視頻 '{video_info.get('title', '未知視頻')}' 創建對話代理")

    # 創建系統提示
    system_prompt = f"""你是一個友好、專業的 AI 助手，專門解答關於以下視頻的問題：
    標題: {video_info.get('title', '未知')}
    作者: {video_info.get('author', '未知')}

    你可以使用搜索工具查找視頻中的相關內容來回答問題。
    如果用戶問的問題與視頻無關，請禮貌地提醒他們你的專長是討論這個特定視頻的內容。
    總是提供準確、有幫助、相關的信息，並引用視頻中的具體內容。
    使用繁體中文回應。
    """

    # 創建模型
    logger.info("初始化 ChatOpenAI 模型")
    try:
        # 嘗試使用推薦的參數
        model = ChatOpenAI(temperature=0.7, model="gpt-3.5-turbo")
    except TypeError:
        # 如果參數不匹配，嘗試用不同的參數格式
        try:
            model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)
        except TypeError:
            # 如果還是不行，使用最小化參數
            logger.warning("使用最小化參數初始化 ChatOpenAI")
            model = ChatOpenAI()

    # 創建搜索工具
    logger.info("創建視頻內容搜索工具")
    search_tool = StructuredTool.from_function(
        func=lambda query: search_video_content(query, vector_store),
        name="search_video_content",
        description="從視頻內容中搜索相關信息，用於回答關於視頻的問題"
    )

    # 定義代理節點
    def agent_node(state):
        """LLM 代理節點

        Args:
            state: 當前狀態

        Returns:
            Dict: 更新後的狀態
        """
        messages = state["messages"]

        # 如果是第一條消息，添加系統提示
        if len(messages) == 1 and isinstance(messages[0], HumanMessage):
            logger.info("添加系統提示")
            messages = [SystemMessage(content=system_prompt)] + messages

        # 調用模型
        logger.info("調用 LLM 處理消息")
        response = model.invoke(messages)
        return {"messages": [response]}

    # 定義工具節點
    logger.info("創建工具節點")
    tools_node = ToolNode([search_tool])

    # 創建圖
    logger.info("創建對話圖")
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
    logger.info("添加記憶體支持 (MemorySaver)")
    memory = MemorySaver()
    logger.info("記憶體初始化完成")

    # 編譯圖
    logger.info("編譯對話代理圖")
    return workflow.compile(checkpointer=memory)

# 生成新的對話 ID
def generate_thread_id() -> str:
    """生成新的對話 ID

    Returns:
        str: 對話 ID
    """
    thread_id = str(uuid.uuid4())
    logger.info(f"生成新的對話 ID: {thread_id}")
    return thread_id

# 調試 MemorySaver 功能
def debug_memory_state(memory_saver, thread_id: str) -> None:
    """調試 MemorySaver 的狀態和內容

    Args:
        memory_saver: MemorySaver 實例
        thread_id (str): 對話 ID
    """
    if not memory_saver:
        logger.warning("調試記憶體: 沒有有效的 MemorySaver 實例")
        return

    try:
        logger.info(f"調試記憶體: 檢查 MemorySaver 類型 {type(memory_saver)}")

        # 檢查 get 方法
        if hasattr(memory_saver, "get"):
            logger.info("調試記憶體: MemorySaver 有 get 方法")

            # 嘗試獲取記憶內容
            try:
                memory_content = memory_saver.get(thread_id)
                logger.info(f"調試記憶體: get() 返回類型 {type(memory_content)}")

                if memory_content is None:
                    logger.info("調試記憶體: 沒有找到對應 thread_id 的記憶")
                elif isinstance(memory_content, dict):
                    keys = list(memory_content.keys())
                    logger.info(f"調試記憶體: 字典包含的鍵: {keys}")

                    if "messages" in memory_content:
                        msg_count = len(memory_content["messages"])
                        logger.info(f"調試記憶體: 訊息數量: {msg_count}")

                        if msg_count > 0:
                            message_types = [type(msg).__name__ for msg in memory_content["messages"]]
                            logger.info(f"調試記憶體: 訊息類型: {message_types}")
                elif isinstance(memory_content, str):
                    logger.info(f"調試記憶體: 內容是字符串: {memory_content[:100]}...")
                elif isinstance(memory_content, list):
                    logger.info(f"調試記憶體: 列表長度: {len(memory_content)}")
                    if memory_content:
                        item_types = [type(item).__name__ for item in memory_content[:5]]
                        logger.info(f"調試記憶體: 前5個元素類型: {item_types}")
                else:
                    logger.info(f"調試記憶體: 未知類型內容: {str(memory_content)[:100]}...")
            except Exception as e:
                logger.error(f"調試記憶體: 獲取記憶內容時出錯: {str(e)}", exc_info=True)
        else:
            logger.warning("調試記憶體: MemorySaver 沒有 get 方法")

        # 檢查可用的屬性和方法
        memory_attrs = [attr for attr in dir(memory_saver) if not attr.startswith("_")]
        logger.info(f"調試記憶體: 可用屬性和方法: {memory_attrs}")

    except Exception as e:
        logger.error(f"調試記憶體: 檢查記憶功能時出錯: {str(e)}", exc_info=True)

# 使用代理進行對話
def chat_with_video(chat_agent, query: str, thread_id: str, video_id: str) -> str:
    """與視頻內容對話

    Args:
        chat_agent: 對話代理圖
        query (str): 用戶查詢
        thread_id (str): 對話 ID
        video_id (str): 視頻 ID

    Returns:
        str: AI 回應
    """
    logger.info(f"處理用戶查詢: '{query}'")
    logger.info(f"對話 ID: {thread_id}, 視頻 ID: {video_id}")

    # 設置配置
    config = {"configurable": {"thread_id": thread_id}}
    logger.info(f"配置線程 ID: {thread_id}")

    # 調試記憶體狀態
    if hasattr(chat_agent, "checkpointer"):
        logger.info("正在調試記憶體狀態...")
        debug_memory_state(chat_agent.checkpointer, thread_id)

    # 創建消息
    messages = [HumanMessage(content=query)]

    # 初始狀態
    state = {
        "messages": messages,
        "video_id": video_id
    }

    # 打印之前的對話歷史
    print_conversation_history(chat_agent, thread_id)

    # 打印當前的用戶查詢
    print(f"\n用戶: {query}")

    # 調用代理
    try:
        logger.info(f"使用線程 ID {thread_id} 調用對話代理")

        # 檢查是否是首次對話，如果是則可能需要初始化記憶
        if hasattr(chat_agent, "checkpointer") and chat_agent.checkpointer:
            try:
                # 獲取歷史數據，但不嘗試訪問其內容，只檢查是否為 None
                existing_history = chat_agent.checkpointer.get(thread_id)

                # 記錄獲取到的數據類型，幫助診斷問題
                logger.info(f"檢查記憶體: 獲取到的歷史數據類型: {type(existing_history)}")

                # 只檢查歷史是否存在，不嘗試訪問其內部結構
                if existing_history is None:
                    logger.info(f"首次對話，將初始化記憶體 (thread_id: {thread_id})")
            except Exception as e:
                # 捕獲所有可能的錯誤，但不影響主流程
                logger.warning(f"檢查記憶體時出錯: {str(e)}", exc_info=True)

        # 調用對話代理
        result = chat_agent.invoke(state, config=config)

        # 初始化預設回應
        response_text: str = "處理您的問題時出現系統錯誤，請重試。"

        # 檢查結果格式
        if not isinstance(result, dict) or "messages" not in result:
            logger.warning(f"代理返回的結果格式不是預期的: {type(result)}")
        else:
            # 獲取最後一條消息
            messages_list = result.get("messages", [])
            if not messages_list:
                logger.warning("代理返回的消息列表為空")
                response_text = "處理您的問題時出現系統錯誤，消息列表為空。"
            else:
                last_message = messages_list[-1]
                if isinstance(last_message, AIMessage):
                    # 明確轉換為字符串，以滿足類型檢查
                    content = last_message.content
                    if isinstance(content, str):
                        response_text = content
                    else:
                        response_text = str(content)
                    logger.info("成功生成回應")
                    logger.info(f"回應長度: {len(response_text)} 字符")
                else:
                    logger.warning(f"最後一條消息不是 AI 消息: {type(last_message)}")
                    # 確保返回的是字符串
                    response_text = str(last_message)

        # 再次調試記憶體狀態（回應後）
        if hasattr(chat_agent, "checkpointer"):
            logger.info("對話完成後調試記憶體狀態...")
            debug_memory_state(chat_agent.checkpointer, thread_id)

        # 打印新的回應
        print(f"AI: {response_text}")
        print("\n" + "="*50 + "\n")

        return response_text
    except Exception as e:
        logger.error(f"對話代理調用出錯: {str(e)}", exc_info=True)  # 添加完整的錯誤堆棧
        error_msg = f"處理您的問題時出錯: {str(e)}"
        print(f"系統錯誤: {error_msg}")
        return error_msg

# 獲取並打印對話歷史
def print_conversation_history(chat_agent, thread_id: str) -> None:
    """獲取並打印對話歷史

    Args:
        chat_agent: 對話代理圖
        thread_id (str): 對話 ID
    """
    try:
        # 使用 chat_agent 的 checkpointer 獲取對話歷史
        if hasattr(chat_agent, "checkpointer") and chat_agent.checkpointer:
            logger.info(f"嘗試從記憶體中檢索線程 ID: {thread_id} 的對話歷史")

            try:
                # 獲取歷史數據
                history = chat_agent.checkpointer.get(thread_id)

                # 詳細記錄獲取到的數據類型和結構，幫助診斷問題
                logger.info(f"獲取到的歷史數據類型: {type(history)}")

                # 檢查歷史是否存在
                if history is None:
                    logger.info(f"線程 ID {thread_id} 沒有歷史訊息")
                    print("\n" + "="*20 + " 新對話開始 " + "="*20)
                    return

                # 打印歷史數據的前200個字符作為診斷信息
                try:
                    if isinstance(history, (dict, list, str)):
                        history_preview = str(history)[:200]
                        logger.info(f"歷史數據預覽: {history_preview}...")
                    else:
                        logger.info(f"歷史數據類型: {type(history)}，無法獲取預覽")
                except Exception as preview_error:
                    logger.warning(f"無法獲取歷史數據預覽: {str(preview_error)}")

                # 檢查 history 是字典還是其他類型
                if isinstance(history, dict):
                    # 嘗試獲取 messages 字段
                    if "messages" in history and isinstance(history["messages"], list):
                        # 正常的字典格式
                        messages = history["messages"]
                        msg_count = len(messages)
                        logger.info(f"找到 {msg_count} 條歷史訊息")

                        if msg_count > 0:
                            print("\n" + "="*20 + " 對話歷史 " + "="*20)

                            for i, msg in enumerate(messages):
                                if isinstance(msg, HumanMessage):
                                    print(f"\n用戶: {msg.content}")
                                elif isinstance(msg, AIMessage):
                                    print(f"AI: {msg.content}")
                                elif isinstance(msg, SystemMessage):
                                    print(f"系統: {msg.content}")
                                elif isinstance(msg, FunctionMessage):
                                    print(f"工具 ({msg.name}): {msg.content}")
                                else:
                                    print(f"未知訊息類型 ({type(msg)}): {str(msg)}")

                            print("="*50)
                        else:
                            print("\n" + "="*20 + " 新對話開始 " + "="*20)
                    else:
                        # 可能是其他格式的字典
                        logger.warning("歷史數據是字典類型，但沒有有效的 messages 字段")
                        print("\n" + "="*20 + " 新對話開始 " + "="*20)
                        print(f"注意：對話歷史格式不正確，已重置對話")

                elif isinstance(history, str):
                    # 如果 history 是字符串，可能是序列化的數據或錯誤消息
                    logger.warning(f"歷史數據是字符串類型: '{history[:100]}...'")
                    print("\n" + "="*20 + " 新對話開始 " + "="*20)
                    print(f"注意：對話歷史格式不正確，已重置對話")

                elif isinstance(history, list):
                    # 如果直接是訊息列表
                    msg_count = len(history)
                    logger.info(f"找到 {msg_count} 條歷史訊息 (列表格式)")

                    if msg_count > 0:
                        print("\n" + "="*20 + " 對話歷史 " + "="*20)

                        for i, msg in enumerate(history):
                            if isinstance(msg, HumanMessage):
                                print(f"\n用戶: {msg.content}")
                            elif isinstance(msg, AIMessage):
                                print(f"AI: {msg.content}")
                            elif isinstance(msg, SystemMessage):
                                print(f"系統: {msg.content}")
                            elif isinstance(msg, FunctionMessage):
                                print(f"工具 ({msg.name}): {msg.content}")
                            else:
                                print(f"未知訊息類型 ({type(msg)}): {str(msg)}")

                        print("="*50)
                    else:
                        print("\n" + "="*20 + " 新對話開始 " + "="*20)

                else:
                    # 其他未預期的數據類型
                    logger.warning(f"歷史數據類型不是預期的字典或列表: {type(history)}")
                    print("\n" + "="*20 + " 新對話開始 " + "="*20)
                    print(f"注意：對話歷史格式不支持，已重置對話")

            except (KeyError, TypeError, IndexError) as e:
                # 捕獲可能的索引或鍵錯誤
                logger.error(f"解析歷史數據時出錯: {str(e)}", exc_info=True)
                print("\n" + "="*20 + " 新對話開始 " + "="*20)
                print(f"注意：獲取對話歷史時出錯，已重置對話")
        else:
            logger.warning("對話代理沒有有效的記憶體檢查點")
            print("\n" + "="*20 + " 新對話開始 " + "="*20)
    except Exception as e:
        logger.error(f"獲取對話歷史時出錯: {str(e)}", exc_info=True)
        print(f"\n系統訊息: 獲取對話歷史時出錯: {str(e)}")
        print("="*50)