"""
文本處理模塊

該模塊負責處理文本分割和向量化：
- 將字幕文本分割成較小的塊
- 創建和管理向量存儲
- 處理相似性搜索
"""

import logging
import os
from typing import List, Optional

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

# 設置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def split_and_vectorize_transcript(transcript_text: str, video_id: str, chunk_size: int = 1000, chunk_overlap: int = 100) -> FAISS:
    """將字幕文本分割並向量化

    Args:
        transcript_text (str): 字幕文本
        video_id (str): YouTube 視頻 ID
        chunk_size (int, optional): 文本塊大小，默認為 1000
        chunk_overlap (int, optional): 文本塊重疊大小，默認為 100

    Returns:
        FAISS: 向量存儲對象
    """
    logger.info(f"開始處理視頻 {video_id} 的字幕文本")

    # 創建文本分割器
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    # 將文本分割成文檔
    logger.info(f"使用塊大小 {chunk_size} 和重疊大小 {chunk_overlap} 分割文本")
    docs = text_splitter.create_documents([transcript_text])
    logger.info(f"文本已分割為 {len(docs)} 個文檔")

    # 為每個文檔添加元數據
    for doc in docs:
        doc.metadata = {"source": f"youtube_video_{video_id}"}

    # 創建嵌入模型
    logger.info("創建 OpenAI 嵌入模型")
    embeddings = OpenAIEmbeddings()

    # 創建向量存儲
    logger.info("創建 FAISS 向量存儲")
    vector_store = FAISS.from_documents(docs, embeddings)

    # 保存向量存儲
    save_dir = f"data/vector_stores/{video_id}"
    os.makedirs(save_dir, exist_ok=True)
    vector_store.save_local(save_dir)
    logger.info(f"向量存儲已保存到 {save_dir}")

    return vector_store

def create_persistent_vector_store(video_id: str, documents: List[Document], embeddings=None) -> FAISS:
    """創建持久化向量存儲

    Args:
        video_id (str): YouTube 視頻 ID
        documents (List[Document]): 文檔列表
        embeddings: 嵌入模型，默認為 OpenAIEmbeddings

    Returns:
        FAISS: 向量存儲對象
    """
    logger.info(f"為視頻 {video_id} 創建持久化向量存儲")

    # 如果未提供嵌入模型，使用默認模型
    if embeddings is None:
        embeddings = OpenAIEmbeddings()

    # 使用 FAISS 創建向量存儲
    vector_store = FAISS.from_documents(documents, embeddings)

    # 保存向量存儲
    save_dir = f"data/vector_stores/{video_id}"
    os.makedirs(save_dir, exist_ok=True)
    vector_store.save_local(save_dir)
    logger.info(f"向量存儲已保存到 {save_dir}")

    return vector_store

def load_vector_store(video_id: str, embeddings=None) -> Optional[FAISS]:
    """加載持久化向量存儲

    Args:
        video_id (str): YouTube 視頻 ID
        embeddings: 嵌入模型，默認為 OpenAIEmbeddings

    Returns:
        Optional[FAISS]: 向量存儲對象，如果加載失敗則返回 None
    """
    logger.info(f"嘗試加載視頻 {video_id} 的向量存儲")

    # 如果未提供嵌入模型，使用默認模型
    if embeddings is None:
        embeddings = OpenAIEmbeddings()

    # 嘗試加載現有的向量存儲
    save_dir = f"data/vector_stores/{video_id}"

    try:
        if os.path.exists(save_dir):
            vector_store = FAISS.load_local(save_dir, embeddings)
            logger.info(f"成功從 {save_dir} 加載向量存儲")
            return vector_store
        else:
            logger.warning(f"找不到向量存儲 {save_dir}")
            return None
    except Exception as e:
        logger.error(f"加載向量存儲時出錯: {str(e)}")
        return None