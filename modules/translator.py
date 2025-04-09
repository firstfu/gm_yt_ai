"""
翻譯模塊

該模塊負責處理字幕翻譯：
- 使用 LLM 進行高質量翻譯
- 保持時間戳對應關係
"""

import logging
import time
from typing import Any, Dict, List

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# 設置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def translate_segments(segments: List[Dict[str, Any]], target_language: str = "中文", batch_size: int = 10) -> List[Dict[str, Any]]:
    """翻譯字幕分段

    Args:
        segments (List[Dict[str, Any]]): 字幕段落列表
        target_language (str, optional): 目標語言，默認為 "中文"
        batch_size (int, optional): 批處理大小，默認為 10

    Returns:
        List[Dict[str, Any]]: 包含翻譯的字幕段落列表
    """
    logger.info(f"開始將字幕翻譯為 {target_language}，共 {len(segments)} 個段落")

    # 創建 LLM
    llm = ChatOpenAI(temperature=0.1, model="gpt-3.5-turbo")

    # 創建翻譯提示模板
    prompt_template = PromptTemplate(
        input_variables=["text"],
        template=f"將以下文本準確地翻譯成{target_language}，保持原意。翻譯應自然、流暢，適合口語表達。\n\n文本：{{text}}\n\n{target_language}翻譯："
    )

    # 創建翻譯鏈
    translate_chain = LLMChain(llm=llm, prompt=prompt_template)

    # 翻譯每個分段
    translated_segments = []
    total_segments = len(segments)

    # 批處理翻譯以提高效率
    for i in range(0, total_segments, batch_size):
        batch = segments[i:i+batch_size]
        logger.info(f"正在翻譯批次 {i//batch_size + 1}/{(total_segments-1)//batch_size + 1}，共 {len(batch)} 個段落")

        batch_translations = []
        for segment in batch:
            try:
                # 獲取翻譯
                translation = translate_chain.run(segment["text"]).strip()

                # 創建帶有翻譯的新段落
                translated_segment = segment.copy()
                translated_segment["translation"] = translation
                batch_translations.append(translated_segment)

            except Exception as e:
                logger.error(f"翻譯段落時出錯: {str(e)}")
                # 如果翻譯失敗，使用原文作為翻譯
                translated_segment = segment.copy()
                translated_segment["translation"] = segment["text"] + " (翻譯失敗)"
                batch_translations.append(translated_segment)

            # 短暫暫停以避免 API 限制
            time.sleep(0.2)

        # 將批次結果添加到總結果
        translated_segments.extend(batch_translations)

    logger.info(f"字幕翻譯完成，共 {len(translated_segments)} 個段落")
    return translated_segments

def batch_translate_text(texts: List[str], target_language: str = "中文") -> List[str]:
    """批量翻譯文本

    Args:
        texts (List[str]): 要翻譯的文本列表
        target_language (str, optional): 目標語言，默認為 "中文"

    Returns:
        List[str]: 翻譯後的文本列表
    """
    logger.info(f"開始批量翻譯 {len(texts)} 個文本到 {target_language}")

    # 創建 LLM
    llm = ChatOpenAI(temperature=0.1, model="gpt-3.5-turbo")

    # 創建翻譯提示模板
    prompt_template = PromptTemplate(
        input_variables=["text"],
        template=f"將以下文本準確地翻譯成{target_language}，保持原意。翻譯應自然、流暢，適合閱讀。\n\n文本：{{text}}\n\n{target_language}翻譯："
    )

    # 創建翻譯鏈
    translate_chain = LLMChain(llm=llm, prompt=prompt_template)

    # 翻譯每個文本
    translations = []
    for i, text in enumerate(texts):
        try:
            logger.info(f"正在翻譯文本 {i+1}/{len(texts)}")
            translation = translate_chain.run(text).strip()
            translations.append(translation)
        except Exception as e:
            logger.error(f"翻譯文本時出錯: {str(e)}")
            # 如果翻譯失敗，使用原文
            translations.append(text + " (翻譯失敗)")

        # 短暫暫停以避免 API 限制
        time.sleep(0.5)

    logger.info("批量翻譯完成")
    return translations