"""
YouTube 工具模塊

該模塊負責處理 YouTube 相關功能，包括：
- 從 URL 提取視頻 ID
- 獲取視頻基本信息
- 提取和處理視頻字幕
- 格式化字幕數據
"""

import json
import logging
import re
import time
from typing import Any, Dict, List, Union

import requests

# 嘗試匯入，如果失敗則給出更明確的錯誤信息
try:
    from pytube import YouTube
    # 從合適的模組匯入 API
    from youtube_transcript_api._api import YouTubeTranscriptApi
except ImportError:
    raise ImportError("請確保已安裝 pytube 和 youtube_transcript_api 套件：pip install pytube youtube-transcript-api")

# 設置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_video_id(youtube_url: str) -> str:
    """從 YouTube URL 中提取視頻 ID

    Args:
        youtube_url (str): YouTube 視頻 URL

    Returns:
        str: 視頻 ID

    Raises:
        ValueError: 當無法從 URL 中提取視頻 ID 時
    """
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
    """獲取視頻的基本信息，使用冗餘方法以提高穩健性

    Args:
        video_id (str): YouTube 視頻 ID

    Returns:
        dict: 包含視頻標題、作者、長度等信息的字典

    Raises:
        Exception: 當獲取視頻信息失敗時
    """
    # 方法 1: 使用 pytube
    def get_info_via_pytube():
        logger.info(f"使用 pytube 獲取視頻 {video_id} 的信息")
        yt = YouTube(f"https://www.youtube.com/watch?v={video_id}", use_oauth=False, allow_oauth_cache=True)

        # 提取視頻基本信息
        info = {
            "title": yt.title,
            "author": yt.author,
            "length": yt.length,
            "publish_date": yt.publish_date,
            "thumbnail_url": yt.thumbnail_url,
            "views": yt.views
        }

        logger.info(f"成功使用 pytube 獲取視頻信息: {yt.title}")
        return info

    # 方法 2: 使用 YouTube oEmbed API (無需 API key)
    def get_info_via_oembed():
        logger.info(f"使用 oEmbed API 獲取視頻 {video_id} 的信息")
        oembed_url = f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json"

        response = requests.get(oembed_url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            # 嘗試從 oEmbed 獲取可用信息
            info = {
                "title": data.get("title", "未知標題"),
                "author": data.get("author_name", "未知作者"),
                "thumbnail_url": data.get("thumbnail_url", ""),
                # oEmbed 不提供以下數據，設置默認值
                "length": 0,
                "publish_date": None,
                "views": 0
            }

            logger.info(f"成功使用 oEmbed API 獲取視頻信息: {info['title']}")
            return info
        else:
            logger.warning(f"oEmbed API 請求失敗: {response.status_code}")
            return None

    # 方法 3: 使用視頻頁面元數據 (備用方法)
    def get_info_via_scraping():
        logger.info(f"嘗試從頁面抓取視頻 {video_id} 的信息")
        try:
            url = f"https://www.youtube.com/watch?v={video_id}"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36',
            }
            response = requests.get(url, headers=headers, timeout=10)

            if response.status_code != 200:
                logger.warning(f"頁面請求失敗: {response.status_code}")
                return None

            # 嘗試從 HTML 中提取視頻標題
            title_match = re.search(r'<title>(.*?)</title>', response.text)
            title = title_match.group(1).replace(' - YouTube', '') if title_match else "未知標題"

            # 嘗試提取作者名
            author_match = re.search(r'"author":"(.*?)"', response.text)
            author = author_match.group(1) if author_match else "未知作者"

            # 提取縮略圖 URL
            thumbnail_url = f"https://i.ytimg.com/vi/{video_id}/maxresdefault.jpg"

            info = {
                "title": title,
                "author": author,
                "thumbnail_url": thumbnail_url,
                "length": 0,
                "publish_date": None,
                "views": 0
            }

            logger.info(f"成功從頁面抓取視頻信息: {title}")
            return info
        except Exception as e:
            logger.error(f"抓取視頻信息時出錯: {str(e)}")
            return None

    # 嘗試多種方法獲取視頻信息，從最可靠的方法開始
    methods = [
        get_info_via_pytube,
        get_info_via_oembed,
        get_info_via_scraping
    ]

    errors = []
    for method in methods:
        try:
            info = method()
            if info:
                return info
        except Exception as e:
            error_msg = f"{method.__name__}: {str(e)}"
            logger.error(error_msg)
            errors.append(error_msg)
            time.sleep(1)  # 添加短暫延遲，避免頻繁請求

    # 所有方法都失敗時，創建一個基本信息對象
    logger.warning("所有獲取視頻信息的方法都失敗，使用基本信息")
    basic_info = {
        "title": f"YouTube 視頻 ({video_id})",
        "author": "未知作者",
        "thumbnail_url": f"https://i.ytimg.com/vi/{video_id}/maxresdefault.jpg",
        "length": 0,
        "publish_date": None,
        "views": 0,
        "error": f"無法獲取完整視頻信息: {'; '.join(errors)}"
    }

    # 通知使用者資訊獲取過程中發生問題，但仍提供基本功能
    logger.info(f"使用基本視頻信息以繼續處理: {basic_info['title']}")
    return basic_info

def get_video_transcript(video_id: str, language: str = "en") -> Any:
    """獲取視頻字幕

    Args:
        video_id (str): YouTube 視頻 ID
        language (str, optional): 字幕語言代碼，默認為 'en'

    Returns:
        Any: 包含字幕段落的物件或列表

    Raises:
        Exception: 當獲取字幕失敗時
    """
    try:
        logger.info(f"正在獲取視頻 {video_id} 的字幕，目標語言: {language}")
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

        # 嘗試獲取指定語言的字幕
        try:
            transcript = transcript_list.find_transcript([language])
            logger.info(f"找到 {language} 語言的字幕")
        except:
            # 如果失敗，嘗試獲取可用的字幕並翻譯
            logger.info(f"找不到 {language} 語言的字幕，嘗試獲取英文字幕並翻譯")
            transcript = transcript_list.find_transcript(['en'])
            if language != 'en':
                logger.info(f"將英文字幕翻譯為 {language}")
                transcript = transcript.translate(language)

        # 獲取字幕內容
        result = transcript.fetch()
        logger.info(f"成功獲取字幕，共 {len(result)} 個片段")
        return result
    except Exception as e:
        logger.error(f"獲取視頻字幕時出錯: {str(e)}")
        raise Exception(f"獲取視頻字幕時出錯: {str(e)}")

def format_transcript(transcript: list) -> dict:
    """格式化字幕數據

    Args:
        transcript (list): 原始字幕數據，可能是字典列表或 FetchedTranscriptSnippet 物件列表

    Returns:
        dict: 格式化後的字幕數據，包含完整文本和分段信息
    """
    logger.info("正在格式化字幕數據")
    formatted_text = ""
    segments = []

    for segment in transcript:
        # 檢查物件類型，兼容舊版和新版 API
        if hasattr(segment, 'text') and hasattr(segment, 'start') and hasattr(segment, 'duration'):
            # 新版 API 返回的是物件
            text = segment.text
            start = segment.start
            duration = segment.duration
        elif isinstance(segment, dict):
            # 舊版 API 返回的是字典
            text = segment['text']
            start = segment['start']
            duration = segment['duration']
        else:
            # 嘗試將物件轉換為字典（兼容性處理）
            try:
                segment_dict = segment.to_dict() if hasattr(segment, 'to_dict') else segment.__dict__
                text = segment_dict.get('text', str(segment))
                start = segment_dict.get('start', 0)
                duration = segment_dict.get('duration', 0)
            except Exception as e:
                logger.warning(f"無法處理字幕片段，使用默認值: {str(e)}")
                text = str(segment)
                start = 0
                duration = 0

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

    result = {
        "full_text": formatted_text.strip(),
        "segments": segments
    }

    logger.info(f"字幕格式化完成，總字數: {len(formatted_text.split())}")
    return result

def format_timestamp(seconds: float) -> str:
    """將秒數轉換為時:分:秒格式

    Args:
        seconds (float): 秒數

    Returns:
        str: 格式化的時間戳 (HH:MM:SS 或 MM:SS)
    """
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)

    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    else:
        return f"{minutes:02d}:{seconds:02d}"