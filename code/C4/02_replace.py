import os
import re
import time
import requests
from langchain.schema import Document
from langchain_deepseek import ChatDeepSeek
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import logging
from langchain_community.chat_models import ChatTongyi
from dotenv import load_dotenv
import json

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 设置Hugging Face镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 加载环境变量
load_dotenv()

def get_bilibili_video_info_api(bvid):
    """使用B站官方API获取视频信息"""
    url = f"https://api.bilibili.com/x/web-interface/view?bvid={bvid}"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Referer': 'https://www.bilibili.com/',
    }
    
    try:
        logger.info(f"正在使用B站API获取视频信息: {url}")
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        if data.get('code') == 0:
            video_data = data['data']
            title = video_data['title']
            author = video_data['owner']['name']
            view_count = video_data['stat']['view']
            length = video_data['duration']  # 时长(秒)
            
            return {
                'title': title,
                'author': author,
                'view_count': view_count,
                'length': length,
                'url': f"https://www.bilibili.com/video/{bvid}"
            }
        else:
            logger.warning(f"B站API返回错误: {data.get('message', '未知错误')}")
            return None
            
    except Exception as e:
        logger.error(f"使用B站API获取视频信息失败: {str(e)}")
        return None

# 1. 初始化视频数据
video_urls = [
    "https://www.bilibili.com/video/BV1Bo4y1A7FU",
    "https://www.bilibili.com/video/BV1ug4y157xA",
    "https://www.bilibili.com/video/BV1yh411V7ge",
]

bili = []
docs = []

for url in video_urls:
    try:
        # 从URL中提取BVID
        bvid = url.split('/')[-1]
        
        # 获取视频信息
        video_info = get_bilibili_video_info_api(bvid)
        
        if video_info:
            # 创建Document对象
            content = f"标题: {video_info['title']}\n作者: {video_info['author']}\n观看次数: {video_info['view_count']}\n时长: {video_info['length']}秒"
            
            metadata = {
                'title': video_info['title'],
                'author': video_info['author'],
                'view_count': int(video_info['view_count']),
                'length': int(video_info['length']),
                'source': url
            }
            
            doc = Document(page_content=content, metadata=metadata)
            docs.append(doc)
            bili.append(doc)
            logger.info(f"成功加载视频: {video_info['title']}")
        else:
            logger.warning(f"无法获取视频信息: {url}")
        
        # 避免请求过于频繁
        time.sleep(1)
        
    except Exception as e:
        logger.error(f"处理视频 {url} 时出错: {str(e)}")

if not bili:
    logger.error("没有成功加载任何视频，程序退出")
    exit()

logger.info(f"成功加载 {len(bili)} 个视频")

# 2. 创建向量存储
try:
    logger.info("正在加载嵌入模型...")
    embed_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
    
    logger.info("正在创建向量存储...")
    vectorstore = Chroma.from_documents(docs, embed_model)
    logger.info("向量存储创建成功")
except Exception as e:
    logger.error(f"创建向量存储失败: {str(e)}")
    exit()

# 3. 配置元数据字段信息
metadata_field_info = [
    AttributeInfo(
        name="title",
        description="视频标题（字符串）",
        type="string",
    ),
    AttributeInfo(
        name="author",
        description="视频作者（字符串）",
        type="string",
    ),
    AttributeInfo(
        name="view_count",
        description="视频观看次数（整数）",
        type="integer",
    ),
    AttributeInfo(
        name="length",
        description="视频长度（整数，单位为秒）",
        type="integer"
    )
]

# 4. 创建自查询检索器
try:
    logger.info("正在初始化通义千问模型...")
    llm = ChatTongyi(model="qwen3-max", temperature=0.7)
    
    logger.info("正在创建自查询检索器...")
    retriever = SelfQueryRetriever.from_llm(
        llm=llm,
        vectorstore=vectorstore,
        document_contents="记录视频标题、作者、观看次数和时长等信息的视频元数据",
        metadata_field_info=metadata_field_info,
        enable_limit=True,
        verbose=True
    )
    logger.info("自查询检索器创建成功")
except Exception as e:
    logger.error(f"创建检索器失败: {str(e)}")
    exit()

# 5. 执行查询示例
queries = [
    "时间最短的视频",
    "时长大于600秒的视频",
    "观看次数最多的视频",
    "作者是'Datawhale'的视频"
]

for query in queries:
    print(f"\n{'='*60}")
    print(f"查询: '{query}'")
    print(f"{'='*60}")
    
    try:
        results = retriever.invoke(query)
        
        if results:
            print(f"\n找到 {len(results)} 个匹配结果:")
            for i, doc in enumerate(results, 1):
                print(f"\n结果 #{i}")
                print(f"标题: {doc.metadata.get('title', '未知标题')}")
                print(f"作者: {doc.metadata.get('author', '未知作者')}")
                print(f"观看次数: {doc.metadata.get('view_count', '未知')}")
                print(f"时长: {doc.metadata.get('length', '未知')}秒")
                print("-"*50)
        else:
            print("\n未找到匹配的视频")
            
    except Exception as e:
        print(f"查询执行失败: {str(e)}")

print("\n程序执行完毕")