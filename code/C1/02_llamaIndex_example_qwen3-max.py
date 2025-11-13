import os

# 将当前文件所在目录设置为工作目录
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings 
from llama_index.llms.dashscope import DashScope  # 正确的导入路径
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# 加载环境变量
load_dotenv()

# 确保已安装必要的包: pip install llama-index-llms-dashscope

# 设置通义千问3模型
Settings.llm = DashScope(
    model_name="qwen3-max",  # 通义千问最大版本
    api_key=os.getenv("DASHSCOPE_API_KEY"),  # 通义千问API Key
    temperature=0.7,
    max_tokens=2048
)

# 设置嵌入模型
Settings.embed_model = HuggingFaceEmbedding("BAAI/bge-small-zh-v1.5")

# 加载文档
print("开始加载markdown文件...")
docs = SimpleDirectoryReader(
    input_files=["../../data/C1/markdown/easy-rl-chapter1.md"]
).load_data()

print(f"成功加载 {len(docs)} 个文档")

# 创建索引
print("创建向量索引...")
index = VectorStoreIndex.from_documents(docs)

# 创建查询引擎
query_engine = index.as_query_engine()

# 显示提示词模板（可选）
print("\n使用的提示词模板:")
print(query_engine.get_prompts())

# 执行查询
print("\n执行查询: '文中举了哪些例子?'")
response = query_engine.query("文中举了哪些例子?")
print("\n回答:")
print(response)