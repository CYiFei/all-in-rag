import os
import nltk
import sys

# 将当前文件所在目录设置为工作目录
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# === 重要：必须在导入任何库之前设置NLTK_DATA ===
os.environ['NLTK_DATA'] = os.path.expanduser('~/nltk_data')
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
nltk.data.path = [os.environ['NLTK_DATA']] + nltk.data.path

# 确保NLTK数据目录存在
nltk_data_dir = os.environ['NLTK_DATA']
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)


# === 确认文件路径 ===
markdown_path = "../../data/C1/markdown/easy-rl-chapter1.md"

# 检查文件是否存在
if not os.path.exists(markdown_path):
    # 尝试相对路径和绝对路径
    abs_path = os.path.abspath(markdown_path)
    if not os.path.exists(abs_path):
        print(f"❌ 文件不存在: {markdown_path}")
        print(f"尝试的绝对路径: {abs_path}")
        print("请检查文件路径是否正确")
        sys.exit(1)
    else:
        markdown_path = abs_path

print(f"✅ 文件路径确认: {markdown_path}")

# === 使用更可靠的TextLoader ===
from langchain_community.document_loaders import TextLoader

print("开始加载markdown文件...")
try:
    loader = TextLoader(markdown_path, encoding='utf-8')
    docs = loader.load()
    print(f"✅ 文本加载成功，共 {len(docs)} 个文档片段")
except Exception as e:
    print(f"❌ 文本加载失败: {e}")
    print("尝试使用其他编码...")
    try:
        loader = TextLoader(markdown_path, encoding='gbk')
        docs = loader.load()
        print(f"✅ 使用gbk编码成功加载")
    except Exception as e2:
        print(f"❌ 无法加载文件: {e2}")
        sys.exit(1)

# 文本分块
from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)
chunks = text_splitter.split_documents(docs)
print(f"✅ 文本分块完成，共 {len(chunks)} 个文本块")

# 模型加载
print("开始下载模型权重文件...")
from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)
print("✅ 模型权重文件下载完成！")

# 向量存储
from langchain_core.vectorstores import InMemoryVectorStore
vectorstore = InMemoryVectorStore(embeddings)
vectorstore.add_documents(chunks)

# 提示词模板
from langchain_core.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_template("""请根据下面提供的上下文信息来回答问题。
请确保你的回答完全基于这些上下文。
如果上下文中没有足够的信息来回答问题，请直接告知：“抱歉，我无法根据提供的上下文找到相关信息来回答此问题。”

上下文:
{context}

问题: {question}

回答:""")

# 配置大语言模型
from langchain_community.chat_models import ChatTongyi
from dotenv import load_dotenv

# 加载环境变量（已通过验证脚本确认）
load_dotenv()
llm = ChatTongyi(model="qwen3-max", temperature=0.7)

# 用户查询
question = "文中举了哪些例子？"

# 查询相关文档
retrieved_docs = vectorstore.similarity_search(question, k=3)
docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

# 生成回答
answer = llm.invoke(prompt.format(question=question, context=docs_content))
print("\n" + "="*50)
print("问题:", question)
print("回答:")
print(answer)
print("="*50)