# 添加文件路径
TEXT_PATH = 'hongloumeng.txt'
MODEL_PATH = 'models/bge-large-zh-v1.5'
DB_PATH = 'HLM_chroma_db'
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50
###########################################
import os
from langchain_community.document_loaders import TextLoader

# 1 加载书籍文件，采用utf-8编码
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
loader = TextLoader(TEXT_PATH, encoding='utf-8')
docs = loader.load()
# 2 分块 按第x回
import re
full_text = docs[0].page_content
chapters = re.split(r'(第[一二三四五六七八九十百零\d]+回[^。！？\n]*[。！？]?)', full_text)
clearn_chapters = []
for i in range(1, len(chapters), 2):
    if i+1 < len(chapters):
        title = chapters[i].strip()
        content = chapters[i+1].strip()
        clearn_chapters.append((title, title + '\n' + content))

# 备选，如果书籍未按第x回划分则整体作为一块
# if not clearn_chapters:
#     clearn_chapters = [('full_text', full_text)]

# 3 转Document列表
from langchain_core.documents import Document
chapter_docs = [
    Document(page_content=chap, metadata={'source': 'hongloumeng.txt',
                                          'chapter_title': title})
    for title, chap in clearn_chapters
]

# 4 按语义分小段，每段400字
from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=[
        "\n\n",  # 优先按空行分（保留段落）
        "。", "！", "？",
        "\n",  # 标题后通常有换行
        "；", "，", ""
    ],
    is_separator_regex=False
)
chunks = text_splitter.split_documents(chapter_docs)
print(f'create:{len(chunks)}个段落')

# 5 load embeddings模型
from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(
    model_name=MODEL_PATH,
    model_kwargs={"device": "cpu"},
    encode_kwargs={'batch_size': 32}
)

# 6 create chroma vectordb
from langchain_chroma import Chroma
db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=DB_PATH
)
print('over')
