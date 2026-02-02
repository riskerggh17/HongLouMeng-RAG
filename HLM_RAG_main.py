import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
os.environ["HF_HUB_OFFLINE"] = "1"  # 👈 强制离线模式
# 1 获取qwen的API
load_dotenv()
llm = ChatOpenAI(
    model='qwen-max',
    api_key=os.getenv("Qwen_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    temperature=0.0
)

# 2 文件路径
MODEL_PATH = 'models/bge-large-zh-v1.5'
DB_PATH = 'HLM_chroma_db'
RERANKER_PATH = "models/bge-reranker-v2-m3"
TOP_K= 8
# 3 load embeddings model
from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(
    model_name=MODEL_PATH,
    model_kwargs={"device": "cpu"},
    encode_kwargs={'batch_size': 32}
)


# 4 load Chromadb
from langchain_chroma import Chroma
vectorstore = Chroma(persist_directory=DB_PATH,
                     embedding_function=embeddings)


# 5 加载重排序模型
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
base_retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
tokenizer = AutoTokenizer.from_pretrained(RERANKER_PATH, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(RERANKER_PATH, local_files_only=True)
model.eval()  # 推理模式

def reranker_docs(query: str, docs: list, top_n: int = 2):
    """重排序文档"""
    if not docs:
        return []

    texts = [doc.page_content for doc in docs]
    pairs = [[query, text] for text in texts]

    with torch.no_grad():
        inputs = tokenizer(
            pairs,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512
        )
        scores = model(**inputs).logits.view(-1).float().tolist()
    # 按分数降序排序
    scored_docs = list(zip(docs, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, score in scored_docs[:top_n]]


# 6 构造prompt
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "你是一个严格受限的《红楼梦》问答引擎，仅能基于用户提供的原文片段作答。\n"
     "请遵守以下不可违背的规则：\n\n"
     "1. **回答条件**：仅当同时满足以下两点时，才允许回答：\n"
     "   (a) 问题的答案**逐字、完整地出现在提供的某一段原文中**；\n"
     "   (b) 该原文段落**独立包含全部必要信息**，无需结合其他段落或推理。\n\n"
     "2. **回答格式**：\n"
     "   - 若满足条件，用**最简短的一句话**直接输出答案，**不得添加任何解释、修饰、标点或引号**；\n"
     "   - 若不满足条件，**必须且只能输出以下11个字**：\n"
     "       根据提供的资料无法确定\n\n"
     "3. **绝对禁止行为**（违反即错误）：\n"
     "   - 使用常识、历史知识、人物关系推断（如‘王夫人是宝玉母亲’）；\n"
     "   - 总结、概括、改写原文（如‘棺木是好木材’而非‘樯木’）；\n"
     "   - 回答部分信息（如只说‘通灵宝玉’而漏掉‘莫失莫忘，仙寿恒昌’）；\n"
     "   - 输出任何额外文字，包括‘。’、‘！’、空格、星号、说明等。\n\n"
     "4. **特别强调**：\n"
     "   - 即使你 100% 确信答案正确，只要原文未**逐字写出**，就必须回答‘根据提供的资料无法确定’；\n"
     "   - ‘无法确定’不是失败，而是系统设计的核心要求。"
     ),
    ("human", "问题：{question}\n\n相关原文：\n{context}")
])


# 保持格式统一
def format_docs(docs):
    # 移除多余空格，保留原始文本
    return "\n\n".join([
        f"[出自：{d.metadata.get('chapter_title', '未知回目')}]\n{d.page_content.strip()}"
        for d in docs
    ])


# 7 构造rag链
def rag_with_rerank(question: str) -> str:
    raw_docs = base_retriever.invoke(question)
    ranked_docs = reranker_docs(question, raw_docs)
    context = format_docs(ranked_docs)
    messages = prompt.invoke({"question": question, "context": context})
    response = llm.invoke(messages)
    return StrOutputParser().invoke(response)


# 8 启动问答系统
print("\n✅ 《红楼梦》本地问答系统已就绪！")
print("输入问题（输入 'quit' 退出）：")
# 主循环
while True:
    query = input('<<<:').strip()
    if query.lower() in ['quit', 'exit', 'q']:
        break
    if not query:
        continue
    try:
        answer = rag_with_rerank(query)
        print(f'>>>：{answer}')
    except Exception as e:
        print(f'>>>发生错误：{e}')
