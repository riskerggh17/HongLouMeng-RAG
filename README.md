# HongLouMeng-RAG 📚

### **📚 项目简介**

**HongLouMeng-RAG** 是一个面向《红楼梦》的本地化检索增强生成（RAG）问答系统。
它通过语义检索从原著中精准定位相关段落，并严格遵循 **“逐字回答”原则**：

> **仅当答案完整、原样出现在原文中时，才予以返回；否则一律输出“根据提供的资料无法确定”。**

本系统采用 `bge-large-zh-v1.5` 进行中文向量化，结合 `bge-reranker-v2-m3` 重排序提升检索精度，所有文本处理与检索过程均在本地完成，无需联网。
最终答案由大模型（Qwen API）基于检索结果生成，但受严格规则约束，**杜绝幻觉、推理与主观解读**，确保每一条回答均可在原著中直接验证。

适用于文学研究、教学辅助、文本考证等对**准确性与忠实性**要求极高的场景。

### **✨ 核心特点**

- ✅ **零幻觉设计**：不总结、不推断、不改写，只输出原文存在的内容
- ✅ **本地化检索**：Embedding与重排序模型允许完全离线运行
- ✅ **回目级分块**：按“第X回”结构切分，保留上下文完整性
- ✅ **中文优化**：专为古典白话文设计的语义理解流程

1.项目结构：

```python
HongLouMeng-RAG/
├── hongloumeng.txt                 ← 你的文本（可提供下载链接）
├── models/                         ← 手动下载的模型
│   ├── bge-large-zh-v1.5/
│   └── bge-reranker-v2-m3/
├── HLM_chroma_db/                  ← 自动生成
├── Create_db.py                   ← 生成文本向量数据库脚本，先运行
├── HLM_RAG_tt.py                   ← 主问答程序
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

创建虚拟环境后安装依赖

```python
pip install -r requirements.txt
```

2.获取qwenAPI

- [阿里云百炼平台](https://bailian.console.aliyun.com/cn-beijing/)

依次点击**模型服务-密钥管理**，获取API后在.env文件中赋值给**Qwen_API_KEY**。

3.下载模型

将以下模型下载至 `models/` 目录：

- [bge-large-zh-v1.5](https://huggingface.co/BAAI/bge-large-zh-v1.5)
- [bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3)

> 💡 点击链接后进入Files and versions即可找到模型文件，推荐使用 `huggingface-cli` 或浏览器下载并解压到对应目录。

4.创建向量数据库

 运行**Create_db.py**文件即可创建向量数据库，每次运行大概**三十分钟**。

5.启动问答系统

运行**HLM_RAG_main.py**文件稍等片刻即可启动系统，在运行框输入问题，1到5秒内系统会给出结果。

6.系统演示

![image-20260203141334972](C:\Users\48444\AppData\Roaming\Typora\typora-user-images\image-20260203141334972.png)

7.**🤝 贡献与反馈**

欢迎提交以下内容：

- 支持其他古典小说（如《三国演义》）
- 添加 Web UI（Gradio / Streamlit）
- 优化分块策略
- 改进 prompt 设计