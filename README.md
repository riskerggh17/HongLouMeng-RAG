



项目结构：

```python
HongLouMeng-RAG/
├── hongloumeng.txt                 ← 你的文本（可提供下载链接）
├── models/                         ← 手动下载的模型
│   ├── bge-large-zh-v1.5/
│   └── bge-reranker-v2-m3/
├── HLM_chroma_db/                  ← 自动生成
├── Create_d.py                   ← 生成文本向量数据库脚本，先运行
├── HLM_RAG_tt.py                   ← 主问答程序
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

