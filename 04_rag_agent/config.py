"""
    配置文件
"""

import os

# 模型服务
LLM_ENDPOINT = "http://127.0.0.1:32804/v1"
LLM_MODEL_NAME = "/models/Qwen2.5-7B-Instruct"
LLM_API_KEY = os.getenv("LLM_API_KEY", "EMPTY")
LLM_TEMPERATURE = 0.3

EMBEDDING_ENDPOINT = "http://127.0.0.1:32805/v1"
EMBEDDING_MODEL_NAME = "/models/BAAI/bge-m3"
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "EMPTY")
EMBEDDING_DIM = 1024

RERANKER_ENDPOINT = "http://127.0.0.1:32806"
RERANKER_MODEL_NAME = "/models/BAAI/bge-reranker-v2-m3"
RERANKER_API_KEY = os.getenv("RERANKER_API_KEY", "EMPTY")

# Milvus 配置
MILVUS_URI = "http://127.0.0.1:19530"
COLLECTION_NAME = "PDF_Hybrid_Vector"

# 搜索参数
DEFAULT_TOP_K = 3
DEFAULT_THRESHOLD = 0.6

# 消息修剪
MESSAGES_KEEP = 2

# 大模型对话提示词参数
SYSTEM_PROMPT = """
# 角色：
你是一个专业的问题解答助手

## 技能：
1. 当用户的提问或查询中，涉及大模型、卷积神经网络等深度学习和人工智能等相关词汇或信息时，你可以调用'检索文档工具'来搜索相关的知识内容；

## 限制
1. 如果需要调用'检索文档工具'，用户的提问有可能是包含了指代词或者信息不全的，这时你要自己根据上下文，把用户当前的提问进行补全或重写，能够独立完整地表达语义，再送入'检索文档工具'进行调用
2. 如果检索到的内容和用户提问相关，则结合检索到的上下文，根据用户输入给出回答；
3. 如果检索到的内容和用户提问不相干，则坚决不使用检索到的上下文，仅根据用户输入给出回答；
"""
