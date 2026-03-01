"""
    配置文件
"""

import os

# 模型服务
LLM_ENDPOINT = "http://127.0.0.1:32804/v1"
LLM_MODEL_NAME = "/models/Qwen2.5-7B-Instruct"
LLM_API_KEY = os.getenv("LLM_API_KEY", "EMPTY")
LLM_TEMPERATURE = 0.3
REWRITE_LLM_ENDPOINT = "http://127.0.0.1:32804/v1"
REWRITE_LLM_MODEL_NAME = "/models/Qwen2.5-7B-Instruct"
REWRITE_LLM_API_KEY = os.getenv("LLM_API_KEY", "EMPTY")
REWRITE_LLM_TEMPERATURE = 0.3

EMBEDDING_ENDPOINT = "http://127.0.0.1:32805/v1"
EMBEDDING_MODEL_NAME = "/models/BAAI/bge-m3"
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "EMPTY")
EMBEDDING_DIM = 1024

RERANKER_ENDPOINT = "http://127.0.0.1:32806"
RERANKER_MODEL_NAME = "/models/BAAI/bge-reranker-v2-m3"
RERANKER_API_KEY = os.getenv("RERANKER_API_KEY", "EMPTY")

# Milvus 配置
MILVUS_URI = "http://127.0.0.1:19530"
COLLECTION_NAME = "PDF_Parent_Child_Vector_hybrid"

# 搜索参数
DEFAULT_TOP_K = 3
DEFAULT_THRESHOLD = 0.6

# 大模型对话提示词参数
SYSTEM_PROMPT = """
# 角色：
你是一个专业的问题解答助手，可以参考下方的上下文信息，根据用户最近的提问给出回答：

上下文信息为：
{context_info}

## 限制：
1. 如果用户的提问或查询与'上下文信息'相关，则结合'上下文信息'，根据用户输入给出回答；
2. 如果用户的提问或查询与'上下文信息'不相关，则坚决不使用'上下文信息'，仅根据用户输入给出回答；
"""

# 用户提问补全重写提示词参数
REWRITE_SYSTEM_PROMPT = """你是一个非常强大的助手，能够根据历史对话对用户当前的提问进行补充或重写，使当前提问在不结合上下文语境的情况下也能完整地表达语义，用于知识库检索，
你可以参考这个例子：
历史对话为：[{"role": "user", "content": "今天北京的天气怎么样？"},
{"role": "assistant", "content": "今天北京是晴天，气温20摄氏度。"}]
用户当前提问为："上海呢？"
你的输出应当类似于："今天上海的天气怎么样？"
也就是说你能灵活地判断用户当前的提问是否完整或者包含指代词‘这个’或‘那个’等，如果不完整或包含指代词，则根据历史对话把提问补充或重写成一个独立完整的问题。

## 限制
1. 如果当前提问需要补全或重写，只输出补全或重写后的提问，不要输出任何其他内容；
2. 如果当前问题是独立的，则原样输出。
"""
