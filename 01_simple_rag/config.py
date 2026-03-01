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

# Milvus 配置
MILVUS_URI = "http://127.0.0.1:19530"
COLLECTION_NAME = "PDF_Vector"

# 搜索参数
DEFAULT_LIMIT = 3

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
