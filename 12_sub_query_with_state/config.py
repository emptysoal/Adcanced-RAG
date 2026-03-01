"""
    配置文件
"""

import os

# 模型服务
SUB_QUERY_LLM_ENDPOINT = "http://127.0.0.1:32804/v1"
SUB_QUERY_LLM_MODEL_NAME = "/models/Qwen2.5-7B-Instruct"
SUB_QUERY_LLM_API_KEY = os.getenv("LLM_API_KEY", "EMPTY")
SUB_QUERY_LLM_TEMPERATURE = 0.3
LLM_ENDPOINT = "http://127.0.0.1:32804/v1"
LLM_MODEL_NAME = "/models/Qwen2.5-7B-Instruct"
LLM_API_KEY = os.getenv("LLM_API_KEY", "EMPTY")
LLM_TEMPERATURE = 0.5

EMBEDDING_ENDPOINT = "http://127.0.0.1:32805/v1"
EMBEDDING_MODEL_NAME = "/models/BAAI/bge-m3"
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "EMPTY")
EMBEDDING_DIM = 1024

RERANKER_ENDPOINT = "http://127.0.0.1:32806"
RERANKER_MODEL_NAME = "/models/BAAI/bge-reranker-v2-m3"
RERANKER_API_KEY = os.getenv("RERANKER_API_KEY", "EMPTY")

# Milvus 配置
MILVUS_URI = "http://127.0.0.1:19530"
CNN_COLLECTION_NAME = "PDF_CNN_Vector"
LM_COLLECTION_NAME = "PDF_LM_Vector"

# 搜索参数
DEFAULT_TOP_K = 3
DEFAULT_THRESHOLD = 0.6

# 消息修剪
MESSAGES_KEEP = 8

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

# 分解子查询提示词参数
SUB_QUERY_SYSTEM_PROMPT = """
你是一个专业的查询分析专家。你的任务是将用户的复杂问题分解为多个简单的子查询，以便进行并行信息检索。
如果用户的提问已经无法再分解，则只进行补全或重写，避免模糊指代，明确实体名称和时间范围；如果需分解，则按照以下说明分解子查询。

## 分解原则
1. **独立性**：每个子查询应能独立检索，不依赖其他子查询的结果
2. **完整性**：所有子查询的并集必须能完整回答原问题
3. **原子性**：子查询不可再分，聚焦单一事实或概念
4. **具体性**：避免模糊指代，明确实体名称和时间范围

## 示例 1
用户查询为：中国和英国的首都分别是哪里？
你的输出应当包含 2 个子查询，类似于：
第一个子查询：中国的首都是哪里？
第二个子查询：英国的首都是哪里？
就是一定要分解为完全独立的，且无法再细分的子问题，且所有子问题能覆盖原问题
## 示例 2
用户查询为：黄河是哪个国家的？全长是多少？
你的输出应当包含 2 个子查询，类似于：
第一个子查询：黄河是哪个国家的？
第二个子查询：黄河全长是多少米？
这个示例除了示例1中提到的，还展示了一定要消除模糊指代，考虑上下文在‘全长是多少？’前加上了黄河
"""
