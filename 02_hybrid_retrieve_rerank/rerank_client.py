"""
    requests 客户端，访问使用 vLLM 部署的 Reranker 模型
    部署命令：
        vllm serve /models/BAAI/bge-reranker-v2-m3 --dtype=float16 --trust-remote-code
"""

import requests
from config import RERANKER_ENDPOINT, RERANKER_MODEL_NAME, RERANKER_API_KEY

score_url = RERANKER_ENDPOINT + "/score"
rerank_url = RERANKER_ENDPOINT + "/rerank"
headers = {
    "accept": "application/json",
    "Content-Type": "application/json"
}


def score_request(text_1, text_2):
    data = {
        "model": RERANKER_MODEL_NAME,
        "encoding_format": "float",
        "text_1": text_1,
        "text_2": text_2
    }
    response = requests.post(score_url, headers=headers, json=data)

    response_code = response.status_code
    response_data = response.json()
    score = response_data["data"]

    return {"code": response_code, "score": score}


def rerank_request(query, documents):
    data = {
        "model": RERANKER_MODEL_NAME,
        "query": query,
        "documents": documents
    }
    response = requests.post(rerank_url, headers=headers, json=data)

    response_code = response.status_code
    response_data = response.json()
    rerank_result = response_data["results"]

    return {"code": response_code, "results": rerank_result}


if __name__ == '__main__':
    test_text_1 = "What is the capital of France?"
    test_text_2 = [
        "The capital of Brazil is Brasilia.",
        "The capital of France is Paris."
    ]
    score_ret = score_request(test_text_1, test_text_2)
    print(score_ret)

    test_query = "What is the capital of France?"
    test_documents = [
        "The capital of Brazil is Brasilia.",
        "The capital of France is Paris.",
        "Horses and cows are both animals"
    ]
    rerank_ret = rerank_request(test_query, test_documents)
    print(rerank_ret)
