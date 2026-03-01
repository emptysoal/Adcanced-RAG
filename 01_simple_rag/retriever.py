"""
    检索器
使用 llama-index 对 Milvus 的集成，加载已有的 vector store，并从中检索与查询相关的文档段
"""

from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.embeddings.openai_like import OpenAILikeEmbedding

from config import *


class MilvusRetriever:
    def __init__(
            self,
            embedding_model=EMBEDDING_MODEL_NAME,
            embedding_endpoint=EMBEDDING_ENDPOINT,
            embedding_api_key=EMBEDDING_API_KEY,
            milvus_uri=MILVUS_URI,
            milvus_collection=COLLECTION_NAME,
            similarity_top_k=DEFAULT_LIMIT
    ):
        embedding = OpenAILikeEmbedding(
            model_name=embedding_model,
            api_base=embedding_endpoint,
            api_key=embedding_api_key
        )

        # 加载 vector store，overwrite=False 这个是重点
        vector_store = MilvusVectorStore(
            uri=milvus_uri,
            collection_name=milvus_collection,
            overwrite=False,
            enable_dense=True,
            dim=1024,
            index_config={"index_type": "AUTOINDEX", "metric_type": "COSINE"}
        )

        self.index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            embed_model=embedding
        )

        self.similarity_top_k = similarity_top_k

    def retrieve(self, query: str) -> list[dict]:
        # 检索测试
        retriever = self.index.as_retriever(similarity_top_k=self.similarity_top_k)
        node_with_score_list = retriever.retrieve(query)
        # print(node_with_score_list)

        results = []
        for node_with_score in node_with_score_list:
            node = node_with_score.node
            score = node_with_score.score
            result = {
                "file_name": node.metadata["file_name"],
                "text": node.get_content(),
                "score": round(score, 4)
            }
            results.append(result)

        return results


if __name__ == '__main__':
    milvus_retriever = MilvusRetriever()

    # 检索测试
    test_query = "卷积神经网络的原理是什么？"
    ret = milvus_retriever.retrieve(test_query)
    print(ret)
