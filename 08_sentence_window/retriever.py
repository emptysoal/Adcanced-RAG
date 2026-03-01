"""
    检索器
使用 llama-index 对 Milvus 的集成，加载已有的 vector store，并从中检索与查询相关的文档段
"""

from llama_index.core.schema import NodeWithScore
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.vector_stores.milvus.utils import BM25BuiltInFunction
from llama_index.embeddings.openai_like import OpenAILikeEmbedding

from rerank_client import rerank_request
from config import *


class MilvusRetriever:
    def __init__(
            self,
            embedding_model=EMBEDDING_MODEL_NAME,
            embedding_endpoint=EMBEDDING_ENDPOINT,
            embedding_api_key=EMBEDDING_API_KEY,
            embedding_dim=EMBEDDING_DIM,
            milvus_uri=MILVUS_URI,
            milvus_collection=COLLECTION_NAME,
            similarity_top_k=DEFAULT_TOP_K,
            similarity_threshold=DEFAULT_THRESHOLD
    ):
        embedding = OpenAILikeEmbedding(
            model_name=embedding_model,
            api_base=embedding_endpoint,
            api_key=embedding_api_key
        )

        # 加载 vector store，overwrite=False 这个是重点
        analyzer_params = {
            "type": "chinese",
            "stop_words": ["的", "了"]
        }
        bm25_function = BM25BuiltInFunction(analyzer_params=analyzer_params, enable_match=True)
        vector_store = MilvusVectorStore(
            uri=milvus_uri,
            collection_name=milvus_collection,
            overwrite=False,
            enable_dense=True,
            dim=embedding_dim,
            index_config={"index_type": "AUTOINDEX", "metric_type": "COSINE"},
            enable_sparse=True,  # enable sparse to implement full text search
            sparse_embedding_function=bm25_function
        )

        self.index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            embed_model=embedding
        )

        self.similarity_top_k = similarity_top_k
        self.similarity_threshold = similarity_threshold

    @staticmethod
    def _merge_results(
            semantic_results: list[NodeWithScore],
            full_text_results: list[NodeWithScore]
    ) -> list[NodeWithScore]:
        """合并并去重结果"""
        seen_ids = set()
        merged = []

        total_results = semantic_results + full_text_results
        for result in total_results:
            chunk_id = result.node.id_
            if chunk_id not in seen_ids:
                seen_ids.add(chunk_id)
                merged.append(result)

        return merged

    def _get_prev_next_nodes(self, node_with_score_list: list[NodeWithScore]) -> list[NodeWithScore]:
        node_id_set = set()  # 收集传入的全部 Node id ，避免重复
        for node_with_score in node_with_score_list:
            node_id_set.add(node_with_score.node.id_)

        new_node_with_score_list = []
        for node_with_score in node_with_score_list:
            node = node_with_score.node

            # 获取它的前一个节点
            prev_node_info = node.prev_node
            if prev_node_info is not None:  # 如果存在前一个节点
                prev_node_id = prev_node_info.node_id
                prev_nodes = self.index.vector_store.get_nodes([prev_node_id])
                prev_node = prev_nodes[0]
                if prev_node_id not in node_id_set:
                    node_id_set.add(prev_node_id)
                    prev_node_with_score = NodeWithScore(node=prev_node, score=node_with_score.score * 0.9)
                    new_node_with_score_list.append(prev_node_with_score)

            new_node_with_score_list.append(node_with_score)

            # 获取它的后一个节点
            next_node_info = node.next_node
            if next_node_info is not None:
                next_node_id = next_node_info.node_id
                next_nodes = self.index.vector_store.get_nodes([next_node_id])
                next_node = next_nodes[0]
                if next_node_id not in node_id_set:
                    node_id_set.add(next_node_id)
                    next_node_with_score = NodeWithScore(node=next_node, score=node_with_score.score * 0.9)
                    new_node_with_score_list.append(next_node_with_score)

        return new_node_with_score_list

    def retrieve(self, query: str) -> list[dict]:
        # 语义检索
        semantic_retriever = self.index.as_retriever(similarity_top_k=int(self.similarity_top_k * 1.5))
        semantic_node_with_score_list = semantic_retriever.retrieve(query)

        # 全文检索
        full_text_retriever = self.index.as_retriever(vector_store_query_mode="sparse",
                                                      similarity_top_k=int(self.similarity_top_k * 1.5))
        full_text_node_with_score_list = full_text_retriever.retrieve(query)

        # 合并去重
        merged_results = self._merge_results(semantic_node_with_score_list, full_text_node_with_score_list)
        if not merged_results:
            return []

        # 提取文档内容
        documents = [result.node.get_content() for result in merged_results]

        # rerank
        rerank_ret = rerank_request(query, documents)
        rerank_result = rerank_ret["results"] if rerank_ret["code"] == 200 else []
        # [{"index": int, "document": {"text": ""}, "relevance_score": float}, ...]
        if not rerank_result:
            return []

        # 过滤结果
        results = rerank_result[:self.similarity_top_k]  # Top K
        results = [result for result in results if result["relevance_score"] > self.similarity_threshold]
        # print(results)
        # [{'index': 1, 'document': {'text': '123'}, 'relevance_score': 0.9970703125},
        #  {'index': 4, 'document': {'text': '456'}, 'relevance_score': 0.978515625}]

        # 构建 rerank 后的 NodeWithScore
        ranked_node_with_score_list = []
        for result in results:
            origin_res_idx = result["index"]
            ranked_node_with_score = NodeWithScore(
                node=merged_results[origin_res_idx].node,
                score=result["relevance_score"]
            )
            ranked_node_with_score_list.append(ranked_node_with_score)

        # 句子窗口检索(PrevNextNodePostprocessor)
        post_node_with_score_list = self._get_prev_next_nodes(ranked_node_with_score_list)

        # 格式化输出
        final_results = []
        for post_node_with_score in post_node_with_score_list:
            final_results.append(
                {
                    "id": post_node_with_score.node.id_,
                    "file_name": post_node_with_score.node.metadata["file_name"],
                    "text": post_node_with_score.node.get_content(),
                    "score": post_node_with_score.score
                }
            )

        return final_results


if __name__ == '__main__':
    milvus_retriever = MilvusRetriever()

    # 检索测试
    test_query = "卷积神经网络的原理是什么？"
    ret = milvus_retriever.retrieve(test_query)
    print(ret)
