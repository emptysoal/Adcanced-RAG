"""
    构建知识库：递归检索器 + 节点引用
使用递归检索来遍历节点关系并根据“引用”获取节点。
节点引用是一个强大的概念。首次执行检索时，您可能希望检索的是引用而不是原始文本。多个引用可以指向同一个节点。
"""

import json
from pathlib import Path

from llama_index.readers.file import PDFReader
from llama_index.core.retrievers import RecursiveRetriever
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import IndexNode
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.embeddings.openai_like import OpenAILikeEmbedding

from config import *

embedding = OpenAILikeEmbedding(
    model_name=EMBEDDING_MODEL_NAME,
    api_base=EMBEDDING_ENDPOINT,
    api_key=EMBEDDING_API_KEY
)

vector_store = MilvusVectorStore(
    uri=MILVUS_URI,
    collection_name=COLLECTION_NAME,
    overwrite=True,
    enable_dense=True,  # enable_dense defaults to True
    dim=EMBEDDING_DIM,
    index_config={"index_type": "AUTOINDEX", "metric_type": "COSINE"}
)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# 1. 加载文档
doc_dir = "../documents/doc_pdf"
loader = PDFReader(return_full_document=True)
docs = []
for doc_name in os.listdir(doc_dir):
    doc_path = os.path.join(doc_dir, doc_name)
    doc = loader.load_data(file=Path(doc_path))
    docs.append(doc[0])

# 2. 分割文档
# 自定义文本分割
node_parser = SentenceSplitter(chunk_size=1024)
base_nodes = node_parser.get_nodes_from_documents(docs)

"""上面都是之前写过多次的常规的文档加载、分割、索引、检索代码
下面开始做递归分割和检索"""

# 代码块引用：较小的子代码块引用较大的父代码块
# sub_chunk_sizes = [128, 256, 512]
sub_chunk_sizes = [256]
sub_node_parsers = [
    SentenceSplitter(chunk_size=c, chunk_overlap=20) for c in sub_chunk_sizes
]

# 把原来的大块文本段，拆分为更小的文本段，且 Node 类型从 TextNode 转为 IndexNode
all_nodes = []
for base_node in base_nodes:
    for n in sub_node_parsers:
        sub_nodes = n.get_nodes_from_documents([base_node])
        sub_inodes = [
            IndexNode.from_text_node(sn, base_node.node_id) for sn in sub_nodes
        ]
        all_nodes.extend(sub_inodes)

    # also add original node to node
    original_node = IndexNode.from_text_node(base_node, base_node.node_id)
    all_nodes.append(original_node)
# print(all_nodes)

all_nodes_dict = {n.node_id: n for n in all_nodes}
# 保存节点信息，后续加载 vector store 做检索时需要
with open("all_nodes_dict.json", "w", encoding="utf-8") as f:
    json.dump({k: v.to_dict() for k, v in all_nodes_dict.items()}, f, ensure_ascii=False)

# 3. 文本嵌入、存储、索引
vector_index_chunk = VectorStoreIndex(
    all_nodes,
    embed_model=embedding,
    storage_context=storage_context
)

# 检索
query = "卷积神经网络有哪些应用场景？"
vector_retriever_chunk = vector_index_chunk.as_retriever(similarity_top_k=5)
retriever_chunk = RecursiveRetriever(
    "vector",
    retriever_dict={"vector": vector_retriever_chunk},
    node_dict=all_nodes_dict,
    verbose=True
)

nodes = retriever_chunk.retrieve(query)
print(nodes)
