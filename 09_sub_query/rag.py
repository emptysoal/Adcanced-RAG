"""

"""

import operator
from typing import Annotated, TypedDict
from pydantic import BaseModel, Field

from langchain.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from langchain_openai import ChatOpenAI

from retriever import MilvusRetriever
from config import *

milvus_retriever = MilvusRetriever()

# 这里的分解子查询和回复的大模型使用的是同一个，也可以使用不同的
sub_query_llm = ChatOpenAI(
    model_name=LLM_MODEL_NAME,
    openai_api_base=LLM_ENDPOINT,
    openai_api_key=LLM_API_KEY,
    temperature=LLM_TEMPERATURE
)

llm = ChatOpenAI(
    model_name=LLM_MODEL_NAME,
    openai_api_base=LLM_ENDPOINT,
    openai_api_key=LLM_API_KEY,
    temperature=LLM_TEMPERATURE
)


# 定义状态
class AgentInput(TypedDict):
    """每个子查询的简单输入状态。"""
    query: str


class RouterState(TypedDict):
    """路由节点的状态"""
    query: str
    sub_queries: list[str]
    results: Annotated[list[str], operator.add]  # Reducer collects parallel results
    final_answer: str


# 构建路由器工作流程
# 为分类器定义结构化的输出模式
class SubQueryResult(BaseModel):
    sub_queries: list[str] = Field(description="存放查询分解后得到的子查询。")


def decompose_query(state: RouterState) -> dict:
    """对查询进行分解，得到子查询"""
    structured_llm = sub_query_llm.with_structured_output(SubQueryResult)

    sub_query_result = structured_llm.invoke(
        [
            SystemMessage(content=SUB_QUERY_SYSTEM_PROMPT),
            HumanMessage(content=state["query"])
        ]
    )

    return {"sub_queries": sub_query_result.sub_queries}


# 测试查询分解
# test_query = {"query": "卷积神经网络的原理是什么？有哪些应用场景？"}
# ret = decompose_query(test_query)
# print(ret)


def route_to_retrievers(state: RouterState) -> list[Send]:
    """把各子查询进行分发。"""
    return [
        Send("retrieve_docs", {"query": sub_query}) for sub_query in state['sub_queries']
    ]


def retrieve_docs(state: AgentInput) -> dict:
    """根据用户的查询，从向量数据库检索相关的文档"""

    # 检索
    doc_chunks = milvus_retriever.retrieve(state["query"])
    files = [doc_chunk_dict["file_name"] for doc_chunk_dict in doc_chunks]
    docs = [doc_chunk_dict["text"] for doc_chunk_dict in doc_chunks]

    context_list = []
    for file_name, doc_content in zip(files, docs):
        one_context = f"文档名：\n{file_name}\n文档内容：\n{doc_content}"
        context_list.append(one_context)
    context = "\n\n".join(context_list)

    return {"results": [context]}


def synthesize_results(state: RouterState) -> dict:
    """将所有子查询节点的结果组合成一个连贯的答案。"""
    if not state["results"]:
        return {"final_answer": "没有从任何知识来源中找到结果。"}

    # Format results for synthesis
    context = "\n\n".join(state["results"])

    system_message = SYSTEM_PROMPT.format(context_info=context)

    synthesis_response = llm.invoke(
        [
            SystemMessage(content=system_message),
            HumanMessage(content=state["query"])
        ]
    )

    return {"final_answer": synthesis_response.content}


def build_workflow():
    # Build workflow
    workflow = StateGraph(RouterState)

    # Add nodes
    workflow.add_node("decompose_query", decompose_query)
    workflow.add_node("retrieve_docs", retrieve_docs)
    workflow.add_node("synthesize", synthesize_results)

    # Add edges to connect nodes
    workflow.add_edge(START, "decompose_query")
    workflow.add_conditional_edges(
        "decompose_query",
        route_to_retrievers,
        ["retrieve_docs"]
    )
    workflow.add_edge("retrieve_docs", "synthesize")
    workflow.add_edge("synthesize", END)

    # Compile the agent
    rag = workflow.compile()

    # Show the workflow
    # graph = rag.get_graph(xray=True)
    # mermaid_code = graph.draw_mermaid()
    # print(mermaid_code)

    return rag


if __name__ == '__main__':
    rag_workflow = build_workflow()

    # input_ = {"query": "卷积神经网络的应用场景有哪些？"}
    # input_ = {"query": "大模型的应用场景有哪些？"}
    input_ = {"query": "卷积神经网络和大模型的应用场景分别有哪些？"}
    # result = rag_workflow.invoke(input_)
    # print(result)

    for stream_mode, data in rag_workflow.stream(
            input_,
            stream_mode=["updates", "messages"]
    ):
        if stream_mode == "updates":
            for source, update in data.items():
                print("=== Node Name:", source)
                print("=== Updated State:", update)
        if stream_mode == "messages":
            msg, metadata = data
            if msg.content and metadata["langgraph_node"] == "synthesize":
                print(msg.content, end="", flush=True)
