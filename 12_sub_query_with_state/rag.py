"""

"""

import operator
from typing import Annotated, Literal, TypedDict
from pydantic import BaseModel, Field

from langchain.messages import SystemMessage, RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.types import Send, Overwrite
from langchain_openai import ChatOpenAI

from retriever import MilvusRetriever
from config import *

cnn_milvus_retriever = MilvusRetriever(milvus_collection=CNN_COLLECTION_NAME)  # 检索 CNN 相关的文档
llm_milvus_retriever = MilvusRetriever(milvus_collection=LM_COLLECTION_NAME)  # 检索 LLM 相关的文档

sub_query_llm = ChatOpenAI(
    model_name=SUB_QUERY_LLM_MODEL_NAME,
    openai_api_base=SUB_QUERY_LLM_ENDPOINT,
    openai_api_key=SUB_QUERY_LLM_API_KEY,
    temperature=SUB_QUERY_LLM_TEMPERATURE
)

llm = ChatOpenAI(
    model_name=LLM_MODEL_NAME,
    openai_api_base=LLM_ENDPOINT,
    openai_api_key=LLM_API_KEY,
    temperature=LLM_TEMPERATURE
)
# llm = ChatOpenAI(
#     model_name=SUB_QUERY_LLM_MODEL_NAME,
#     openai_api_base=SUB_QUERY_LLM_ENDPOINT,
#     openai_api_key=SUB_QUERY_LLM_API_KEY,
#     temperature=SUB_QUERY_LLM_TEMPERATURE
# )


# 定义状态
class AgentInput(TypedDict):
    """每个子查询的简单输入状态。"""
    query: str


class AgentOutput(TypedDict):
    """单个子查询的输出"""
    category: str
    result: str


class Classification(TypedDict):
    """分解后的某个子查询，和它对应的类别（属于哪个垂直领域）"""
    # category: Literal["CNN(Convolutional Neural Network)", "LLM(Large Language Model)"]
    category: Literal["CNN", "LLM"]
    query: str


class RouterState(MessagesState):
    """路由节点的状态"""
    classifications: list[Classification]
    results: Annotated[list[AgentOutput], operator.add]  # Reducer collects parallel results


# 构建路由器工作流程
# 为分类器定义结构化的输出模式
class ClassificationResult(BaseModel):
    classifications: list[Classification] = Field(description="存放查询分解后得到的子查询的信息。")


def classify_query(state: RouterState) -> dict:
    """对查询进行分解，得到子查询以及对应的类别"""
    structured_llm = sub_query_llm.with_structured_output(ClassificationResult)

    classification_result = structured_llm.invoke(
        [SystemMessage(content=SUB_QUERY_SYSTEM_PROMPT)] + state["messages"]
    )

    return {"classifications": classification_result.classifications}


# 测试查询分解
# test_message = {"role": "user", "content": "卷积神经网络的原理是什么？有哪些应用场景？"}
# test_message = {"role": "user", "content": "卷积神经网络和大模型分别有哪些应用场景？"}
# test_message = [
#     {"role": "user", "content": "你好，我的名字是李明，很高兴认识你。"},
#     {"role": "assistant", "content": "你好，我是一个AI助手，今天有什么能帮助你吗？"},
#     {"role": "user", "content": "卷积神经网络和大模型分别有哪些应用场景？"}
# ]
# ret = classify_query({"messages": test_message})
# print(ret)


def route_to_retrievers(state: RouterState) -> list[Send]:
    """根据子查询及分类向各节点分发。"""
    return [
        Send(c["category"], {"query": c["query"]}) for c in state['classifications']
    ]


def retrieve_cnn_docs(state: AgentInput) -> dict:
    """根据用户的查询，从 CNN 向量数据库检索相关的文档"""

    # 检索
    doc_chunks = cnn_milvus_retriever.retrieve(state["query"])
    files = [doc_chunk_dict["file_name"] for doc_chunk_dict in doc_chunks]
    docs = [doc_chunk_dict["text"] for doc_chunk_dict in doc_chunks]

    context_list = []
    for file_name, doc_content in zip(files, docs):
        one_context = f"文档名：\n{file_name}\n文档内容：\n{doc_content}"
        context_list.append(one_context)
    context = "\n\n".join(context_list)

    return {"results": [{"category": "CNN", "result": context}]}


def retrieve_llm_docs(state: AgentInput) -> dict:
    """根据用户的查询，从 LLM 向量数据库检索相关的文档"""

    # 检索
    doc_chunks = llm_milvus_retriever.retrieve(state["query"])
    files = [doc_chunk_dict["file_name"] for doc_chunk_dict in doc_chunks]
    docs = [doc_chunk_dict["text"] for doc_chunk_dict in doc_chunks]

    context_list = []
    for file_name, doc_content in zip(files, docs):
        one_context = f"文档名：\n{file_name}\n文档内容：\n{doc_content}"
        context_list.append(one_context)
    context = "\n\n".join(context_list)

    return {"results": [{"category": "LLM", "result": context}]}


def synthesize_results(state: RouterState) -> dict:
    """将所有子查询节点的结果组合成一个连贯的答案。"""
    if not state["results"]:
        return {"final_answer": "没有从任何知识来源中找到结果。"}

    # Format results for synthesis
    formatted = [
        f"{r['category']} 相关的检索结果为:\n{r['result']}" for r in state["results"]
    ]
    context = "\n\n".join(formatted)

    system_message = SYSTEM_PROMPT.format(context_info=context)

    synthesis_response = llm.invoke(
        [SystemMessage(content=system_message)] + state["messages"]
    )

    return {
        "results": Overwrite([]),  # 清空本次检索结果
        "messages": [synthesis_response]
    }


def delete_messages(state: RouterState):
    messages = state['messages']
    if len(messages) > MESSAGES_KEEP:
        # remove the earliest messages
        recent_messages = messages[-MESSAGES_KEEP:]
        return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *recent_messages]}
    return {"messages": []}


def build_workflow():
    # Build workflow
    workflow = StateGraph(RouterState)

    # Add nodes
    workflow.add_node("classify", classify_query)
    workflow.add_node("CNN", retrieve_cnn_docs)
    workflow.add_node("LLM", retrieve_llm_docs)
    workflow.add_node("synthesize", synthesize_results)
    workflow.add_node("trim_messages", delete_messages)

    # Add edges to connect nodes
    workflow.add_edge(START, "classify")
    workflow.add_conditional_edges(
        "classify",
        route_to_retrievers,
        ["CNN", "LLM"]
    )
    workflow.add_edge("CNN", "synthesize")
    workflow.add_edge("LLM", "synthesize")
    workflow.add_edge("synthesize", "trim_messages")
    workflow.add_edge("trim_messages", END)

    checkpointer = InMemorySaver()
    # Compile the agent
    rag = workflow.compile(checkpointer=checkpointer)

    # Show the workflow
    # graph = rag.get_graph(xray=True)
    # mermaid_code = graph.draw_mermaid()
    # print(mermaid_code)

    return rag


if __name__ == '__main__':
    rag_workflow = build_workflow()

    config = {"configurable": {"thread_id": "1"}}

    # input_message = {"role": "user", "content": "卷积神经网络的应用场景有哪些？"}
    # input_message = {"role": "user", "content": "大模型的应用场景有哪些？"}
    input_message = {"role": "user", "content": "卷积神经网络和大模型的应用场景分别有哪些？"}
    # result = rag_workflow.invoke({"messages": [input_message]})
    # print(result)

    for stream_mode, data in rag_workflow.stream(
            {"messages": [input_message]},
            config,
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

    input_message = {"role": "user", "content": "它们的原理分别是什么？"}
    for stream_mode, data in rag_workflow.stream(
            {"messages": [input_message]},
            config,
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
