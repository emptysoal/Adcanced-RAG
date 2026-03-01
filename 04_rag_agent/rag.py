"""
    RAG Agent
RAG 应用的一种形式是将其视为一个带有信息检索工具的简单 Agent。
通过实现一个封装了向量存储的工具来构建一个最小的 RAG Agent
"""

from langchain.tools import tool
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import after_model
from langchain.messages import SystemMessage, HumanMessage, ToolMessage, RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.runtime import Runtime
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI

from retriever import MilvusRetriever
from config import *

milvus_retriever = MilvusRetriever()

llm = ChatOpenAI(
    model_name=LLM_MODEL_NAME,
    openai_api_base=LLM_ENDPOINT,
    openai_api_key=LLM_API_KEY,
    temperature=LLM_TEMPERATURE
)


@tool
def retrieve_docs(query: str):
    """接收用户关于大模型或卷积神经网络相关的深度学习方面的提问，检索相关文档，返回和用户提问相关的文档信息"""

    doc_chunks = milvus_retriever.retrieve(query)
    files = [doc_chunk_dict["file_name"] for doc_chunk_dict in doc_chunks]
    docs = [doc_chunk_dict["text"] for doc_chunk_dict in doc_chunks]

    context_list = []
    for file_name, doc_content in zip(files, docs):
        one_context = f"文档名：\n{file_name}\n文档内容：\n{doc_content}"
        context_list.append(one_context)
    context = "参考以下内容进行回答：\n\n" + "\n\n".join(context_list)

    return context


@after_model
def trim_messages(state: AgentState, runtime: Runtime) -> dict | None:
    """Remove old messages to keep conversation manageable."""
    messages = state["messages"]

    if len(messages) <= MESSAGES_KEEP:
        return None

    # remove the earliest messages
    recent_messages = messages[-MESSAGES_KEEP:]

    if not isinstance(recent_messages[0], HumanMessage):  # 裁剪后的消息第一条不是用户消息
        start_idx = 0
        for i in range(len(messages) - MESSAGES_KEEP - 1, -1, -1):  # 从截断处向前遍历，找到最近的用户消息
            if isinstance(messages[i], HumanMessage):
                start_idx = i
                break
        recent_messages = messages[start_idx:]

    return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *recent_messages]}


rag_agent = create_agent(
    model=llm,
    tools=[retrieve_docs],
    system_prompt=SYSTEM_PROMPT,
    checkpointer=InMemorySaver(),
    middleware=[trim_messages]
)

# Show the workflow
# graph = rag_agent.get_graph(xray=True)
# mermaid_code = graph.draw_mermaid()
# print(mermaid_code)

if __name__ == '__main__':
    config: RunnableConfig = {"configurable": {"thread_id": "1"}}

    input_message = {"role": "user", "content": "卷积神经网络的应用场景有哪些？"}
    for stream_mode, data in rag_agent.stream(
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
            if msg.content and metadata["langgraph_node"] == "model":
                print(msg.content, end="", flush=True)

    input_message = {"role": "user", "content": "大模型呢？"}
    for stream_mode, data in rag_agent.stream(
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
            if msg.content and metadata["langgraph_node"] == "model":
                print(msg.content, end="", flush=True)
