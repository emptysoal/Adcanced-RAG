"""

"""

from langgraph.graph import MessagesState, START, END, StateGraph
from langchain.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage, RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.checkpoint.memory import InMemorySaver
from langchain_openai import ChatOpenAI

from retriever import MilvusRetriever
from config import *

milvus_retriever = MilvusRetriever()

hyde_llm = ChatOpenAI(
    model=HYD_LLM_MODEL_NAME,
    base_url=HYD_LLM_ENDPOINT,
    api_key=HYD_LLM_API_KEY,
    temperature=HYD_LLM_TEMPERATURE
)

llm = ChatOpenAI(
    model_name=LLM_MODEL_NAME,
    openai_api_base=LLM_ENDPOINT,
    openai_api_key=LLM_API_KEY,
    temperature=LLM_TEMPERATURE
)


class State(MessagesState):
    files: list[str]
    documents: list[str]
    fake_answer: str


# Nodes
def generate_fake_answer(state: State):
    """假设文档生成"""

    response = hyde_llm.invoke(
        [SystemMessage(content=HYD_SYSTEM_PROMPT)] + state["messages"]
    )

    return {"fake_answer": response.content}


def retrieve_docs(state: State):
    """根据用户的查询，从向量数据库i检索相关的文档"""

    # 获取最近的用户的消息
    # last_message = state["messages"][-1]
    # assert isinstance(last_message, HumanMessage), "In retriever node, last message is not HumanMessage."
    # user_query = last_message.content

    # 检索
    doc_chunks = milvus_retriever.retrieve(state["fake_answer"])
    files = [doc_chunk_dict["file_name"] for doc_chunk_dict in doc_chunks]
    docs = [doc_chunk_dict["text"] for doc_chunk_dict in doc_chunks]

    return {"files": files, "documents": docs}


def llm_call(state: State):
    # context = "\n\n".join(state["documents"])

    context_list = []
    for file_name, doc_content in zip(state["files"], state["documents"]):
        one_context = f"文档名：\n{file_name}\n文档内容：\n{doc_content}"
        context_list.append(one_context)
    context = "\n\n".join(context_list)

    system_message = SYSTEM_PROMPT.format(context_info=context)

    response = llm.invoke(
        [SystemMessage(content=system_message)] + state["messages"]
    )

    return {"messages": [response]}


def delete_messages(state: State):
    messages = state['messages']
    if len(messages) > 2:
        # remove the earliest messages
        recent_messages = messages[-2:]
        return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *recent_messages]}
    return {"messages": []}


def build_workflow():
    # Build workflow
    workflow = StateGraph(State)

    # Add nodes
    workflow.add_node("generate_fake_answer", generate_fake_answer)
    workflow.add_node("retrieve_docs", retrieve_docs)
    workflow.add_node("llm_call", llm_call)
    workflow.add_node("trim_messages", delete_messages)

    # Add edges to connect nodes
    workflow.add_edge(START, "generate_fake_answer")
    workflow.add_edge("generate_fake_answer", "retrieve_docs")
    workflow.add_edge("retrieve_docs", "llm_call")
    workflow.add_edge("llm_call", "trim_messages")
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

    input_message = {"role": "user", "content": "卷积神经网络的应用场景有哪些？"}
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
            if msg.content and metadata["langgraph_node"] == "llm_call":
                print(msg.content, end="", flush=True)

    # 测试查询重写能力
    input_message_2 = {"role": "user", "content": "大模型呢？"}
    for stream_mode, data in rag_workflow.stream(
            {"messages": [input_message_2]},
            config,
            stream_mode=["updates", "messages"]
    ):
        if stream_mode == "updates":
            for source, update in data.items():
                print("=== Node Name:", source)
                print("=== Updated State:", update)
        if stream_mode == "messages":
            msg, metadata = data
            if msg.content and metadata["langgraph_node"] == "llm_call":
                print(msg.content, end="", flush=True)
