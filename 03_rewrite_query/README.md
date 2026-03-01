# RAG：查询重写

```mermaid
graph TD;
	__start__([<p>__start__</p>]):::first
	rewrite_query(rewrite_query)
	retrieve_docs(retrieve_docs)
	llm_call(llm_call)
	trim_messages(trim_messages)
	__end__([<p>__end__</p>]):::last
	__start__ --> rewrite_query;
	llm_call --> trim_messages;
	retrieve_docs --> llm_call;
	rewrite_query --> retrieve_docs;
	trim_messages --> __end__;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc
```