import os
import neo4j
from neo4j import GraphDatabase
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Tuple, List, Optional
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
import os
from langchain_community.graphs import Neo4jGraph
from langchain_community.document_loaders import WikipediaLoader
from langchain.text_splitter import TokenTextSplitter
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_experimental.graph_transformers import LLMGraphTransformer
from neo4j import GraphDatabase
# from yfiles_jupyter_graphs import GraphWidget
from langchain_community.vectorstores import Neo4jVector
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars
from langchain_core.runnables import ConfigurableField, RunnableParallel, RunnablePassthrough
from llmsherpa.readers import LayoutPDFReader
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import AzureOpenAIEmbeddings
import glob

# Neo4j env
os.environ["NEO4J_URI"] = "bolt://localhost:7687"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "yourpassword!"
os.environ["NEO4J_DB"] = "yourpassword"

graph = Neo4jGraph(database=os.getenv("NEO4J_DB"))

def query(user_question):

    llm=AzureChatOpenAI(temperature=0, api_version=os.getenv('OPENAI_API_VERSION'), azure_deployment=os.getenv('OPENAI_MODEL_NAME')) # gpt-4-0125-preview occasionally has issues

    retrieval_query = """
    WITH node, score
    ORDER BY score DESC LIMIT 10
    WHERE score > 0.8
    MATCH (node)<-[:HAS_EMBEDDING]-(c:Chunk)
    RETURN { id: id(node) } as metadata, c.sentences as text, score LIMIT 4
    """

    vector_index = Neo4jVector.from_existing_index(
        # OpenAIEmbeddings(),
        embedding = AzureOpenAIEmbeddings(),
        search_type="vector",
        # node_label="Chunks",
        index_name = "chunkVectorIndex",
        # text_node_properties=["text"],
        embedding_node_property="value",
        url = os.getenv("NEO4J_URI"),
        username = os.getenv("NEO4J_USERNAME"),
        password = os.getenv("NEO4J_PASSWORD"),
        database = os.getenv("NEO4J_DB"),
        retrieval_query = retrieval_query
    )

    def retriever(question: str):
        print(f"Search query: {question}")
        unstructured_data = [el.page_content for el in vector_index.similarity_search(question)]
        print("Unstructured Data: ", unstructured_data)
        final_data = f"""Unstructured data:
                        {"#Document ". join(unstructured_data)}
                    """
        return final_data


    # Condense a chat history and follow-up question into a standalone question
    _template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question,
    in its original language.
    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:"""  # noqa: E501
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

    def _format_chat_history(chat_history: List[Tuple[str, str]]) -> List:
        buffer = []
        for human, ai in chat_history:
            buffer.append(HumanMessage(content=human))
            buffer.append(AIMessage(content=ai))
        return buffer

    _search_query = RunnableBranch(
        # If input includes chat_history, we condense it with the follow-up question
        (
            RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(
                run_name="HasChatHistoryCheck"
            ),  # Condense follow-up question and chat into a standalone_question
            RunnablePassthrough.assign(
                chat_history=lambda x: _format_chat_history(x["chat_history"])
            )
            | CONDENSE_QUESTION_PROMPT
            | AzureChatOpenAI(temperature=0)
            | StrOutputParser(),
        ),
        # Else, we have no chat history, so just pass through the question
        RunnableLambda(lambda x : x["question"]),
    )

    template = """Answer the question based only on the following context:
    {context}
    Only use information from the context. You're an expert in the engineering/oil and gas domain. You're helping your clients find insights into malfunctions and perform root cause analysis.
    Question: {question}
    Use natural language to answer the question.
    Answer:"""
    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        RunnableParallel(
            {
                "context": _search_query | retriever,
                "question": RunnablePassthrough(),
            }
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    answer = chain.invoke({"question": user_question})

    return answer
