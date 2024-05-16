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
# from langchain.document_loaders import WikipediaLoader
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
import json
import glob

# Neo4j env
os.environ["NEO4J_URI"] = "bolt://localhost:7687"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "Password01!"
os.environ["NEO4J_DB"] = "etl"

graph = Neo4jGraph(database=os.getenv("NEO4J_DB"))


def query(user_question):

    # query = """
    #     WITH node AS answerEmb, score 
    #     ORDER BY score DESC LIMIT 10
    #     MATCH (answerEmb) <-[:HAS_EMBEDDING]- (answer) -[:HAS_PARENT*]-> (l:Lesson)
    #     WITH l, answer, score
    #     MATCH (t:Topic) <-[*]- (l) <-[:HAS_PARENT*]- (chunk:Chunk)
    #     WITH d, l, answer, chunk, score ORDER BY d.url_hash, s.title, chunk.idx ASC
    #     // 3 - prepare results
    #     WITH d, s, collect(answer) AS answers, collect(chunk) AS chunks, max(score) AS maxScore
    #     RETURN {source: d.url, page: chunks[0].page_idx+1, matched_chunk_id: id(answers[0])} AS metadata, 
    #         reduce(text = "", x IN chunks | text + x.sentences + '.') AS text, maxScore AS score LIMIT 5;
    # """
    # cypher query
    retrieval_query = """
        WITH node AS answerEmb, score 
        ORDER BY score DESC LIMIT 4
        MATCH (answerEmb) <-[:HAS_EMBEDDING]- (answer) -[:HAS_PARENT*]-> (l:Lesson)
        WITH l, answer, score
        OPTIONAL MATCH (l2:Lesson)-[ht2:HAS_TOPIC]->(t:Topic) <-[ht:HAS_TOPIC]- (l)
        WITH l, answer, score, COLLECT(abs(ht2.coef-ht.coef)) as diffscore
        WITH l, answer, score, (apoc.coll.sort(diffscore))[0] as minscore
        OPTIONAL MATCH (l)-[ht3:HAS_TOPIC]->(t)-[ht4:HAS_TOPIC]-(l3:Lesson)
        WHERE abs(ht3.coef-ht4.coef) = minscore
        WITH DISTINCT l, t, l3, score, answer
        RETURN 
        'abstract: ' + COALESCE(l.abstract, "") + ',\n' +
        '\ndrivingEvent: ' + COALESCE(l.drivingEvent, "") + ',\n' +
        '\nevidenceRecurrenceControlEffectiveness: ' + COALESCE(l.evidenceRecurrenceControlEffectiveness, "") + ',\n' +
        '\nrecommendations: ' + COALESCE(l.recommendations, "") + ',\n' +
        '\nlessonsLearned: ' + COALESCE(l.lessonsLearned, "") + ',\n' 
        AS text, score, {matched_chunk_id: id(l)} as metadata LIMIT 4
    """
    # retrieval_query = """  
    #     WITH node AS answerEmb, score 
    #     ORDER BY score DESC LIMIT 3
    #     MATCH (answerEmb) <-[:HAS_EMBEDDING]- (answer) -[:HAS_PARENT*]-> (s:Section)
    #     WITH s, answer, score
    #     MATCH (d:Document) <-[*]- (s) <-[:HAS_PARENT*]- (chunk:Chunk)-[:HAS_ENTITY]->(e:Entity)<-[:HAS_ENTITY]-(addChunk:Chunk)
    #     WHERE id(chunk) <> id(addChunk)
    #     WITH d, s, answer, chunk, addChunk, e.name as entityname ,score ORDER BY d.url_hash, s.title, chunk.block_idx ASC
    #     // 3 - prepare results
    #     WITH d, s, entityname, collect(answer) AS answers, collect(chunk) AS chunks, collect(addChunk) as additional_context, max(score) AS maxScore
    #     RETURN {source: d.url, page: chunks[0].page_idx+1, matched_chunk_id: id(answers[0])} AS metadata, 
    #         reduce(text = "", x IN chunks | text + x.sentences + '.') AS text, 
    #         { entity_name: entityname, additional_context: reduce(text = "", x IN additional_context | text + x.sentences + '.')} AS `additional_context`, maxScore AS score LIMIT 5;
    # """
    llm=AzureChatOpenAI(temperature=0, api_version=os.getenv('OPENAI_API_VERSION'), azure_deployment=os.getenv('OPENAI_MODEL_NAME')) # gpt-4-0125-preview occasionally has issues
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

    # Extract entities from text
    class Entities(BaseModel):
        """Identifying information about entities."""

        names: List[str] = Field(
            ...,
            description="All the person, organization, or business entities that "
            "appear in the text",
        )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are extracting organization and person entities from the text.",
            ),
            (
                "human",
                "Use the given format to extract information from the following "
                "input: {question}",
            ),
        ]
    )

    entity_chain = prompt | llm.with_structured_output(Entities)

    def generate_full_text_query(input: str) -> str:
        """
        Generate a full-text search query for a given input string.

        This function constructs a query string suitable for a full-text search.
        It processes the input string by splitting it into words and appending a
        similarity threshold (~2 changed characters) to each word, then combines
        them using the AND operator. Useful for mapping entities from user questions
        to database values, and allows for some misspelings.
        """
        full_text_query = ""
        words = [el for el in remove_lucene_chars(input).split() if el]
        for word in words[:-1]:
            full_text_query += f" {word}~2 AND"
        full_text_query += f" {words[-1]}~2"
        return full_text_query.strip()

    # Fulltext index query
    # def structured_retriever(question: str) -> str:
    #     """
    #     Collects the neighborhood of entities mentioned
    #     in the question
    #     """
    #     result = ""
    #     entities = entity_chain.invoke({"question": question})
    #     print("Entities detected: ", entities)
    #     for entity in entities.names:
    #         response = graph.query(
    #             """CALL db.index.fulltext.queryNodes('entity', $query, {limit:2})
    #             YIELD node,score
    #             CALL {
    #             MATCH (node)-[r:!MENTIONS]->(neighbor)
    #             RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
    #             UNION
    #             MATCH (node)<-[r:!MENTIONS]-(neighbor)
    #             RETURN neighbor.id + ' - ' + type(r) + ' -> ' +  node.id AS output
    #             }
    #             RETURN output LIMIT 50
    #             """,
    #             {"query": generate_full_text_query(entity)},
    #         )
    #         print("Full Text Query: ", generate_full_text_query(entity))
    #         result += "\n".join([el['output'] for el in response])
    #     print("Full Text Query Results: ", result)
    #     return result

    def retriever(question: str):
        print(f"Search query: {question}")
        # structured_data = structured_retriever(question)
        unstructured_data = vector_index.similarity_search(question)
        unstructured_page_content = [el.page_content for el in unstructured_data]
        unstructured_metadata = [(json.dumps(el.metadata)) for el in unstructured_data]
        print("Unstructured Data: ", unstructured_page_content)
        print("Metadata: ", unstructured_metadata)
        # final_data = f"""Structured data:
        #                 {structured_data}
        #                 Unstructured data:
        #                 {"#Document ". join(unstructured_data)}
        #             """
        final_data = f"""Unstructured data:
                        {"#Document ". join(unstructured_page_content)}
                        Metadata:
                        {"#Metadata ". join(unstructured_metadata)}
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
    Answer:
    #----
    Answer_Body
    #----
    At the end of each answer you should contain metadata for relevant document in the form of (source, page).
    For example, if context has `metadata`:(source:'docu_url', page:1), you should display:
    Metadata:
    - Document: 'doc_url', Page: 1 
    - Document: 'doc_url', Page: x
    """
    
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
