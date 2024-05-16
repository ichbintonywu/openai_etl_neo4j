from llmsherpa.readers import LayoutPDFReader
import glob
from neo4j import GraphDatabase
from datetime import datetime
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from typing import Tuple, List, Optional
from llmsherpa.readers import LayoutPDFReader
from typing import Dict
from logging import exception
import streamlit as st
from openai import AzureOpenAI
import hashlib
import google.generativeai as genai
import os
from neo4j import GraphDatabase
from openai import AzureOpenAI

pdf_file_name= '13QM.pdf'

file_location = "/Users/bryanlee/Documents/Petronas/rca_demo/myexpert/files"
pdf_url = pdf_file_name
pdf_files = glob.glob(pdf_url)
llmsherpa_api_url = "https://readers.llmsherpa.com/api/document/developer/parseDocument?renderFormat=all"
pdf_reader = LayoutPDFReader(llmsherpa_api_url)

# Please change the following variables to your own Neo4j instance
NEO4J_URL = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "yourpassword"
NEO4J_DATABASE = "yourdb"

PROJECT_ID = st.secrets["PROJECT_ID"] 
REGION = st.secrets["REGION"]
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"] 
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

os.environ["AZURE_OPENAI_API_KEY"] = st.secrets["AZURE_OPENAI_API_KEY"] 
os.environ["AZURE_OPENAI_ENDPOINT"] = st.secrets["AZURE_OPENAI_ENDPOINT"] 
os.environ["OPENAI_MODEL_NAME"] = st.secrets["OPENAI_MODEL_NAME"] 
os.environ["OPENAI_API_VERSION"] = st.secrets["OPENAI_API_VERSION"] 
os.environ["OPENAI_API_TYPE"] = st.secrets["OPENAI_API_TYPE"] 

llm=AzureChatOpenAI(temperature=0, api_version=os.getenv('OPENAI_API_VERSION'), azure_deployment=os.getenv('OPENAI_MODEL_NAME'))
EMBEDDING_MODEL = "text-embedding-ada-002"

def generateEntities(chunk):
    # Extract entities from chunks
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
                """
                You are an expert at extracting and recognizing entities from a paragraph of text using your engineering domain knowledge
                DO NOT MAKE THINGS UP, if there are no entities, just return an empty list
                DO NOT FAKE an extraction by giving a fake entity or a fake person name e.g. "John Doe"
                ## 1. ONLY EXTRACT entities that are of the following categories: "Equipment", "Pump", "Piping", "Shaft", "Company", "Person Name"
                ## 2. Do not extract any nouns, e.g. "is", "are", "who"
                ## 3. Do not extract any generic terms, e.g. "test", "solution", "work", "option 1", "option2"
                ## 4. Entity should not be a decimal, interger, float, number or a date
                ## 5. When the paragraph/text is of whitespaces, do not assign any entities to it
                ## 6. Do not extract units, e.g. "kg/m3", "cm2", "3.0 ton/hr"
                ## 7. Do not extract information that are not relevant and important to the engineering (pumps, equipments, plants), oil and gas and human domain knowledge
                ## 8. Coreference Resolution
                ## 8.1. Maintain Entity Consistency: When extracting entities, it's vital to ensure consistency.
                If an entity, such as "John Doe", is mentioned multiple times in the text but is referred to by different names or pronouns (e.g., "Joe", "he"),
                always use the most complete identifier for that entity throughout the knowledge graph. In this example, use "John Doe" as the entity ID.
                Remember, the knowledge graph should be coherent and easily understandable, so maintaining consistency in entity references is crucial.
                """,
            ),
            (
                "human",
                "Use the given format to extract information from the following "
                "input: {question}",
            ),
        ]
    )

    entity_chain = prompt | llm.with_structured_output(Entities)
    entities_list = entity_chain.invoke(chunk)

    return entities_list

## not used
def combineEntities(entities):
    class Entity(BaseModel):
        """Identifying information about entities."""
        entity: str
        label: str

    # Extract entities from chunks
    class Entities(BaseModel):
        """Identifying information about entities."""

        entities: List[Entity]

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are an expert at extracting and recognizing entities from a paragraph of text using your oil and gas and mechanical and petrochemical engineering domain knowledge
                DO NOT MAKE THINGS UP, if there are no entities, just return an empty list
                Based on the existing list of entities extracted:
                "list_of_entities": {full_list_of_entities}
                ## 1. Categorise the entities into different groups based on the semantics and domain of the list of entities by giving it a label, for example:
                [{{ "entity": "apple", "label": "fruit" }}]

                ## 2. Remove any entity that is too generic and doesn't add value to engineering, petrochemical, oil and gas domain. BE SPECIFIC!
                """,
            ),
            (
                "human",
                "Use the given format to extract information from the following "
                "input: {full_list_of_entities}",
            ),
        ]
    )

    entity_chain = prompt | llm.with_structured_output(Entities)
    entities_list = entity_chain.invoke({"full_list_of_entities": entities})

    return entities_list


def run_prompt(prompt_text, context_text):
    generative_multimodal_model = genai.GenerativeModel("gemini-1.0-pro-latest")
    responses = generative_multimodal_model.generate_content(f"{prompt_text}\n{context_text}", stream=False)
    return responses.candidates[0].content.parts[0].text

def summarize_table(table_text):
  return run_prompt("""
      You're an expert in extracting contents from structured HTML contents.
      You're given a table that is formatted in HTML format.
      Translate the table contents into structured and understandable content in natural language. 
      Remove all the "|" and html tags and dividers.
      For each row and column in the table, make sense of of the data in the table and translate it into an understandable text.
      """, table_text)

def initialiseNeo4j():
    cypher_schema = [
        "CREATE CONSTRAINT sectionKey IF NOT EXISTS FOR (c:Section) REQUIRE (c.key) IS UNIQUE;",
        "CREATE CONSTRAINT chunkKey IF NOT EXISTS FOR (c:Chunk) REQUIRE (c.key) IS UNIQUE;",
        "CREATE CONSTRAINT documentKey IF NOT EXISTS FOR (c:Document) REQUIRE (c.url_hash) IS UNIQUE;",
        "CREATE CONSTRAINT tableKey IF NOT EXISTS FOR (c:Table) REQUIRE (c.key) IS UNIQUE;",
        "CALL db.index.vector.createNodeIndex('chunkVectorIndex','Embedding','value',1536,'cosine');"
    ]

    driver = GraphDatabase.driver(NEO4J_URL, database=NEO4J_DATABASE, auth=(NEO4J_USER, NEO4J_PASSWORD))

    with driver.session() as session:
        for cypher in cypher_schema:
            session.run(cypher)
    driver.close()

def ingestDocumentNeo4j(doc, doc_location):
    cypher_pool = [
        # 0 - Document
        "MERGE (d:Document {url_hash: $doc_url_hash_val}) ON CREATE SET d.url = $doc_url_val RETURN d;",
        # 1 - Section
        "MERGE (p:Section {key: $doc_url_hash_val+'|'+$block_idx_val+'|'+$title_hash_val}) ON CREATE SET p.page_idx = $page_idx_val, p.title_hash = $title_hash_val, p.block_idx = $block_idx_val, p.title = $title_val, p.tag = $tag_val, p.level = $level_val RETURN p;",
        # 2 - Link Section with the Document
        "MATCH (d:Document {url_hash: $doc_url_hash_val}) MATCH (s:Section {key: $doc_url_hash_val+'|'+$block_idx_val+'|'+$title_hash_val}) MERGE (d)<-[:HAS_DOCUMENT]-(s);",
        # 3 - Link Section with a parent section
        "MATCH (s1:Section {key: $doc_url_hash_val+'|'+$parent_block_idx_val+'|'+$parent_title_hash_val}) MATCH (s2:Section {key: $doc_url_hash_val+'|'+$block_idx_val+'|'+$title_hash_val}) MERGE (s1)<-[:UNDER_SECTION]-(s2);",
        # 4 - Chunk
        """
        WITH $chunk_entities as chunk_entities
        MERGE (c:Chunk {key: $doc_url_hash_val+'|'+$block_idx_val+'|'+$sentences_hash_val})
        ON CREATE SET c.sentences = $sentences_val, c.sentences_hash = $sentences_hash_val, c.block_idx = $block_idx_val, c.page_idx = $page_idx_val, c.tag = $tag_val, c.level = $level_val
        WITH c, chunk_entities
        UNWIND chunk_entities as chunk_entity
        MERGE (e:Entity { name: toLower(chunk_entity) })
        MERGE (c)-[:HAS_ENTITY]->(e)
        RETURN c;
        """,
        # 5 - Link Chunk to Section
        "MATCH (c:Chunk {key: $doc_url_hash_val+'|'+$block_idx_val+'|'+$sentences_hash_val}) MATCH (s:Section {key:$doc_url_hash_val+'|'+$parent_block_idx_val+'|'+$parent_hash_val}) MERGE (s)<-[:HAS_PARENT]-(c);",
        # 6 - Table
        """
        MERGE (t:Table {key: $doc_url_hash_val+'|'+$block_idx_val+'|'+$name_val})
        ON CREATE SET t.name = $name_val, t.doc_url_hash = $doc_url_hash_val, t.block_idx = $block_idx_val, t.page_idx = $page_idx_val, t.html = $html_val, t.rows = $rows_val 
        WITH t
        MERGE (c:Chunk {key: $doc_url_hash_val+'|'+$block_idx_val+'|'+$name_val})
        ON CREATE SET c.sentences = $summarize_text, c.block_idx = $block_idx_val, c.page_idx = $page_idx_val
        MERGE (c)-[:HAS_PARENT]->(t)
        RETURN t;
        """,
        # 7 - Link Table to Section
        "MATCH (t:Table {key: $doc_url_hash_val+'|'+$block_idx_val+'|'+$name_val}) MATCH (s:Section {key: $doc_url_hash_val+'|'+$parent_block_idx_val+'|'+$parent_hash_val}) MERGE (s)<-[:HAS_PARENT]-(t);",
        # 8 - Link Table to Document if no parent section
        "MATCH (t:Table {key: $doc_url_hash_val+'|'+$block_idx_val+'|'+$name_val}) MATCH (s:Document {url_hash: $doc_url_hash_val}) MERGE (s)<-[:HAS_PARENT]-(t);"
        # 9 - vector search and return list of section, url of document
        """MATCH (n:Embedding) where id(n)=274 ##### !!!!! replaced with vector index search
            with n
            match (s:Section)-[*1..3]-(n) with s
            match (s)--(c:Chunk)
            with s,collect(c.sentences) as col1
            with apoc.coll.sort(col1) as col2,s
            match (d:Document)--{1,2}(s)
            return col2,d.url;"""
    ]

    driver = GraphDatabase.driver(NEO4J_URL, database=NEO4J_DATABASE, auth=(NEO4J_USER, NEO4J_PASSWORD))

    with driver.session() as session:
        cypher = ""

        # 1 - Create Document node
        doc_url_val = doc_location
        doc_url_hash_val = hashlib.md5(doc_url_val.encode("utf-8")).hexdigest()

        cypher = cypher_pool[0]
        session.run(cypher, doc_url_hash_val=doc_url_hash_val, doc_url_val=doc_url_val)

        # 2 - Create Section nodes

        countSection = 0
        for sec in doc.sections():
            sec_title_val = sec.title
            sec_title_hash_val = hashlib.md5(sec_title_val.encode("utf-8")).hexdigest()
            sec_tag_val = sec.tag
            sec_level_val = sec.level
            sec_page_idx_val = sec.page_idx
            sec_block_idx_val = sec.block_idx

            # MERGE section node
            if not sec_tag_val == 'table':
                cypher = cypher_pool[1]
                session.run(cypher, page_idx_val=sec_page_idx_val
                                , title_hash_val=sec_title_hash_val
                                , title_val=sec_title_val
                                , tag_val=sec_tag_val
                                , level_val=sec_level_val
                                , block_idx_val=sec_block_idx_val
                                , doc_url_hash_val=doc_url_hash_val
                            )

                # Link Section with a parent section or Document

                sec_parent_val = str(sec.parent.to_text())

                if sec_parent_val == "None":    # use Document as parent

                    cypher = cypher_pool[2]
                    session.run(cypher, page_idx_val=sec_page_idx_val
                                    , title_hash_val=sec_title_hash_val
                                    , doc_url_hash_val=doc_url_hash_val
                                    , block_idx_val=sec_block_idx_val
                                )

                else:   # use parent section
                    sec_parent_title_hash_val = hashlib.md5(sec_parent_val.encode("utf-8")).hexdigest()
                    sec_parent_page_idx_val = sec.parent.page_idx
                    sec_parent_block_idx_val = sec.parent.block_idx

                    cypher = cypher_pool[3]
                    session.run(cypher, page_idx_val=sec_page_idx_val
                                    , title_hash_val=sec_title_hash_val
                                    , block_idx_val=sec_block_idx_val
                                    , parent_page_idx_val=sec_parent_page_idx_val
                                    , parent_title_hash_val=sec_parent_title_hash_val
                                    , parent_block_idx_val=sec_parent_block_idx_val
                                    , doc_url_hash_val=doc_url_hash_val
                                )
            # **** if sec_parent_val == "None":

            countSection += 1
        # **** for sec in doc.sections():


        # ------- Continue within the blocks -------
        # 3 - Create Chunk nodes from chunks

        countChunk = 0
        for chk in doc.chunks():

            chunk_block_idx_val = chk.block_idx
            chunk_page_idx_val = chk.page_idx
            chunk_tag_val = chk.tag
            chunk_level_val = chk.level
            chunk_sentences = "\n".join(chk.sentences)

            # add logic here to extract entities
            chunk_entities = generateEntities(chunk_sentences)
            chunk_entities_names = chunk_entities.names
            print("Chunk Sentences: ", chunk_sentences)
            print("Chunk Entities: ", chunk_entities_names)
            print("\n")

            # MERGE Chunk node
            if not chunk_tag_val == 'table':
                chunk_sentences_hash_val = hashlib.md5(chunk_sentences.encode("utf-8")).hexdigest()

                # MERGE chunk node
                cypher = cypher_pool[4]
                session.run(cypher, sentences_hash_val=chunk_sentences_hash_val
                                , sentences_val=chunk_sentences
                                , block_idx_val=chunk_block_idx_val
                                , page_idx_val=chunk_page_idx_val
                                , tag_val=chunk_tag_val
                                , level_val=chunk_level_val
                                , doc_url_hash_val=doc_url_hash_val
                                , chunk_entities=chunk_entities_names
                            )

                # Link chunk with a section
                # Chunk always has a parent section
                chk_parent_val = str(chk.parent.to_text())

                if not chk_parent_val == "None":
                    chk_parent_hash_val = hashlib.md5(chk_parent_val.encode("utf-8")).hexdigest()
                    chk_parent_page_idx_val = chk.parent.page_idx
                    chk_parent_block_idx_val = chk.parent.block_idx

                    cypher = cypher_pool[5]
                    session.run(cypher, sentences_hash_val=chunk_sentences_hash_val
                                    , block_idx_val=chunk_block_idx_val
                                    , parent_hash_val=chk_parent_hash_val
                                    , parent_block_idx_val=chk_parent_block_idx_val
                                    , doc_url_hash_val=doc_url_hash_val
                                )

                # Link sentence
                #   >> TO DO for smaller token length

                countChunk += 1
        # **** for chk in doc.chunks():

        # 4 - Create Table nodes


    # for tb in doc.tables():
    #   page_idx_val = tb.page_idx
    #   block_idx_val = tb.block_idx
    #   name_val = 'block#' + str(block_idx_val) + '_' + tb.name
    #   html_val = tb.to_html()
    #   rows_val = len(tb.rows)
    #   summarize_text = summarize_table("Html Content: " + html_val)
    #   print(summarize_text)

        countTable = 0
        for tb in doc.tables():
            page_idx_val = tb.page_idx
            block_idx_val = tb.block_idx
            name_val = 'block#' + str(block_idx_val) + '_' + tb.name
            html_val = tb.to_html()
            rows_val = len(tb.rows)
            summarize_text = summarize_table(html_val)

            # MERGE table node
            cypher = cypher_pool[6]
            session.run(cypher, block_idx_val=block_idx_val
                            , page_idx_val=page_idx_val
                            , name_val=name_val
                            , html_val=html_val
                            , rows_val=rows_val
                            , doc_url_hash_val=doc_url_hash_val
                            , summarize_text=summarize_text
                        )

            # Link table with a section
            # Table always has a parent section

            table_parent_val = str(tb.parent.to_text())

            if not table_parent_val == "None":
                table_parent_hash_val = hashlib.md5(table_parent_val.encode("utf-8")).hexdigest()
                table_parent_page_idx_val = tb.parent.page_idx
                table_parent_block_idx_val = tb.parent.block_idx

                cypher = cypher_pool[7]
                session.run(cypher, name_val=name_val
                                , block_idx_val=block_idx_val
                                , parent_page_idx_val=table_parent_page_idx_val
                                , parent_hash_val=table_parent_hash_val
                                , parent_block_idx_val=table_parent_block_idx_val
                                , doc_url_hash_val=doc_url_hash_val
                            )

            else:   # link table to Document
                cypher = cypher_pool[8]
                session.run(cypher, name_val=name_val
                                , block_idx_val=block_idx_val
                                , doc_url_hash_val=doc_url_hash_val
                            )
            countTable += 1

        # **** for tb in doc.tables():

        print(f'\'{doc_url_val}\' Done! Summary: ')
        print('#Sections: ' + str(countSection))
        print('#Chunks: ' + str(countChunk))
        print('#Tables: ' + str(countTable))

    driver.close()

def get_embedding(client, text, model):
    response = client.embeddings.create(
                    input=text,
                    model=model,
                )
    return response.data[0].embedding

def LoadEmbedding(label, property):
    driver = GraphDatabase.driver(NEO4J_URL, auth=(NEO4J_USER, NEO4J_PASSWORD), database=NEO4J_DATABASE)
    #openai_client = OpenAI (api_key = OPENAI_KEY)

    openai_client = AzureOpenAI(
      api_key = os.getenv("AZURE_OPENAI_API_KEY"),
      api_version = "2024-02-01",
      azure_endpoint =os.getenv("AZURE_OPENAI_ENDPOINT")
    )

    with driver.session() as session:
        # get chunks in document, together with their section titles
        result = session.run(f"MATCH (ch:{label}) -[:HAS_PARENT]-> (s:Section) RETURN id(ch) AS id, s.title + ' >> ' + ch.{property} AS text")
        # call OpenAI embedding API to generate embeddings for each proporty of node
        # for each node, update the embedding property
        count = 0
        for record in result:
            id = record["id"]
            text = record["text"]

            # For better performance, text can be batched
            embedding = get_embedding(openai_client, text, EMBEDDING_MODEL)

            # key property of Embedding node differentiates different embeddings
            cypher = "CREATE (e:Embedding) SET e.key=$key, e.value=$embedding"
            cypher = cypher + " WITH e MATCH (n) WHERE id(n) = $id CREATE (n) -[:HAS_EMBEDDING]-> (e)"
            session.run(cypher,key=property, embedding=embedding, id=id )
            count = count + 1

        session.close()

        print("Processed " + str(count) + " " + label + " nodes for property @" + property + ".")
        return count

st.markdown("**Please fill the below form :**")
with st.form(key="Choose a PDF file", clear_on_submit = False):
    File = st.file_uploader(label = "Upload file", type=["pdf"])

    if st.form_submit_button("Upload the PDF File"):
        pdf_file_name = File.name   
        pdf_files = glob.glob(pdf_file_name)

        print(f'#PDF files found: {len(pdf_files)}!')
        pdf_reader = LayoutPDFReader(llmsherpa_api_url)

        # parse documents and create graph
        startTime = datetime.now()

        for pdf_file in pdf_files:
            print(pdf_file)
        try:
            doc = pdf_reader.read_pdf(pdf_file)

            # find the first / in pdf_file from right
            idx = pdf_file.rfind('/')
            pdf_file_name = pdf_file[idx+1:]

            # open a local file to write the JSON
            with open(pdf_file_name + '.json', 'w') as f:
            # convert doc.json from a list to string
                f.write(str(doc.json))

            ingestDocumentNeo4j(doc, pdf_file)

            for tb in doc.tables():
                page_idx_val = tb.page_idx
                block_idx_val = tb.block_idx
                name_val = 'block#' + str(block_idx_val) + '_' + tb.name
                html_val = tb.to_html()
                rows_val = len(tb.rows)
                summarize_text = summarize_table("Html Content: " + html_val)
                print(summarize_text)

        except Exception as e:
            print("Error: ", e)

        print(f'Total time: {datetime.now() - startTime}')

        initialiseNeo4j()
        LoadEmbedding("Chunk", "sentences")
