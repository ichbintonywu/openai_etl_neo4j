from neo4j import GraphDatabase
from graphdatascience import GraphDataScience
import streamlit as st
import json
from openai import AzureOpenAI
    
AZURE_API_KEY = st.secrets["AZURE_API_KEY"]
AZUER_VERSION = st.secrets["AZUER_VERSION"]
AZURE_ENDPOINT = st.secrets["AZURE_ENDPOINT"]
AZURE_DEPLOYMENT_NAME = st.secrets["AZURE_DEPLOYMENT_NAME"]
AZURE_MODEL_EMBEDDING = st.secrets["AZURE_MODEL_EMBEDDING"]

client = AzureOpenAI(
    api_key = AZURE_API_KEY,  
    api_version = AZUER_VERSION,
    azure_endpoint = AZURE_ENDPOINT
    ) 
deployment_name = AZURE_DEPLOYMENT_NAME

host = "bolt://"+ st.secrets["NEO4J_HOST"]+":"+st.secrets["NEO4J_PORT"]
user = st.secrets["NEO4J_USER"]
password = st.secrets["NEO4J_PASSWORD"]
db = st.secrets["NEO4J_DB"]

# print(db)
URI = host
AUTH = (user, password)
driver = GraphDatabase.driver(URI, auth=AUTH)

def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input = [text], model=AZURE_MODEL_EMBEDDING).data[0].embedding

def extractNER(question):
    response = client.chat.completions.create(
    model="sabikiGPT4Deployment",  
    messages=[
        {"role": "system", "content": "You help extract all relevant entities from the question, and put entities in a list of Json map using format {\"entity\": \"entity details\", \"type\": \"type details\"}"},
        {"role": "user", "content": question},
    ],
    temperature=0,
    max_tokens=100
    )
    # Get the generated text
    generated_text = response.choices[0].message.content
    entities = json.loads(generated_text)
    sentence = ""

    for entry in entities:
        dict_values = entry.values()
        value_list = list(dict_values)
        sentence = sentence+value_list[0]+" " +value_list[1] +" "
    print(sentence)
    
    question_embedding = get_embedding(sentence)
    print (len(question_embedding))
    vector_index_search_str =f"""CALL db.index.vector.queryNodes(
    'abstract-embeddings', // index name
    2, // topK neighbors to return
    {question_embedding}// input vector
    )
    YIELD node, score
    with node, score
    match (filename)--(node)
    with filename.filename as filename, node,score
    optional match (node)-[:NEXT_CHUNK]->(contextnode1)
    optional match (contextnode2)-[:NEXT_CHUNK]->(node)
    with filename,node,contextnode1,contextnode2,score where filename is not NULL
    with  filename,node.chunk_content as content,contextnode1.chunk_content as context, contextnode2.chunk_content as context2,score
    with filename, coalesce(context,' ')+' , '+content+' , ' +coalesce(context2,' ') as content,score limit 1
    with filename, apoc.text.replace(content,'\n','.') as content,score
    return filename,content,score
    """
    print (vector_index_search_str)
    vector_index_query_run=gdsrun(vector_index_search_str)
    print(vector_index_query_run)
    return vector_index_query_run

    # Initialize spaCy for named entity recognition
    # nlp = spacy.load("en_core_web_sm")

    # Process the generated text with spaCy
    # doc = nlp(generated_text)
    # doc = nlp(question)
    # Extract entities
    # entities = []
    # for ent in doc.ents:
    #     entities.append({"text": ent.text, "label": ent.label_})
    # print("Entities found:", entities)

def gdsrun(query):
    gds = GraphDataScience(
        host,
        auth=(user, password))
    gds.set_database(db)
    return gds.run_cypher(query)

def gdsrun_db(query,indb):
    gds = GraphDataScience(
        host,
        auth=(user, password))
    gds.set_database(indb)
    return gds.run_cypher(query)

def pairwise_list(input_list):
    return [(input_list[i], input_list[i + 1]) for i in range(len(input_list) - 1)]

def stitchDoc_Chunks(doc_name,Chunk_label,chunks):
    merge_str_pre =f"""MERGE (n:`{doc_name}`"""+"""{filename:"""+f"""'{doc_name}'"""+"""}) MERGE(n)-[r:HAS_CHUNKS]->
    """
    for chunk in chunks:
        openaiembedding = get_embedding(chunk) 
        escaped_chunk = chunk.replace("'", "\\'")
        merge_str= merge_str_pre+f"""(:{Chunk_label}"""+ \
        """{"""+f"""chunk_content:'{escaped_chunk}"""+ \
        """',""" + \
        f"""openai_embedding:{openaiembedding}""" + \
        """})"""

        # print (merge_str)

        get_mergemain_run=gdsrun(merge_str)
    paired_chunks= pairwise_list(chunks)
    for pair in paired_chunks:
        escaped_start_chunk= pair[0].replace("'", "\\'")
        escaped_end_chunk= pair[1].replace("'", "\\'")
        merge_str= f"""match (s:{Chunk_label}"""+ \
        """{"""+f"""chunk_content:'{escaped_start_chunk}"""+ \
        """'})"""+f""" match (d:{Chunk_label}""" + """{"""+\
        f"""chunk_content:'{escaped_end_chunk}""" + \
        """'})"""+\
        """ merge (s)-[:NEXT_CHUNK]->(d)""" 

        get_mergenext_run=gdsrun(merge_str)


    vector_index_drop_str = f"""
    drop index `abstract-embeddings`
    """
    vector_index_deletion_run=gdsrun(vector_index_drop_str)
    vector_index_creation_str = f"""
        call db.index.vector.createNodeIndex('abstract-embeddings','{Chunk_label}','openai_embedding',1536,'cosine')
        """
    vector_index_creation_run=gdsrun(vector_index_creation_str)

def stitchDoc_Chunks_ollama(doc_name,Chunk_label,chunks,indb):
    merge_str_pre =f"""MERGE (n:`{doc_name}`"""+"""{filename:"""+f"""'{doc_name}'"""+"""}) """
    merge_str_post= """MERGE (n)-[r:HAS_CHUNKS]->(c)
    """
    for chunk in chunks:
        # escaped_chunk = chunk.replace("'", "\\'")
        escaped_chunk = chunk
        merge__str = merge_str_pre+f""" WITH n MATCH (c:{Chunk_label}"""+ \
        """{"""+f"""text:'{escaped_chunk}"""+ \
        """'}) """+ merge_str_post
        get_mergemain_run=gdsrun_db(merge__str,indb)

    paired_chunks= pairwise_list(chunks)
    for pair in paired_chunks:
        # escaped_start_chunk= pair[0].replace("'", "\\'")
        # escaped_end_chunk= pair[1].replace("'", "\\'")
        escaped_start_chunk= pair[0]
        escaped_end_chunk= pair[1]
        merge_str= f"""match (s:{Chunk_label}"""+ \
        """{"""+f"""text:'{escaped_start_chunk}"""+ \
        """'})"""+f""" match (d:{Chunk_label}""" + """{"""+\
        f"""text:'{escaped_end_chunk}""" + \
        """'})"""+\
        """ merge (s)-[:NEXT_CHUNK]->(d)""" 
        get_mergenext_run=gdsrun_db(merge_str,indb)

def do_cypher_tx(tx,cypher):
    results = tx.run(cypher)
    values = []
    for record in results:
        values.append(record.values())
    return values

# @st.cache_resource
def exec_cypher_query(qry_str):
    with driver.session() as session:
        result = session.execute_read(do_cypher_tx,qry_str)
        return result

def write_movie_tx(tx, label1,label1_prop_name,label1_prop_property,
                   label2,label2_prop_name,label2_prop_property,
                   rel_type,rel_prop_name,rel_prop_property):
    merge_str ="""
    MATCH (n:""" +label1 + """ {"""+label1_prop_name+""":"""+"\""+label1_prop_property+"\""+"""})
    MATCH (m:""" +label2 + """ {"""+label2_prop_name+""":"""+"\""+label2_prop_property+"\""+"""})
    MERGE (n)"""+ \
    """-[r:"""+rel_type+""" {"""+rel_prop_name+""":'"""+rel_prop_property+ \
    """'}]-> (m)
    """

    query = (merge_str)
    print(query)
    result = tx.run(query, label1=label1,label1_prop_name=label1_prop_name,label1_prop_property=label1_prop_property,
                   label2=label2,label2_prop_name=label2_prop_name,label2_prop_property=label2_prop_property,
                   rel_type=rel_type,rel_prop_name=rel_prop_name,rel_prop_property=rel_prop_property)
    record = result.single()
    return "SUCCESS"

def simple_write_tx(tx, cypher_write):
    query = (cypher_write)
    result = tx.run(query, )
    record = result.single()
    return "SUCCESS"

def exec_cypher_write(label1,label1_prop_name,label1_prop_property,
                   label2,label2_prop_name,label2_prop_property,
                   rel_type,rel_prop_name,rel_prop_property):
    with driver.session() as session:
        cpyher_write_result = session.execute_write(write_movie_tx, label1,label1_prop_name,label1_prop_property,
                   label2,label2_prop_name,label2_prop_property,
                   rel_type,rel_prop_name,rel_prop_property)
        return cpyher_write_result

def exec_simple_cypher_write(cypher_write):
 with driver.session() as session:
        cpyher_write_result = session.execute_write(simple_write_tx,cypher_write)
        return cpyher_write_result   