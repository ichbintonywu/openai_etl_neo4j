import streamlit as st
import PyPDF2
from pathlib import Path
from streamlit_extras.add_vertical_space import add_vertical_space
from openai import AzureOpenAI
import os
from langchain_community.graphs import Neo4jGraph  
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import AzureChatOpenAI
from langchain_core.documents import Document

AZURE_API_KEY = st.secrets["AZURE_API_KEY"]
AZUER_VERSION = st.secrets["AZUER_VERSION"]
AZURE_ENDPOINT = st.secrets["AZURE_ENDPOINT"]
AZURE_DEPLOYMENT_NAME = st.secrets["AZURE_DEPLOYMENT_NAME"]

os.environ["AZURE_OPENAI_API_KEY"] = st.secrets["AZURE_OPENAI_API_KEY"] 
os.environ["AZURE_OPENAI_ENDPOINT"] = st.secrets["AZURE_OPENAI_ENDPOINT"] 
os.environ["OPENAI_MODEL_NAME"] = st.secrets["OPENAI_MODEL_NAME"] 
os.environ["OPENAI_API_VERSION"] = st.secrets["OPENAI_API_VERSION"] 
os.environ["OPENAI_API_TYPE"] = st.secrets["OPENAI_API_TYPE"] 

llm = AzureChatOpenAI(temperature=0, model_name="gpt-4-32k",api_version=os.getenv('OPENAI_API_VERSION'), azure_deployment=os.getenv('OPENAI_MODEL_NAME'))
# llm=AzureChatOpenAI(temperature=0, api_version=os.getenv('OPENAI_API_VERSION'), azure_deployment=os.getenv('OPENAI_MODEL_NAME'))
llm_transformer = LLMGraphTransformer(llm=llm)

client = AzureOpenAI(
    api_key = AZURE_API_KEY,  
    api_version = AZUER_VERSION,
    azure_endpoint = AZURE_ENDPOINT
    ) 
deployment_name = AZURE_DEPLOYMENT_NAME

system = "You are an entity and relationship extraction expert helping us extract relevant information."

NEO4J_HOST = "bolt://"+st.secrets["NEO4J_HOST"]+":"+st.secrets["NEO4J_PORT"] 
NEO4J_USER = st.secrets["NEO4J_USER"]
NEO4J_PASSWORD = st.secrets["NEO4J_PASSWORD"]
NEO4J_DATABASE = 'openaiconstructpdf'
os.environ["NEO4J_URI"] = NEO4J_HOST
os.environ["NEO4J_USERNAME"] = NEO4J_USER
os.environ["NEO4J_PASSWORD"] = NEO4J_PASSWORD

graph = Neo4jGraph(database = NEO4J_DATABASE)

txt_path = 'output2.txt'

def extract_text_from_pdf(pdf_path, txt_path):
    try:
        with open(pdf_path, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            text = ""
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num] 
                text += page.extract_text()
            
            with open(txt_path, 'w', encoding='utf-8') as txt_file:
                txt_file.write(text)
                
            print("Text extracted and saved successfully.")
    except Exception as e:
        print("An error occurred:", str(e))

st.set_page_config(page_title="PDF Converter Demo",layout="wide", page_icon="📈")
st.image('Neo4j-5-Feature-1024x971.png', caption='What is new in Neo4j 5?')

if st.session_state.get('nodes_list_state') is None:
    st.session_state['nodes_list_state'] = []
if st.session_state.get('rels_list_state') is None:
    st.session_state['rels_list_state'] = []

txt_node_list = st.text_input(
"Determinate nodes",
"Software,Enhancement,Feature",
)
txt_rel_list = st.text_input(
"Determinate relationships",
"INCLUDES",
)

st.markdown("**Please fill the below form :**")
with st.form(key="Choose a PDF file", clear_on_submit = False):
    File = st.file_uploader(label = "Upload file", type=["pdf"])

    if st.form_submit_button("Upload the PDF File"):
        st.markdown("**The file is sucessfully Uploaded.**")
        save_folder = './'
        save_path = Path(save_folder, File.name)
        with open(save_path, mode='wb') as w:
            w.write(File.getvalue())

        pdf_path = File.name   

        str_text_extracted=extract_text_from_pdf(pdf_path, txt_path)
        file_path = txt_path 
        with open(file_path, 'r') as file:
            text = file.read()
        st.subheader("PDF nodes/relationships extraction and construction details : ")

        documents = [Document(page_content=text)]
        graph_documents = llm_transformer.convert_to_graph_documents(documents)

        node_set = set()
        relationship_set = set()
        for node in graph_documents[0].nodes:
            node_set.add(node.type)
        for relationship in graph_documents[0].relationships:
            relationship_set.add(relationship.type)
        node_list=list(node_set)
        relationship_list=list(relationship_set)
        #determined nodes and relationships
        st.session_state['nodes_list_state']=node_list
        st.session_state['rels_list_state']=relationship_list

        print(f"Nodes list: {node_list}")
        print(f"Relationships Set:{relationship_list}")
        st.text("LLM suggested node labels:"+ str(node_list))
        st.text("LLM suggested relationship types:"+str(relationship_list))

    if st.form_submit_button("Submit to contruct KG with determinism"):
        pdf_path = File.name
        str_text_extracted=extract_text_from_pdf(pdf_path, txt_path)

        file_path = txt_path  # Replace with the actual file path
        with open(file_path, 'r') as file:
            text = file.read()
        st.subheader("PDF nodes/relationships extraction and construction details : ")

        documents = [Document(page_content=text)]
        graph_documents = llm_transformer.convert_to_graph_documents(documents)

        node_set = set()
        relationship_set = set()
        for node in graph_documents[0].nodes:
            node_set.add(node.type)
        for relationship in graph_documents[0].relationships:
            relationship_set.add(relationship.type)
        node_list=list(node_set)
        relationship_list=list(relationship_set)

        nodes_list = [s.strip('" ').strip() for s in txt_node_list.split(',')]
        relationships_list =  [s.strip('" ').strip() for s in txt_rel_list.split(',')] 

        st.write("LLM generated nodes are:"+str(st.session_state['nodes_list_state']))
        st.write("LLM generated relationships are:"+str(st.session_state['rels_list_state']))
        st.write("Determinated nodes are:"+str(nodes_list))
        st.write("Determinated relationships are:"+str(relationships_list))

        combined_nodes = list(set(nodes_list))
        combined_relationships = list(set(relationships_list))

        print(f"Nodes list after deterministic decision: {combined_nodes}")
        print(f"Relationships list after deterministic decision:{combined_relationships}")   
        llm_transformer_filtered = LLMGraphTransformer(
        llm=llm,
        allowed_nodes=combined_nodes,
        allowed_relationships=combined_relationships,
        )
        graph_documents_filtered = llm_transformer_filtered.convert_to_graph_documents(
            documents
        )
        st.write(f"Nodes:{graph_documents_filtered[0].nodes}")
        st.write(f"Relationships:{graph_documents_filtered[0].relationships}")

        graph.add_graph_documents(graph_documents_filtered)

"---"
url = "https://python.langchain.com/v0.1/docs/use_cases/graph/constructing/"
st.write("Check out this link for more information [link](%s)" % url)


