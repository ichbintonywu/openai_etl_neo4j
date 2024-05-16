import streamlit as st
import PyPDF2
from pathlib import Path
from graphdatascience import GraphDataScience
import json
from streamlit_extras.add_vertical_space import add_vertical_space
from openai import AzureOpenAI
    
AZURE_API_KEY = st.secrets["AZURE_API_KEY"]
AZUER_VERSION = st.secrets["AZUER_VERSION"]
AZURE_ENDPOINT = st.secrets["AZURE_ENDPOINT"]
AZURE_DEPLOYMENT_NAME = st.secrets["AZURE_DEPLOYMENT_NAME"]

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

gds = GraphDataScience(
    NEO4J_HOST,
    auth=(NEO4J_USER, NEO4J_PASSWORD))

gds.set_database("openaidb")
txt_path = 'output2.txt'

def gdsrun(query,db):
    gds = GraphDataScience(
        NEO4J_HOST,
        auth=(NEO4J_USER, NEO4J_PASSWORD))
    gds.set_database(db)
    return gds.run_cypher(query)

# Set up the prompt for GPT-3 to complete
# @retry(tries=3, delay=5)
def process_gpt4(text):
    paragraph = text

    completion = client.chat.completions.create(model=AZURE_DEPLOYMENT_NAME,
    # Try to be as deterministic as possible
    temperature=0,
    messages=[
        {"role": "system", "content": system},
        {"role": "user", "content": text},
    ])

    nlp_results = completion.choices[0].message.content
    
#     if not "relationships" in nlp_results:
#         raise Exception(
#             "GPT-4 is not being nice and isn't returning results in correct format"
#         )
    
    return (nlp_results)


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


st.markdown("**Please fill the below form :**")
with st.form(key="Choose a PDF file", clear_on_submit = True):
    File = st.file_uploader(label = "Upload file", type=["pdf"])
    Submit = st.form_submit_button(label='Submit')   


if Submit :
    st.markdown("**The file is sucessfully Uploaded.**")
    # Save uploaded file to 'F:/tmp' folder.
    save_folder = './'
    save_path = Path(save_folder, File.name)
    with open(save_path, mode='wb') as w:
        w.write(File.getvalue())
    # print(File.name)
    pdf_path = File.name
    
    str_text_extracted=extract_text_from_pdf(pdf_path, txt_path)

    file_path = txt_path  # Replace with the actual file path
    with open(file_path, 'r') as file:
        text = file.read()
    st.subheader("PDF file details : ")
    st.write(text)

with st.form(key="Turn the text file into a Knowledge Graph", clear_on_submit = True):
    database_list ="""SHOW DATABASES;"""
    db_list_fetched=gds.run_cypher(database_list)  
    db_list_fetched_list = db_list_fetched['name'].tolist()
    db_list_fetched_list.remove('system') 
    default_idx = db_list_fetched_list.index('neo4j')
    col1,col2 =st.columns(2)
    col1.selectbox("Select a database to store the extracted information from the PDF file:",index=default_idx, options=db_list_fetched_list,key="pickupuserdb")

    Submit_KG= st.form_submit_button(label='Submit') 
    
    if Submit_KG : 

        selectDB_name =  str(st.session_state["pickupuserdb"])
        print(selectDB_name)
        gds.set_database(selectDB_name)

        prompt ="""
        From the input text below, extract entity strictly as instructed below:

        1. First, look for and extract these properties for this new features of the product Neo4j 5
        2. for entity map, key should be entity, sub map should be using label and name
        3. for relationship map, it should be a tuple
        4. Do NOT create duplicate entities
        5. Summarize the entity properly including Enhancement, their detailed Feature and Description are all different entities
        6. Enhancement should not has relationship with Product, but should have relationship with Features
        7. NEVER Impute missing values
        8. The entity - "Neo4j 5" should only has relationship to entity - "Enhancement"

        Desired Output JSON:
        {"entity": [{"label":"Feature","name":"Optimized Query Plans"}]}
        {"entity": [{"label":"Product","name":"Neo4j 5"}]}
        {"entity": [{"label":"Enhancement","name":"Powerful queries"}]}
        {"relationships": [{"head_entity": {"label": "Product", "name": "Neo4j 5"},"relationship": "HAS_ENHANCEMENT","tail_entity": {"label": "Enhancement", "name": "Effortless Unbounded Scale"}}]}
        {"relationships": [{"head_entity": {"label": "Enhancement", "name": "Powerful queries"},"relationship": "HAS_FEATURE","tail_entity": {"label": "Feature", "name": "Faster K-Hop query"}}]}

        Text: {text}


        """  
        file_path = txt_path  # Replace with the actual file path
        with open(file_path, 'r') as file:
            text = file.read()
        response = process_gpt4(prompt+text)
        
        # The provided JSON map as a string
        json_map = response
        # Split the input into individual JSON strings
        json_strings = json_map.strip().split('\n')

        # Initialize lists to store parsed entities and relationships
        parsed_entities = []
        parsed_relationships = []

        # Iterate through each JSON string and parse it
        for json_string in json_strings:
            parsed_json = json.loads(json_string)
            if 'entity' in parsed_json:
                entity = parsed_json['entity'][0]
                parsed_entities.append(entity)
            elif 'relationships' in parsed_json:
                relationship = parsed_json['relationships'][0]
                parsed_relationships.append(relationship)

        # Print the parsed entities
        for entity in parsed_entities:
        #     print("Entity:", entity['label'], entity['name'])
            
            label = entity['label'] 
            properties = entity['name']
            gds_run_str= """
            CALL apoc.merge.node([\'"""+ label+"""\'],{name:\'"""+ properties +"""\'});

            """
            print(gds_run_str)
            gds.run_cypher(gds_run_str)  

        # Print the parsed relationships
        for relationship in parsed_relationships:
            head_entity = relationship['head_entity']
            relationship_type = relationship['relationship']
            tail_entity = relationship['tail_entity']

            head_label = head_entity['label']
            head_name = head_entity['name'] 
            tail_label = tail_entity['label']
            tail_name = tail_entity['name']
            relationship_type = relationship_type

            gds_run_str= """
            match (n) where n:"""+head_label+""" and n.name = \'"""+head_name+"""\'
            match (m) where m:"""+tail_label+""" and m.name = \'"""+tail_name+"""\'
            CALL apoc.merge.relationship(n, \'"""+ relationship_type +"""\',
            {},{created: datetime()},
            m,{}
            )
            YIELD rel
            RETURN rel;

            """
            print(gds_run_str)
            gds.run_cypher(gds_run_str)  

"---"
url = "https://neo4j.com/press-releases/announcing-neo4j-5/"
st.write("Check out this link for more information [link](%s)" % url)
