import streamlit as st
from streamlit_chat import message
from streamlit_extras.colored_header import colored_header
import os
import sys
sys.path.append("./_drivers")

from neo4j_handler import extractNER
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
deployment_name=AZURE_DEPLOYMENT_NAME


def get_text():
    input_text = st.text_input("You: ", "", key="input")
    return input_text

st.set_page_config(page_title="Chatbot Demo",layout="wide", page_icon="📈")
st.title("Chatbot like a composer")

for key in st.session_state.keys():
    del st.session_state[key]
    
# Set a default model
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = AZURE_DEPLOYMENT_NAME

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is your question to ask from chunked document?"):
    # Add user message to chat history
    prompt_user=prompt
    intention = prompt_user + " Please use the content column from the dataframe below to answer, and please mention the filename from the dataframe \n " + str(extractNER(prompt))
    st.session_state.messages.append({"role": "user", "content": intention})

    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt_user)  

    # st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model=st.session_state["openai_model"],
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
        )
        response = st.write_stream(stream)        
    
    st.session_state.messages.append({"role": "assistant", "content": response})

colored_header(label='', description='', color_name='blue-30')        
st.markdown("""
<style>
table {
    width: 100%;
    border-collapse: collapse;
    border: none !important;
    font-family: "Source Sans Pro", sans-serif;
    color: rgba(49, 51, 63, 0.6);
    font-size: 0.9rem;
}

tr {
    border: none !important;
}

th {
    text-align: center;
    colspan: 3;
    border: none !important;
    color: #0F9D58;
}
                
th, td {
    padding: 2px;
    border: none !important;
}
</style>

<table>
<tr>
    <th colspan="3">Sample Questions to try out as tests</th>
</tr>
<tr>
    <td>Tell me about Any-to-any rolling upgrade?</td>    
    <td>What is Performance excellency Multiple hops for optimal path?</td> 
</tr>
<tr>
    <td>What is Point-in-time restore?</td>
</tr>
</table>
""", unsafe_allow_html=True)