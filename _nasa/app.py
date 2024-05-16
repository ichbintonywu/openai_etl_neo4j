import streamlit as st
import requests
import os
import pandas as pd
# Load environment variables from .env file
from dotenv import load_dotenv, find_dotenv
from PIL import Image
from timeit import default_timer as timer
from datetime import timedelta
import re
import asyncio
from streamlit_float import *
from langchain.globals import set_llm_cache
from langchain_community.cache import InMemoryCache
import sys
sys.path.append("./_nasa")

import read_hybrid
import read_vector

set_llm_cache(InMemoryCache())

# Set the OpenAI API key
os.environ["AZURE_OPENAI_API_KEY"] = ""
os.environ["AZURE_OPENAI_ENDPOINT"] = ""
os.environ["OPENAI_MODEL_NAME"] = ""
os.environ["OPENAI_API_VERSION"] = ""
os.environ["OPENAI_API_TYPE"] = ""

st.set_page_config(
    page_title="Vector only vs KG + vector ",
    page_icon="🚀",
    layout="wide",
)

#float_init(theme=True, include_unstable_primary=False)

# Set the title of the app
st.title('Vector only vs KG + vector 🚀')

if 'uploaded_data' not in st.session_state:
    # Read the file and store it in session state
    st.session_state.uploaded_data = False

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "hybrid_messages" not in st.session_state:
    st.session_state.hybrid_messages = []

if "topic_messages" not in st.session_state:
    st.session_state.topic_messages = []

if "topic_hybrid_messages" not in st.session_state:
    st.session_state.topic_hybrid_messages = []

tab1, tab2 = st.tabs(["Chatbot", "Help"])
# Main content
# Logic for each option
with tab1:
    st.subheader("Let's start exploring your data")

    # Display chat messages from history on app rerun   
    container_tab1 = st.container(height=600)    
    for message in st.session_state.hybrid_messages:
        print("Message: ", message)
        with container_tab1.chat_message(message["role"]):
            # new addition
            if message["role"] == "assistant":
                st.write(message["content"])
            if message["role"] == "user":
                st.write(message["content"])   

    if prompt := st.chat_input("Start chatting!"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt, "mode": "prompt"})
        st.session_state.hybrid_messages.append({"role": "user", "content": prompt, "mode": "prompt"})
        # Display user message in chat message container
        with container_tab1.chat_message("user"):
            st.write(prompt)

        # for hybrid search
        with container_tab1.chat_message("assistant"):
            res_hybrid = read_hybrid.query(prompt)
            print("RES_HYBRID: ", res_hybrid)
            st.write(res_hybrid)
        st.session_state.hybrid_messages.append({"role": "assistant", "content": res_hybrid})
        
        # for vector search 
        # results will be shown in tab2, not in tab1
        res_vector = read_vector.query(prompt)
        print("RES_VECTOR: ", res_vector)
        st.session_state.messages.append({"role": "assistant", "content": res_vector})
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
        <th colspan="3">Sample Questions to try out</th>
    </tr>
    <tr>
        <td>what are the driving events for the loss of electrical power during the mariner 64 mission?</td>
        <td>Do we need Power Subsystem Spares?</td>
        <td>What are the recommendations when facing DDM power supply failure?</td>
    </tr>
    <tr>
        <td>What were the mars observer mission failures?</td>
        <td>Why is the exterior of the mars observer dirty?</td>
        <td>What are the consequences of a dirty exterior on mars observer?</td>
    </tr>
    <tr>
        <td></td>
        <td></td>
    </tr>
    </table>
    """, unsafe_allow_html=True)

with tab2:
    st.subheader("Let's start exploring your data")

    # Create columns for layout
    col1, col2 = st.columns(2)

    # Display chat messages from history on app rerun
    with col1:
        st.subheader("Vector Only approach")
        container_1 = st.container(height=500)    
        for message in st.session_state.messages:
            print("Message: ", message)
            with container_1.chat_message(message["role"]):
                # new addition
                if message["role"] == "assistant":
                    st.write(message["content"])
                if message["role"] == "user":
                    st.write(message["content"])    

    with col2:
        st.subheader("Graph + Vector approach")
        container_2 = st.container(height=500)    
        for message in st.session_state.hybrid_messages:
            print("Message: ", message)
            with container_2.chat_message(message["role"]):
                # new addition
                if message["role"] == "assistant":
                    st.write(message["content"])
                if message["role"] == "user":
                    st.write(message["content"])   

    if prompt := "":
        # Add user message to chat history
        # Display user message in chat message container
        with container_1.chat_message("user"):
            st.write(prompt)
        with container_2.chat_message("user"):
            st.write(prompt)

        with container_1.chat_message("assistant"):
            st.write(res_vector)
        with container_2.chat_message("assistant"):
            st.write(res_hybrid)

 
