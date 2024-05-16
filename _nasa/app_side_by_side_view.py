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
import read_hybrid
import read_vector
import read_hybrid_topic
from streamlit_float import *

# Define the location of the FastAPI server
BACKEND = "http://localhost:8000"

# Set the OpenAI API key
os.environ["AZURE_OPENAI_API_KEY"] = ""
os.environ["AZURE_OPENAI_ENDPOINT"] = ""
os.environ["OPENAI_MODEL_NAME"] = ""
os.environ["OPENAI_API_VERSION"] = ""
os.environ["OPENAI_API_TYPE"] = ""

st.set_page_config(
    page_title="Vector only vs KG + vector",
    page_icon="🚀",
    layout="wide",
)

float_init(theme=True, include_unstable_primary=False)

# Set the title of the app
st.title('Nasa 🚀')

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

tab1, tab2 = st.tabs(["LLM", "Topics"])
# Main content
# Logic for each option

with tab1:
    st.header("Let's start exploring your data")

    # Create columns for layout
    col1, col2 = st.columns(2)

    # Display chat messages from history on app rerun
    with col1:
        st.subheader("Vector Only approach")
        container_1 = st.container(height=600)    
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
        container_2 = st.container(height=600)    
        for message in st.session_state.hybrid_messages:
            print("Message: ", message)
            with container_2.chat_message(message["role"]):
                # new addition
                if message["role"] == "assistant":
                    st.write(message["content"])
                if message["role"] == "user":
                    st.write(message["content"])   

    if prompt := st.chat_input("Start searching, without Topics"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt, "mode": "prompt"})
        st.session_state.hybrid_messages.append({"role": "user", "content": prompt, "mode": "prompt"})
        # Display user message in chat message container
        with container_1.chat_message("user"):
            st.write(prompt)
        with container_2.chat_message("user"):
            st.write(prompt)

        with container_1.chat_message("assistant"):
            res_vector = read_vector.query(prompt)
            print("RES_VECTOR: ", res_vector)
            # container_1.write(res)
            st.write(res_vector)
        with container_2.chat_message("assistant"):
            res_hybrid = read_hybrid.query(prompt)
            print("HYBRID_TOPIC_VECTOR: ", res_hybrid)
            # container_1.write(res)
            st.write(res_hybrid)

        # for hybrid search
        # res_hybrid = read_hybrid.query(prompt)
        #container_2.chat_message("assistant").write(res_hybrid)
                
        st.session_state.messages.append({"role": "assistant", "content": res_vector})
        st.session_state.hybrid_messages.append({"role": "assistant", "content": res_hybrid})

with tab2:
    st.header("Let's start exploring your data")

    # Create columns for layout
    col1, col2 = st.columns(2)

    # Display chat messages from history on app rerun
    with col1:
        st.subheader("Vector Only approach")
        container_1 = st.container(height=600)    
        for message in st.session_state.topic_messages:
            print("Message: ", message)
            with container_1.chat_message(message["role"]):
                # new addition
                if message["role"] == "assistant":
                    st.write(message["content"])
                if message["role"] == "user":
                    st.write(message["content"])    

    with col2:
        st.subheader("Graph + Vector approach")
        container_2 = st.container(height=600)    
        for message in st.session_state.topic_hybrid_messages:
            print("Message: ", message)
            with container_2.chat_message(message["role"]):
                # new addition
                if message["role"] == "assistant":
                    st.write(message["content"])
                if message["role"] == "user":
                    st.write(message["content"])   

    if prompt := st.chat_input("Start searching, with Topics?"):
        # Add user message to chat history
        st.session_state.topic_messages.append({"role": "user", "content": prompt, "mode": "prompt"})
        st.session_state.topic_hybrid_messages.append({"role": "user", "content": prompt, "mode": "prompt"})
        # Display user message in chat message container
        with container_1.chat_message("user"):
            st.write(prompt)
        with container_2.chat_message("user"):
            st.write(prompt)

        with container_1.chat_message("assistant"):
            res_vector = read_vector.query(prompt)
            print("RES_VECTOR: ", res_vector)
            # container_1.write(res)
            st.write(res_vector)
        with container_2.chat_message("assistant"):
            res_hybrid = read_hybrid.query(prompt)
            print("HYBRID_VECTOR: ", res_hybrid)
            # container_1.write(res)
            st.write(res_hybrid)

        # for hybrid search
        # res_hybrid = read_hybrid.query(prompt)
        #container_2.chat_message("assistant").write(res_hybrid)
                
        st.session_state.topic_messages.append({"role": "assistant", "content": res_vector})
        st.session_state.topic_hybrid_messages.append({"role": "assistant", "content": res_hybrid})

# with col2:
#     st.subheader("Vector Only approach")
#     container_2 = st.container(height=600)    
#     for message in st.session_state.hybrid_messages:
#         print("Message: ", message)
#         # with container_1.chat_message(message["role"]):
#             # new addition
#         if message["role"] == "assistant":
#             container_2.chat_message("assistant").write(message["content"])
#         if message["role"] == "user":
#             container_2.chat_message("user").write(message["content"])
#     if prompt := st.chat_input("What is up?"):
#         # Add user message to chat history
#         st.session_state.hybrid_messages.append({"role": "user", "content": prompt, "mode": "hybrid"})
#         # Display user message in chat message container
#         # with container_1.chat_message("user"):
#         container_2.chat_message("user").write(prompt)
#             # container_1.markdown(prompt)
#         # with container_1.chat_message("assistant"):
#         res = read_vector.query(prompt)
#         print("RES: ", res)
#         # container_1.write(res)
#         container_2.chat_message("assistant").write(res)
                
#         st.session_state.hybrid_messages.append({"role": "assistant", "content": res, "mode": "hybrid"})


