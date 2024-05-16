import streamlit as st
from langchain.chains import RetrievalQA
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.callbacks.base import BaseCallbackHandler
from langchain.vectorstores.neo4j_vector import Neo4jVector
from streamlit.logger import get_logger
from langchain_community.embeddings import OllamaEmbeddings
from chains import (
    load_llm,
)

url = "bolt://localhost:7687"
username = "neo4j"
password = "Password01!"
ollama_base_url = "http://localhost:11434"
embedding_model_name = "ollama"
llm_name = "llama2"
DATABASE="ollamadb"

logger = get_logger(__name__)

def get_model_parameter():
    model_list=["llama2:latest","mistral-openorca"]
    embeddings="llama2:latest"
    set_chunk_size=1000 #default number
    set_chunk_overlap=200

    with st.form("Select a model",clear_on_submit=False):
        set_chunk_size = st.number_input(label="chunk_size",value=1000,step=100)
        set_chunk_overlap = st.number_input(label="chunk_overlap",value=200,step=50)
        col1,col2 =st.columns(2)
        col1.selectbox("Select model:", model_list,key="ollamamodel")
        submitted_username = st.form_submit_button("Select a model to submit")
        if submitted_username:
            embeddings= str(st.session_state["ollamamodel"]) 
            st.info("You selected model: "+ embeddings)
        return [embeddings,set_chunk_size,set_chunk_overlap]
        # else:
        #     return [embeddings,set_chunk_size,set_chunk_overlap]
get_form_result= get_model_parameter()
get_embeddings = get_form_result[0]
get_chunk_size= get_form_result[1]
get_chunk_overlap =get_form_result[2]

embeddings =  OllamaEmbeddings(
    model=get_embeddings
)
dimension = 4096

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)


llm = load_llm(llm_name, logger=logger, config={"ollama_base_url": ollama_base_url})


def main():

    st.header("📄Chat with your pdf file")

    # upload a your pdf file
    pdf = st.file_uploader("Upload your PDF", type="pdf")

    if pdf is not None:
        pdf_reader = PdfReader(pdf)

        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # langchain_textspliter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=get_chunk_size, chunk_overlap=get_chunk_overlap, length_function=len
        )

        chunks = text_splitter.split_text(text=text)

        # Store the chunks part in db (vector)
        vectorstore = Neo4jVector.from_texts(
            chunks,
            url=url,
            username=username,
            password=password,
            embedding=embeddings,
            index_name="pdf_bot",
            node_label="PdfBotChunk",
            database=DATABASE,
            pre_delete_collection=True,  # Delete existing PDF data
        )
        qa = RetrievalQA.from_chain_type(
            llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever()
        )

        # Accept user questions/query
        query = st.text_input("Ask questions about related your upload pdf file")

        if query:
            stream_handler = StreamHandler(st.empty())
            qa.invoke(query, callbacks=[stream_handler])


if __name__ == "__main__":
    main()
