import streamlit as st
import os
from PyPDF2 import PdfReader
import docx
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from streamlit_chat import message
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Google Gemini API key
gemini_api_key = st.secrets["GEMINI_API_KEY"]

# Configure Google Gemini API
genai.configure(api_key=gemini_api_key)

def main():
    st.set_page_config(page_title="Chat with your file")
    st.header("DocumentGPT")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None

    with st.sidebar:
        uploaded_files = st.file_uploader("Upload your file", type=['pdf'], accept_multiple_files=True)
    #    gemini_api_key_input = st.text_input("Google Gemini API Key", value=gemini_api_key, type="password")
        process = st.button("Process")
    
    if process:
        if not gemini_api_key_input:
            st.info("Please add your Google Gemini API key to continue.")
            st.stop()
        
        files_text = get_files_text(uploaded_files)
        st.write("File loaded...")
        
        text_chunks = get_text_chunks(files_text)
        st.write("File chunks created...")
        
        vector_store = get_vectorstore(text_chunks)
        st.write("Vector Store Created...")
        
        st.session_state.conversation = get_conversation_chain(vector_store, gemini_api_key_input)
        st.session_state.processComplete = True

    if st.session_state.processComplete:
        user_question = st.chat_input("Ask a question about your files.")
        if user_question:
            handle_user_input(user_question)

def get_files_text(uploaded_files):
    text = ""
    for uploaded_file in uploaded_files:
        split_tup = os.path.splitext(uploaded_file.name)
        file_extension = split_tup[1]
        if file_extension == ".pdf":
            text += get_pdf_text(uploaded_file)
        elif file_extension == ".docx":
            text += get_docx_text(uploaded_file)
    return text

def get_pdf_text(pdf):
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_docx_text(file):
    doc = docx.Document(file)
    allText = []
    for docpara in doc.paragraphs:
        allText.append(docpara.text)
    text = ' '.join(allText)
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=900,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    knowledge_base = FAISS.from_texts(text_chunks, embeddings)
    return knowledge_base

def get_conversation_chain(vector_store, gemini_api_key):
    llm = genai.GenerativeModel('gemini-1.5-pro-latest')
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_user_input(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    response_container = st.container()
    with response_container:
        for i, messages in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                message(messages.content, is_user=True, key=str(i))
            else:
                message(messages.content, key=str(i))

if __name__ == '__main__':
    main()
