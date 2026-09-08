from time import sleep
import streamlit as st
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import InMemoryVectorStore

model = ChatOllama(model='llama3.1')

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "document_uploaded" not in st.session_state:
    st.session_state.document_uploaded = False

def process_document(file_path):
    """Loads a PDF, splits it into chunks, and stores it in an in-memory vector database."""
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.split_documents(docs)

    embeddings = OllamaEmbeddings(model='nomic-embed-text')
    vector_db = InMemoryVectorStore.from_documents(documents=docs, embedding=embeddings)

    st.session_state.vector_db = vector_db
    st.session_state.document_uploaded = True

st.subheader('Document Q&A ChatBot')

if not st.session_state.document_uploaded:
    file = st.file_uploader(label='Select your PDF file', type='pdf')
    
    if file:
        file_name = file.name
        with open(file_name, 'wb') as f:
            f.write(file.getvalue())
            
        with st.spinner('Processing document...'):
            process_document(file_name)
            
        st.success('Document processed successfully!')
        sleep(1)
        st.rerun()

if st.session_state.document_uploaded and st.session_state.vector_db:
    
    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])
        
    query = st.chat_input("Ask anything about the document...")
    
    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        st.chat_message("user").markdown(query)
        
        documents = st.session_state.vector_db.similarity_search(query)
        context = "\n\n".join([doc.page_content for doc in documents])

        prompt = f"""You are a helpful assistant. Provide an answer to the user's question based strictly on the provided context. 
        
        Context: {context} 
        
        Question: {query}"""
        
        with st.chat_message("ai"):
            message_placeholder = st.empty()
            full_response = ""
            
            for chunk in model.stream(prompt):
                full_response += chunk.content
                message_placeholder.markdown(full_response + "| ")
            
            message_placeholder.markdown(full_response)
            
        # Save complete response to history
        st.session_state.messages.append({"role": "ai", "content": full_response})