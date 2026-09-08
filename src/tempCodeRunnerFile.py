import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import YoutubeLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import InMemoryVectorStore
model = ChatOllama(model='llama3.1')

loader = YoutubeLoader.from_youtube_url("https://www.youtube.com/watch?v=DHnyDE2qPxY&t=2997s", add_video_info = False)

docs  = loader.load()

spliterr = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = spliterr.split_documents(docs)
print(len(docs))

