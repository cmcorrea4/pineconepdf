import streamlit as st
import pinecone
import os
from langchain_pinecone import PineconeVectorStore
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from pinecone import ServerlessSpec
from pinecone import Pinecone, ServerlessSpec
from langchain.embeddings.openai import OpenAIEmbeddings

PINECONE_API_KEY = st.secrets['API_KEY_DE_PINECONE']
OPENAI_API_KEY = st.secrets['API_KEY_DE_OPENAI']


user_question = st.text_area("Pregunta: ")
if user_question:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002 ")
    vector_store = PineconeVectorStore(index="pdfprueba2", embedding=embeddings)
   

    docs = vector_store.similarity_search(user_question, 3)
    llm = ChatOpenAI(model_name='gpt-4o-mini')
    chain = load_qa_chain(llm, chain_type="stuff")
    respuesta = chain.run(input_documents=docs, question=user_question)

    st.write(respuesta)
