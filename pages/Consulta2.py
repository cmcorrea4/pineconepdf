import streamlit as st
import os
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
import pinecone

# Configurar las credenciales (asegúrate de tener estas variables de entorno configuradas)
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.environ["PINECONE_API_KEY"] = st.secrets["PINECONE_API_KEY"]
os.environ["PINECONE_ENV"] = st.secrets["PINECONE_ENV"]

# Inicializar Pinecone
pinecone.init(
    api_key=os.environ["PINECONE_API_KEY"],
    environment=os.environ["PINECONE_ENV"]
)

# Configurar el embedding y el índice de Pinecone
embeddings = OpenAIEmbeddings()
index_name = "pruebapdf2"
docsearch = Pinecone.from_existing_index(index_name, embeddings)

# Configurar el modelo de lenguaje y la cadena de QA
llm = OpenAI(temperature=0)
chain = load_qa_chain(llm, chain_type="stuff")

# Interfaz de Streamlit
st.title("Consulta de documentos con Langchain y Pinecone")

# Input para la consulta
query = st.text_input("Ingresa tu pregunta:")

if query:
    # Realizar la búsqueda
    docs = docsearch.similarity_search(query)
    
    # Obtener la respuesta
    response = chain.run(input_documents=docs, question=query)
    
    st.subheader("Respuesta:")
    st.write(response)
    
    st.subheader("Documentos relevantes:")
    for i, doc in enumerate(docs):
        st.write(f"Documento {i+1}:")
        st.write(doc.page_content)
        st.write("---")
