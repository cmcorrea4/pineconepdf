import streamlit as st
import pinecone
import os
from langchain_pinecone import PineconeVectorStore
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from pinecone import ServerlessSpec
from pinecone import Pinecone, ServerlessSpec
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from pinecone import Pinecone, ServerlessSpec
from langchain.chains import RetrievalQA

PINECONE_API_KEY = st.secrets['API_KEY_DE_PINECONE']
OPENAI_API_KEY = st.secrets['API_KEY_DE_OPENAI']

pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = 'pdfprueba2'
dimension = 1536  # Ajusta la dimensión según tu modelo de embeddings
metric = 'cosine'
spec = {"serverless": {"cloud": "aws","region": "us-east-1"}}


#pc.init(api_key=PINECONE_API_KEY, environment="us-east-1")

# Conectar al índice pdfprueba2

index = pc.Index(index_name)

# Configura OpenAI para Langchain
os.environ['OPENAI_API_KEY'] =  OPENAI_API_KEY
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Cargar el índice Pinecone como almacén de vectores en Langchain
vector_store = PineconeVectorStore(index=index, embedding=embeddings, text_key='text')

# Cargar la cadena de preguntas y respuestas
#llm = OpenAI(openai_api_key=OPENAI_API_KEY)


# Interfaz de usuario de Streamlit
st.title("Consulta en Base de Datos Vectorial con Langchain")

# Entrada de texto para la consulta
query_text = st.text_area("Escribe tu consulta:")

# Realizar la búsqueda cuando se presiona el botón
if st.button("Buscar en la base de datos"):
    if query_text:
        with st.spinner('Buscando...'):
            # Consultar el índice Pinecone con la pregunta
            query_embedding = embeddings.embed_query(query_text)
            docs = vector_store.similarity_search(query_embedding,k=2)
            llm = ChatOpenAI(model_name='gpt-4o-mini',temperature=0.0)
            #qa_chain = load_qa_chain(llm, chain_type="stuff")
            qa = RetrievalQA.from_chain_type(llm=llm,chain_type="stuff",retriever=vectorstore.as_retriever())  
            response=qa.invoke(query_text)
            st.write(response)
            

            # Utilizar la cadena de Langchain para obtener una respuesta basada en los documentos
            if docs:
                response = qa_chain.run(input_documents=docs, question=query_text)
                st.subheader("Respuesta generada:")
                st.write(response)
            else:
                st.write("No se encontraron resultados.")
    else:
        st.error("Por favor, ingresa una consulta antes de buscar.")
