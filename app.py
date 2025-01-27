import streamlit as st
import pinecone
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import Pinecone
import os

# Configuración de la página
st.set_page_config(page_title="RAG con Pinecone", layout="wide")
st.title("Sistema de Preguntas y Respuestas con RAG")

# Inicialización de variables de entorno (deberías tenerlas en un .env)
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
PINECONE_ENV = st.secrets["PINECONE_ENV"]
INDEX_NAME = "tu-indice"  # Reemplaza con el nombre de tu índice

# Inicialización de Pinecone
@st.cache_resource
def init_pinecone():
    pinecone.init(
        api_key=PINECONE_API_KEY,
        environment=PINECONE_ENV
    )
    return pinecone.Index(INDEX_NAME)

# Inicialización del modelo de embeddings
@st.cache_resource
def init_embeddings():
    return OpenAIEmbeddings()

# Inicialización del chat model
@st.cache_resource
def init_llm():
    return ChatOpenAI(
        temperature=0,
        model_name="gpt-3.5-turbo"
    )

# Configuración del sistema RAG
@st.cache_resource
def setup_rag():
    index = init_pinecone()
    embeddings = init_embeddings()
    
    # Crear el almacén de vectores con Pinecone
    vectorstore = Pinecone(
        index,
        embeddings.embed_query,
        "text"  # nombre del campo que contiene el texto en tus documentos
    )
    
    # Crear el chain de RAG
    llm = init_llm()
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(
            search_kwargs={"k": 3}
        ),
        return_source_documents=True
    )
    
    return qa_chain

# Inicializar el sistema
try:
    qa_chain = setup_rag()
    st.success("Sistema RAG inicializado correctamente!")
except Exception as e:
    st.error(f"Error al inicializar el sistema RAG: {str(e)}")
    st.stop()

# Interface de usuario
st.markdown("### Haz tu pregunta")
question = st.text_input("Pregunta:", key="question_input")

if st.button("Obtener Respuesta"):
    if question:
        try:
            with st.spinner("Procesando tu pregunta..."):
                # Obtener respuesta
                result = qa_chain({"query": question})
                
                # Mostrar respuesta
                st.markdown("### Respuesta:")
                st.write(result["result"])
                
                # Mostrar fuentes
                st.markdown("### Fuentes utilizadas:")
                for i, doc in enumerate(result["source_documents"]):
                    with st.expander(f"Fuente {i+1}"):
                        st.write(doc.page_content)
                        st.write("Metadata:", doc.metadata)
                        
        except Exception as e:
            st.error(f"Error al procesar la pregunta: {str(e)}")
    else:
        st.warning("Por favor, ingresa una pregunta.")

# Sidebar con información adicional
with st.sidebar:
    st.markdown("### Sobre esta aplicación")
    st.write("""
    Esta aplicación utiliza RAG (Retrieval Augmented Generation) para responder
    preguntas basándose en una base de conocimiento almacenada en Pinecone.
    
    El sistema:
    1. Busca información relevante en la base de conocimiento
    2. Recupera los documentos más similares
    3. Genera una respuesta utilizando GPT y la información recuperada
    """)
    
    st.markdown("### Configuración")
    st.write("""
    Para usar esta aplicación, necesitas configurar:
    - API Key de OpenAI
    - API Key de Pinecone
    - Ambiente de Pinecone
    - Índice de Pinecone con tus documentos
    """)
