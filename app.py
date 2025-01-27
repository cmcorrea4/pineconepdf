import streamlit as st
from pinecone import Pinecone, ServerlessSpec
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_pinecone import PineconeVectorStore
import os

# Configuración de la página
st.set_page_config(page_title="RAG con Pinecone", layout="wide")
st.title("Sistema de Preguntas y Respuestas con RAG")

# Sidebar para configuración de credenciales
with st.sidebar:
    st.markdown("### Configuración de Credenciales")
    
    # Campo para OpenAI API Key
    openai_api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        help="Introduce tu API key de OpenAI"
    )
    
    # Campos para Pinecone
    pinecone_api_key = st.text_input(
        "Pinecone API Key",
        type="password",
        help="Introduce tu API key de Pinecone"
    )
    
    # Inicializar Pinecone para obtener índices disponibles
    if pinecone_api_key:
        try:
            pc = Pinecone(api_key=pinecone_api_key)
            available_indexes = pc.list_indexes().names()
            
            # Mostrar índices disponibles o crear nuevo
            st.markdown("### Índices Disponibles")
            if available_indexes:
                index_name = st.selectbox(
                    "Selecciona un índice existente",
                    options=available_indexes,
                    help="Selecciona el índice de Pinecone que quieres usar"
                )
            else:
                st.warning("No hay índices disponibles.")
                index_name = st.text_input(
                    "Nombre para el nuevo índice",
                    value="pdf-vector-store",
                    help="Introduce el nombre para crear un nuevo índice"
                )
        except Exception as e:
            st.error(f"Error al conectar con Pinecone: {str(e)}")
            index_name = None
    else:
        index_name = None
    
    # Botón para inicializar el sistema
    initialize_button = st.button("Inicializar Sistema")

# Variable para almacenar el estado del sistema
if 'system_initialized' not in st.session_state:
    st.session_state.system_initialized = False
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None

# Función para inicializar el sistema RAG
def initialize_rag_system():
    try:
        # Inicializar embeddings y modelo
        embedding_model = OpenAIEmbeddings(openai_api_key=openai_api_key)
        llm = ChatOpenAI(
            temperature=0,
            model_name="gpt-3.5-turbo",
            openai_api_key=openai_api_key
        )
        
        # Inicializar Pinecone y obtener el índice
        pc = Pinecone(api_key=pinecone_api_key)
        
        # Si el índice no existe, lo creamos
        if index_name not in pc.list_indexes().names():
            # Crear especificación para el índice
            spec = ServerlessSpec(
                cloud="aws",
                region="us-west-2"
            )
            
            pc.create_index(
                name=index_name,
                dimension=1536,  # Dimensión para OpenAI embeddings
                metric="cosine",
                spec=spec
            )
        
        # Obtener el índice existente
        index = pc.Index(index_name)

        # Crear vector store con el índice existente
        vectorstore = PineconeVectorStore.from_existing_index(
            index_name=index_name,
            embedding=embedding_model
        )
        
        # Crear el retriever
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        
        # Crear chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
        
        return qa_chain
    
    except Exception as e:
        raise Exception(f"Error al inicializar el sistema: {str(e)}")

# Manejar la inicialización del sistema
if initialize_button:
    if not openai_api_key or not pinecone_api_key or not index_name:
        st.error("Por favor, completa todos los campos de configuración.")
    else:
        try:
            with st.spinner("Inicializando el sistema..."):
                st.session_state.qa_chain = initialize_rag_system()
                st.session_state.system_initialized = True
            st.success("Sistema RAG inicializado correctamente!")
        except Exception as e:
            st.error(f"Error al inicializar el sistema: {str(e)}")
            st.session_state.system_initialized = False

# Interface principal de usuario
if st.session_state.system_initialized and st.session_state.qa_chain:
    st.markdown("### Haz tu pregunta")
    question = st.text_input("Pregunta:", key="question_input")

    if st.button("Obtener Respuesta"):
        if question:
            try:
                with st.spinner("Procesando tu pregunta..."):
                    # Obtener respuesta
                    result = st.session_state.qa_chain.invoke({"query": question})
                    
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
else:
    st.info("👈 Por favor, configura las credenciales en el panel lateral e inicializa el sistema para comenzar.")

# Información adicional en el sidebar
with st.sidebar:
    st.markdown("---")
    st.markdown("### Sobre esta aplicación")
    st.write("""
    Esta aplicación utiliza RAG (Retrieval Augmented Generation) para responder
    preguntas basándose en una base de conocimiento almacenada en Pinecone.
    
    El sistema:
    1. Busca información relevante en la base de conocimiento
    2. Recupera los documentos más similares
    3. Genera una respuesta utilizando GPT y la información recuperada
    """)
