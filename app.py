import streamlit as st
from pinecone import Pinecone, ServerlessSpec
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_pinecone import PineconeVectorStore
import os
import time

# Configuración de la página
st.set_page_config(page_title="RAG con Pinecone", layout="wide")
st.title("Sistema de Preguntas y Respuestas con RAG")

# Función para obtener índices
def get_pinecone_indexes(api_key):
    try:
        pc = Pinecone(api_key=api_key)
        indexes = pc.list_indexes().names()
        return list(indexes)
    except Exception as e:
        st.error(f"Error al obtener índices: {str(e)}")
        return []

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
    
    # Inicializar variables de estado si no existen
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = time.time()
    if 'available_indexes' not in st.session_state:
        st.session_state.available_indexes = []
    
    # Sección de gestión de índices
    st.markdown("### Gestión de Índices")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        refresh_button = st.button("🔄 Actualizar Índices")
    
    # Actualizar lista de índices cuando se presione el botón
    if refresh_button and pinecone_api_key:
        with st.spinner("Actualizando lista de índices..."):
            st.session_state.available_indexes = get_pinecone_indexes(pinecone_api_key)
            st.session_state.last_refresh = time.time()
    
    # Mostrar tiempo desde última actualización
    st.caption(f"Última actualización: {int(time.time() - st.session_state.last_refresh)} segundos atrás")
    
    # Campo para nuevo índice
    new_index_name = st.text_input(
        "Crear nuevo índice",
        help="Introduce el nombre para crear un nuevo índice"
    )
    
    if st.button("Crear Índice") and new_index_name and pinecone_api_key:
        try:
            pc = Pinecone(api_key=pinecone_api_key)
            spec = ServerlessSpec(cloud="aws", region="us-west-2")
            
            with st.spinner("Creando nuevo índice..."):
                pc.create_index(
                    name=new_index_name,
                    dimension=1536,
                    metric="cosine",
                    spec=spec
                )
                st.success(f"Índice '{new_index_name}' creado correctamente!")
                # Actualizar lista de índices
                st.session_state.available_indexes = get_pinecone_indexes(pinecone_api_key)
        except Exception as e:
            st.error(f"Error al crear el índice: {str(e)}")
    
    # Selector de índice
    if pinecone_api_key:
        if not st.session_state.available_indexes:
            st.session_state.available_indexes = get_pinecone_indexes(pinecone_api_key)
        
        if st.session_state.available_indexes:
            index_name = st.selectbox(
                "Selecciona un índice",
                options=st.session_state.available_indexes,
                help="Selecciona el índice de Pinecone que quieres usar"
            )
        else:
            st.warning("No hay índices disponibles.")
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
