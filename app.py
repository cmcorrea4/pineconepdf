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
        indexes = list(pc.list_indexes().names())
        # Asegurar que 'plastico' siempre esté en la lista
        if 'plastico' not in indexes:
            indexes.append('plastico')
        return indexes
    except Exception as e:
        st.error(f"Error al obtener índices: {str(e)}")
        return ['plastico']  # Retornar al menos 'plastico' si hay error

# Inicializar variables de estado si no existen
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = time.time()
if 'available_indexes' not in st.session_state:
    st.session_state.available_indexes = []
if 'selected_index' not in st.session_state:
    st.session_state.selected_index = None

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
    
    # Sección de gestión de índices
    st.markdown("### Gestión de Índices")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        refresh_button = st.button("🔄 Actualizar Índices")
    
    # Actualizar lista de índices cuando se presione el botón o cuando se ingrese la API key
    if (refresh_button or not st.session_state.available_indexes) and pinecone_api_key:
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
        st.markdown("### Índices Disponibles")
        # Asegurar que 'plastico' esté en la lista
        all_indexes = list(st.session_state.available_indexes)
        if 'plastico' not in all_indexes:
            all_indexes.append('plastico')
        
        st.session_state.selected_index = st.selectbox(
            "Selecciona un índice",
            options=all_indexes,
            index=all_indexes.index('plastico') if 'plastico' in all_indexes else 0,
            help="Selecciona el índice de Pinecone que quieres usar"
        )
        st.info(f"Índice seleccionado: {st.session_state.selected_index}")
    else:
        if pinecone_api_key:
            st.warning("No hay índices disponibles.")
        st.session_state.selected_index = None
    
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
        if not st.session_state.selected_index:
            raise Exception("Por favor, selecciona un índice válido.")
            
        # Verificar que el índice existe
        pc = Pinecone(api_key=pinecone_api_key)
        if st.session_state.selected_index not in pc.list_indexes().names():
            raise Exception(f"El índice '{st.session_state.selected_index}' no existe.")
        
        # Inicializar embeddings y modelo
        embedding_model = OpenAIEmbeddings(openai_api_key=openai_api_key)
        llm = ChatOpenAI(
            temperature=0,
            model_name="gpt-3.5-turbo",
            openai_api_key=openai_api_key
        )
        
        # Crear vector store con el índice existente
        vectorstore = PineconeVectorStore.from_existing_index(
            index_name=st.session_state.selected_index,
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
    if not openai_api_key or not pinecone_api_key:
        st.error("Por favor, completa todos los campos de configuración.")
    else:
        try:
            with st.spinner("Inicializando el sistema..."):
                st.session_state.qa_chain = initialize_rag_system()
                st.session_state.system_initialized = True
            st.success("Sistema RAG inicializado correctamente!")
        except Exception as e:
            st.error(str(e))
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
