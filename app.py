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

# Función para obtener índices directamente de Pinecone
def get_pinecone_indexes(api_key):
    try:
        pc = Pinecone(api_key=api_key)
        current_indexes = pc.list_indexes().names()
        st.write("Índices actuales en Pinecone:", current_indexes)
        return list(current_indexes)
    except Exception as e:
        st.error(f"Error al obtener índices: {str(e)}")
        return []

# Función para limpiar todos los estados
def clear_all_states():
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.experimental_rerun()

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
    
    # Botones de control de caché y actualización
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Actualizar"):
            st.experimental_rerun()
    with col2:
        if st.button("🗑️ Limpiar Caché"):
            clear_all_states()
    
    # Verificar conexión con Pinecone y obtener índices
    if pinecone_api_key:
        try:
            st.markdown("### Estado de Conexión")
            pc = Pinecone(api_key=pinecone_api_key)
            available_indexes = get_pinecone_indexes(pinecone_api_key)
            if available_indexes:
                st.success("✅ Conectado a Pinecone")
                
                # Selector de índice
                selected_index = st.selectbox(
                    "Selecciona un índice",
                    options=available_indexes,
                    help="Selecciona el índice de Pinecone que quieres usar"
                )
                
                # Mostrar información del índice seleccionado
                if selected_index:
                    try:
                        index = pc.Index(selected_index)
                        stats = index.describe_index_stats()
                        st.info(f"Índice seleccionado: {selected_index}")
                        st.write("Estadísticas del índice:", stats)
                    except Exception as e:
                        st.error(f"Error al obtener información del índice: {str(e)}")
            else:
                st.warning("⚠️ No hay índices disponibles")
                
        except Exception as e:
            st.error(f"❌ Error de conexión: {str(e)}")
            selected_index = None
    else:
        selected_index = None
    
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
        if not selected_index:
            raise Exception("Por favor, selecciona un índice válido.")
        
        # Inicializar embeddings y modelo
        embedding_model = OpenAIEmbeddings(openai_api_key=openai_api_key)
        llm = ChatOpenAI(
            temperature=0,
            model_name="gpt-3.5-turbo",
            openai_api_key=openai_api_key
        )
        
        # Inicializar Pinecone y crear vector store
        pc = Pinecone(api_key=pinecone_api_key)
        index = pc.Index(selected_index)
        
        vectorstore = PineconeVectorStore(
            index=index,
            embedding=embedding_model,
            text_key="text"
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
    if not openai_api_key or not pinecone_api_key or not selected_index:
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
