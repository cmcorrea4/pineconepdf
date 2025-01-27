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

# Variable para almacenar el estado del sistema
if 'system_initialized' not in st.session_state:
    st.session_state.system_initialized = False
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None

# Botón para inicializar el sistema
if pinecone_api_key and openai_api_key and selected_index:
    initialize_button = st.sidebar.button("🚀 Inicializar Sistema")
else:
    st.sidebar.warning("⚠️ Complete las credenciales y seleccione un índice")
    initialize_button = False

# Función para inicializar el sistema RAG
def initialize_rag_system():
    try:
        if not selected_index:
            raise Exception("Por favor, selecciona un índice válido.")
        
        # Inicializar embeddings y modelo
        st.write("Inicializando modelos...")
        embedding_model = OpenAIEmbeddings(
            openai_api_key=openai_api_key,
            model="text-embedding-3-small"
        )
        
        # Verificar dimensiones del embedding
        st.write("Verificando dimensiones del embedding...")
        test_embed = embedding_model.embed_query("test")
        st.write(f"Dimensiones del embedding: {len(test_embed)}")
        
        # Obtener información del índice
        pc = Pinecone(api_key=pinecone_api_key)
        index = pc.Index(selected_index)
        index_info = index.describe_index_stats()
        st.write("Información del índice:", index_info)
        
        # Verificar que las dimensiones coincidan
        if 'dimension' in index_info:
            st.write(f"Dimensiones del índice: {index_info['dimension']}")
            if len(test_embed) != index_info['dimension']:
                raise Exception(f"Las dimensiones no coinciden: Embedding ({len(test_embed)}) ≠ Índice ({index_info['dimension']})")
        
        llm = ChatOpenAI(
            temperature=0,
            model_name="gpt-3.5-turbo",
            openai_api_key=openai_api_key
        )
        
        # Inicializar Pinecone y verificar el índice
        st.write("Conectando con Pinecone...")
        pc = Pinecone(api_key=pinecone_api_key)
        index = pc.Index(selected_index)
        
        # Obtener estadísticas del índice
        stats = index.describe_index_stats()
        st.write("📊 Estadísticas del índice:", stats)
        if stats['total_vector_count'] == 0:
            raise Exception("El índice está vacío. No hay documentos para buscar.")
        
        # Examinar contenido del índice
        st.write("Examinando contenido del índice...")
        
        # Obtener estadísticas detalladas
        stats = index.describe_index_stats()
        st.write("Estadísticas detalladas del índice:")
        st.json(stats)
        
        # Intentar obtener todos los vectores del namespace
        st.write("Intentando obtener todos los vectores del namespace...")
        
        # Crear un vector de prueba para fetch_all
        dummy_vector = [0.0] * 1536  # Vector de ceros del tamaño correcto
        
        # Obtener todos los vectores
        all_vectors = index.query(
            namespace="Interfaces Multimodales y sus apps",
            vector=dummy_vector,
            top_k=100,  # Aumentar para obtener más resultados
            include_metadata=True,
            include_values=True
        )
        
        if all_vectors.matches:
            st.write(f"Se encontraron {len(all_vectors.matches)} vectores")
            
            # Mostrar detalles de cada vector
            for i, match in enumerate(all_vectors.matches):
                with st.expander(f"Vector {i+1}"):
                    st.write("ID:", match.id)
                    if hasattr(match, 'metadata'):
                        st.write("Metadata:", match.metadata)
                    if hasattr(match, 'score'):
                        st.write("Score:", match.score)
                    # Mostrar primeros elementos del vector si están disponibles
                    if hasattr(match, 'values') and match.values:
                        st.write("Primeros 5 elementos del vector:", match.values[:5])
        else:
            st.warning("No se encontraron vectores en el namespace")
        
        # Probar una búsqueda con una query del dominio
        test_query = "interfaces multimodales y aplicaciones"
        query_embedding = embedding_model.embed_query(test_query)
        
        st.write("Probando búsqueda con query específica:", test_query)
        search_response = index.query(
            namespace="Interfaces Multimodales y sus apps",
            vector=query_embedding,
            top_k=5,
            include_metadata=True,
            include_values=True
        )
        
        if search_response.matches:
            st.write("Resultados encontrados:")
            for i, match in enumerate(search_response.matches):
                with st.expander(f"Resultado {i+1}"):
                    st.write("Score:", match.score)
                    st.write("ID:", match.id)
                    if hasattr(match, 'metadata'):
                        st.write("Metadata:", match.metadata)
        
        # Crear vector store con la configuración verificada
        vectorstore = PineconeVectorStore(
            index=index,
            embedding=embedding_model,
            text_key="text",
            namespace="Interfaces Multimodales y sus apps"
        )
        
        # Probar una búsqueda simple para verificar la recuperación
        st.write("Probando recuperación de documentos...")
        test_query = "test query"
        st.write(f"Realizando búsqueda de prueba con: '{test_query}'")
        test_embedding = embedding_model.embed_query(test_query)
        st.write("✅ Embedding de prueba generado")
        
        st.write("Probando búsqueda directa en vectorstore...")
        test_docs = vectorstore.similarity_search(
            test_query,
            k=3
        )
        st.write(f"📄 Documentos recuperados en prueba: {len(test_docs)}")
        if len(test_docs) > 0:
            st.write("Ejemplo de documento recuperado:", test_docs[0].page_content[:200])
        
        # Crear retriever y chain
        st.write("Configurando retriever y chain de QA...")
        retriever = vectorstore.as_retriever(
            search_kwargs={"k": 3}
        )
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            verbose=True
        )
        
        return qa_chain
    
    except Exception as e:
        raise Exception(f"Error al inicializar el sistema: {str(e)}")

# Manejar la inicialización del sistema
if initialize_button:
    try:
        with st.spinner("🔄 Inicializando el sistema..."):
            st.session_state.qa_chain = initialize_rag_system()
            st.session_state.system_initialized = True
        st.success("✅ Sistema RAG inicializado correctamente!")
    except Exception as e:
        st.error(f"❌ {str(e)}")
        st.session_state.system_initialized = False

# Interface principal de usuario
if st.session_state.system_initialized and st.session_state.qa_chain:
    st.markdown("### 💬 Haz tu pregunta")
    question = st.text_input("Pregunta:", key="question_input")

    if st.button("🔍 Obtener Respuesta"):
        if question:
            try:
                with st.spinner("🔄 Procesando tu pregunta..."):
                    st.write("🔍 Buscando documentos relevantes...")
                    result = st.session_state.qa_chain.invoke({"query": question})
                    
                    # Verificar el resultado
                    st.write("📊 Estructura del resultado:", result.keys())
                    
                    # Mostrar respuesta
                    st.markdown("### 📝 Respuesta:")
                    st.write(result["result"])
                    
                    # Mostrar fuentes
                    st.markdown("### 📚 Fuentes utilizadas:")
                    if "source_documents" in result and result["source_documents"]:
                        for i, doc in enumerate(result["source_documents"]):
                            with st.expander(f"📄 Fuente {i+1}"):
                                st.write("Contenido:", doc.page_content)
                                st.write("Metadata:", doc.metadata)
                    else:
                        st.warning("⚠️ No se encontraron documentos fuente para esta respuesta.")
                        st.write("Esto puede indicar que:")
                        st.write("1. No hay documentos similares en la base de conocimiento")
                        st.write("2. El proceso de recuperación no está funcionando correctamente")
                        st.write("3. Los embeddings no se están generando correctamente")
            except Exception as e:
                st.error(f"❌ Error al procesar la pregunta: {str(e)}")
        else:
            st.warning("⚠️ Por favor, ingresa una pregunta.")
else:
    st.info("👈 Por favor, configura las credenciales en el panel lateral e inicializa el sistema para comenzar.")

# Información adicional en el sidebar
with st.sidebar:
    st.markdown("---")
    st.markdown("### ℹ️ Sobre esta aplicación")
    st.write("""
    Esta aplicación utiliza RAG (Retrieval Augmented Generation) para responder
    preguntas basándose en una base de conocimiento almacenada en Pinecone.
    
    El sistema:
    1. Busca información relevante en la base de conocimiento
    2. Recupera los documentos más similares
    3. Genera una respuesta utilizando GPT y la información recuperada
    """)
