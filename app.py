import streamlit as st
from pinecone import Pinecone, ServerlessSpec
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_pinecone import PineconeVectorStore
import os
import time

# ... [código anterior hasta la función initialize_rag_system] ...

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
        
        # Inicializar Pinecone y verificar el índice
        pc = Pinecone(api_key=pinecone_api_key)
        index = pc.Index(selected_index)
        
        # Obtener estadísticas del índice
        stats = index.describe_index_stats()
        st.write("Estadísticas del índice:", stats)
        if stats['total_vector_count'] == 0:
            raise Exception("El índice está vacío. No hay documentos para buscar.")
        
        # Crear vector store
        vectorstore = PineconeVectorStore(
            index=index,
            embedding=embedding_model,
            text_key="text"
        )
        
        # Probar una búsqueda simple para verificar la recuperación
        st.write("Probando recuperación de documentos...")
        retriever = vectorstore.as_retriever(
            search_kwargs={
                "k": 3,
                "fetch_k": 10  # Buscar en más documentos
            }
        )
        
        # Hacer una búsqueda de prueba
        test_docs = retriever.get_relevant_documents("test query")
        st.write(f"Documentos recuperados en prueba: {len(test_docs)}")
        if len(test_docs) > 0:
            st.write("Ejemplo de documento recuperado:", test_docs[0].page_content[:200])
        
        # Crear chain con configuración para depuración
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            verbose=True  # Activar modo verbose
        )
        
        return qa_chain
    
    except Exception as e:
        raise Exception(f"Error al inicializar el sistema: {str(e)}")

# Interface principal de usuario
if st.session_state.system_initialized and st.session_state.qa_chain:
    st.markdown("### Haz tu pregunta")
    question = st.text_input("Pregunta:", key="question_input")

    if st.button("Obtener Respuesta"):
        if question:
            try:
                with st.spinner("Procesando tu pregunta..."):
                    # Crear embedding de la pregunta para depuración
                    embedding_model = OpenAIEmbeddings(openai_api_key=openai_api_key)
                    question_embedding = embedding_model.embed_query(question)
                    st.write("Embedding de la pregunta generado correctamente")
                    
                    # Obtener respuesta
                    result = st.session_state.qa_chain.invoke({"query": question})
                    
                    # Verificar el resultado
                    st.write("Estructura del resultado:", result.keys())
                    
                    # Mostrar respuesta
                    st.markdown("### Respuesta:")
                    st.write(result["result"])
                    
                    # Mostrar fuentes
                    st.markdown("### Fuentes utilizadas:")
                    if "source_documents" in result and result["source_documents"]:
                        for i, doc in enumerate(result["source_documents"]):
                            with st.expander(f"Fuente {i+1}"):
                                st.write("Contenido:", doc.page_content)
                                st.write("Metadata:", doc.metadata)
                    else:
                        st.warning("⚠️ No se encontraron documentos fuente para esta respuesta.")
                        st.write("Esto puede indicar que:")
                        st.write("1. No hay documentos similares en la base de conocimiento")
                        st.write("2. El proceso de recuperación no está funcionando correctamente")
                        st.write("3. Los embeddings no se están generando correctamente")
            except Exception as e:
                st.error(f"Error al procesar la pregunta: {str(e)}")
        else:
            st.warning("Por favor, ingresa una pregunta.")
else:
    st.info("👈 Por favor, configura las credenciales en el panel lateral e inicializa el sistema para comenzar.")
