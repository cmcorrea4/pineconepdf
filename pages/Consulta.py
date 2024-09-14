
user_question = st.text_input("Pregunta: ")
        if user_question:
            os.environ["OPENAI_API_KEY"] = ke2
            embeddings = OpenAIEmbeddings(model="text-embedding-ada-002 ")
            vector_store = PineconeVectorStore(index="langchain-test-index", embedding=embeddings)
           
    
            docs = vector_store.similarity_search(user_question, 3)
            llm = ChatOpenAI(model_name='gpt-4o-mini')
            chain = load_qa_chain(llm, chain_type="stuff")
            respuesta = chain.run(input_documents=docs, question=user_question)
    
            st.write(respuesta)
