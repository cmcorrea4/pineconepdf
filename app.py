import streamlit as st
import pinecone
import openai
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

# Configura tu API key de Pinecone y OpenAI
PINECONE_API_KEY = st.secrets['API_KEY_DE_PINECONE']
OPENAI_API_KEY = st.secrets['API_KEY_DE_OPENAI']

# Inicializa Pinecone
from pinecone import Pinecone, ServerlessSpec


# Configura OpenAI
openai.api_key = OPENAI_API_KEY

# Función para extraer texto del PDF
def extract_text_from_pdf(pdf_file):
    
    reader = PyPDF2.PdfReader(pdf_file)
    text = ''
    for page_num in range( len(reader.pages)):
        page = reader.pages[page_num]
        text += page.extract_text()
    return text

# Función para dividir el texto en fragmentos
def split_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)
    return chunks

# Función para generar embeddings
def generate_embeddings(text):
    response = openai.Embedding.create(input=text, model='text-embedding-ada-002')
    return response['data'][0]['embedding']

# Función para guardar en Pinecone
def save_to_pinecone(chunks):
    for i, chunk in enumerate(chunks):
        embedding = generate_embeddings(chunk)
        index.upsert([(f'id_{i}', embedding)])

# Interfaz de usuario de Streamlit
st.title("PDF a Base de Datos Vectorial")
pdf_file = st.file_uploader("Sube tu archivo PDF", type="pdf")

if pdf_file is not None:
    text = extract_text_from_pdf(pdf_file)
    chunks = split_text(text)

if st.button("Cargar a la base de datos"):
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index_name = 'pdfprueba2'
    dimension = 1536  # Ajusta la dimensión según tu modelo de embeddings
    metric = 'cosine'
    spec = {
        "serverless": {
           "cloud": "aws",
           "region": "us-east-1"
         }
    }
    if index_name not in pc.list_indexes():
       pc.create_index(name=index_name, dimension=dimension, metric=metric, spec=spec)
       index = pc.Index(index_name)
    save_to_pinecone(chunks)
    st.success("El archivo PDF ha sido procesado y almacenado en la base de datos vectorial.")

