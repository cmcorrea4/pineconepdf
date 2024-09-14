import streamlit as st
import pinecone
import openai
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Configura tu API key de Pinecone y OpenAI
PINECONE_API_KEY = st.secrets['API_KEY_DE_PINECONE']
OPENAI_API_KEY = st.secrets['API_KEY_DE_OPENAI']

# Inicializa Pinecone (se ejecutará solo cuando se presione el botón)
def init_pinecone():
    pinecone.init(api_key=PINECONE_API_KEY, environment="us-east-1")

    index_name = 'pdfprueba'
    dimension = 1536  # Ajusta la dimensión según tu modelo de embeddings
    metric = 'cosine'

    # Verifica si el índice ya existe, si no, lo crea
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(name=index_name, dimension=dimension, metric=metric)

    return pinecone.Index(index_name)

# Configura OpenAI
openai.api_key = OPENAI_API_KEY

# Función para extraer texto del PDF
def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)  # Actualizado para PyPDF2
    text = ''
    for page_num in range(len(reader.pages)):
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
def save_to_pinecone(chunks, index):
    for i, chunk in enumerate(chunks):
        embedding = generate_embeddings(chunk)
        index.upsert([(f'id_{i}', embedding)])

# Interfaz de usuario de Streamlit
st.title("PDF a Base de Datos Vectorial")
pdf_file = st.file_uploader("Sube tu archivo PDF", type="pdf")

# Mostrar el botón después de que se sube un archivo PDF
if pdf_file is not None:
    # Botón para cargar el contenido a Pinecone
    if st.button("Cargar a la base de datos"):
        with st.spinner('Procesando el archivo...'):
            # Inicializar Pinecone y extraer texto solo cuando se presione el botón
            index = init_pinecone()
            text = extract_text_from_pdf(pdf_file)
            chunks = split_text(text)
            save_to_pinecone(chunks, index)
            st.success("El archivo PDF ha sido procesado y almacenado en la base de datos vectorial.")

