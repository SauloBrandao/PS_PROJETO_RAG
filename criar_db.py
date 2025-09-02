from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

BASE="base"

def criar_db():
    documentos = carregar_documentos()
    chunks = divisor_chunks(documentos)
    vetorizar_chunks(chunks)

def carregar_documentos():
    carregador = PyPDFDirectoryLoader(BASE, glob="*pdf")
    documentos = carregador.load()
    return documentos

def divisor_chunks(documentos):
    separador_documentos = RecursiveCharacterTextSplitter (
        chunk_size = 2000,
        chunk_overlap = 500,
        length_function = len,
        add_start_index=True
    )
    chunks = separador_documentos.split_documents(documentos)
    return chunks

def vetorizar_chunks(chunks):
    db = Chroma.from_documents(chunks, OpenAIEmbeddings(), persist_directory="db")
    print("Banco de Dados Criado")

criar_db()