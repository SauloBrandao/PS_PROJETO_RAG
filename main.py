from langchain_chroma.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv


load_dotenv()

DB_PATH = "db"

prompt_template = """
Responda a pergunta do usuário
{pergunta}

com base na base de informações:

{informaçoes}
"""

def perguntar():
    pergunta = input("Escreva sua pergunta sobre T20: ")

    funcao_embedding = OpenAIEmbeddings
    db = Chroma(persist_directory=DB_PATH, embedding_function=funcao_embedding)

    resultados = db.similarity_search_by_image_with_relevance_score(pergunta, k=3)
    if len(resultados) == 0 or resultados[0] [1] < 0.7:
        print("Não consegui encontrar informações relevantes")
        return
    
    textos_resultados = [resultado[0].page_content for resultado in resultados]
    informaçoes = "\n\n----\n\n".join(textos_resultados)
    prompt = ChatPromptTemplate.from_template(prompt_template)
    prompt = prompt.invoke({"pergunta": pergunta, "informaçoes": informaçoes})

    modelo = ChatOpenAI()
    resposta = modelo.invoke(prompt).content
    print("Resposta da Ai: ", resposta)
    
perguntar()
