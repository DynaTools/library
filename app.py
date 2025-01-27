import streamlit as st
import openai
import PyPDF2
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import tempfile

# Configuração da página
st.set_page_config(
    page_title="Biblioteca de PDFs com IA",
    layout="wide",
)

# Título do aplicativo
st.title("📚 Biblioteca de PDFs com IA")

# Configurar a chave da API do OpenAI
openai_api_key = st.secrets["OPENAI_API_KEY"] if "OPENAI_API_KEY" in st.secrets else os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("Por favor, configure sua chave da API do OpenAI nas variáveis de ambiente.")
    st.stop()

openai.api_key = openai_api_key

# Função para extrair texto de PDFs
def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        extracted_text = page.extract_text()
        if extracted_text:
            text += extracted_text
    return text

# Upload de PDFs
uploaded_files = st.file_uploader("Faça upload dos seus arquivos PDF", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    with st.spinner("Processando PDFs..."):
        all_texts = ""
        for uploaded_file in uploaded_files:
            bytes_data = uploaded_file.read()
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(bytes_data)
                tmp_file_path = tmp_file.name
            text = extract_text_from_pdf(tmp_file_path)
            all_texts += text
            os.unlink(tmp_file_path)  # Remove o arquivo temporário

        if not all_texts.strip():
            st.error("Nenhum texto foi extraído dos PDFs carregados.")
            st.stop()

        # Dividir o texto em pedaços
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""],
        )
        chunks = text_splitter.split_text(all_texts)

        # Criar embeddings
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        vector_store = Chroma.from_texts(chunks, embeddings)

    st.success("PDFs processados e indexados com sucesso!")

    # Área para consultas
    st.header("Faça sua pergunta")

    user_query = st.text_input("Digite sua pergunta aqui:")
    if st.button("Consultar"):
        if user_query:
            with st.spinner("Buscando a resposta..."):
                docs = vector_store.similarity_search(user_query, k=5)
                context = "\n\n".join([doc.page_content for doc in docs])

                # Prompt para a API do OpenAI
                prompt = f"""
                Você é um assistente inteligente. Use as seguintes informações para responder à pergunta do usuário.
                Informações:
                {context}

                Pergunta: {user_query}
                Resposta:
                """

                try:
                    response = openai.Completion.create(
                        engine="text-davinci-003",
                        prompt=prompt,
                        max_tokens=500,
                        temperature=0.2,
                        top_p=1,
                        frequency_penalty=0,
                        presence_penalty=0,
                    )
                    answer = response.choices[0].text.strip()
                    st.write("**Resposta:**")
                    st.write(answer)
                except Exception as e:
                    st.error(f"Ocorreu um erro ao consultar a API do OpenAI: {e}")
        else:
            st.warning("Por favor, insira uma pergunta.")

else:
    st.info("Aguardando upload dos arquivos PDF.")

# Rodapé
st.markdown("""
---
Criado com ❤️ usando Streamlit e OpenAI
""")
