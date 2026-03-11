import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
os.environ["TRANSFORMERS_OFFLINE"] = "1"

load_dotenv()
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")
CHROMA_DIR = os.getenv("CHROMA_DIR")


def main():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    # 判斷是否重新 embedding
    if os.path.exists(CHROMA_DIR):
        ans = input("偵測到現有向量庫，是否重新 embedding？(y/N)：").strip().lower()
        if ans == "y":
            import shutil
            shutil.rmtree(CHROMA_DIR)
            vectorstore = _build_vectorstore(embeddings)
        else:
            vectorstore = Chroma(persist_directory=CHROMA_DIR,
                                 embedding_function=embeddings)
    else:
        vectorstore = _build_vectorstore(embeddings)

    # Retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Prompt
    prompt = ChatPromptTemplate.from_template(
        """根據以下資料回答問題，請用繁體中文回答。

        資料：
        {context}

        問題：{question}
        """
    )

    # LLM
    llm = OllamaLLM(model=OLLAMA_MODEL)

    # Chain
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # 問答
    print("RAG 問答系統啟動，輸入 'q' 離開")
    while True:
        question = input("\n你的問題：")
        if question.lower() == "q":
            break
        result = chain.invoke(question)
        print(f"\n回答：{result}")


def _build_vectorstore(embeddings):
    loader = TextLoader("docs/profile.md", encoding="utf-8")
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(documents)
    return Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_DIR)


if __name__ == "__main__":
    main()
