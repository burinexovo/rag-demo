"""
RAG 個人 Profile 問答系統

使用 LangChain + Ollama + ChromaDB 建構，
將 docs/profile.md 切塊後存入向量資料庫，透過語意檢索回答使用者問題。
"""

import os
from dotenv import load_dotenv
# 讀取純文字檔案（如 profile.md）為 LangChain Document 物件
from langchain_community.document_loaders import TextLoader

# 將長文本遞迴切割成固定大小的 chunk，保留上下文重疊
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ChromaDB 向量資料庫的 LangChain 封裝，負責存取 embedding 向量
from langchain_community.vectorstores import Chroma

# 用 HuggingFace sentence-transformers 模型將文本轉為向量
from langchain_community.embeddings import HuggingFaceEmbeddings

# 連接本地 Ollama 服務的 LLM 封裝
from langchain_ollama import OllamaLLM

# 定義 Prompt 模板，支援變數填充（{context}、{question}）
from langchain_core.prompts import ChatPromptTemplate

# 直通元件，讓使用者輸入原封不動傳遞到 chain 的下一步
from langchain_core.runnables import RunnablePassthrough

# 將 LLM 輸出解析為純字串
from langchain_core.output_parsers import StrOutputParser

# 強制離線模式，避免 sentence-transformers 嘗試連線下載模型
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# 載入 .env 環境變數
load_dotenv()
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")
CHROMA_DIR = os.getenv("CHROMA_DIR")


def main():
    # 初始化多語言 embedding 模型（支援中文語意檢索）
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    # 若向量庫已存在，詢問是否重建；否則直接建立
    if os.path.exists(CHROMA_DIR):
        ans = input("偵測到現有向量庫，是否重新 embedding？(y/N)：").strip().lower()
        if ans == "y":
            import shutil
            shutil.rmtree(CHROMA_DIR)
            vectorstore = _build_vectorstore(embeddings)
        else:
            # 直接載入現有向量庫
            vectorstore = Chroma(persist_directory=CHROMA_DIR,
                                 embedding_function=embeddings)
    else:
        vectorstore = _build_vectorstore(embeddings)

    # 建立檢索器，每次查詢回傳最相關的 3 個文本片段
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # 定義 Prompt 模板，指示 LLM 根據檢索到的資料用繁體中文回答
    prompt = ChatPromptTemplate.from_template(
        """根據以下資料回答問題，請用繁體中文回答。

        資料：
        {context}

        問題：{question}
        """
    )

    # 初始化本地 Ollama LLM
    llm = OllamaLLM(model=OLLAMA_MODEL)

    # 組合 RAG Chain：使用者問題 → 語意檢索 → Prompt 填充 → LLM 生成 → 解析輸出
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # 互動式問答迴圈
    print("RAG 問答系統啟動，輸入 'q' 離開")
    while True:
        question = input("\n你的問題：")
        if question.lower() == "q":
            break
        result = chain.invoke(question)
        print(f"\n回答：{result}")


def _build_vectorstore(embeddings):
    """讀取 profile.md，切塊後建立 ChromaDB 向量資料庫。"""
    loader = TextLoader("docs/profile.md", encoding="utf-8")
    documents = loader.load()

    # 每塊 800 字元，相鄰塊重疊 100 字元以保留上下文
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(documents)

    return Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_DIR)


if __name__ == "__main__":
    main()
