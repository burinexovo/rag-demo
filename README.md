# RAG Demo — 個人 Profile 問答系統

基於 RAG（Retrieval-Augmented Generation）架構的個人 Profile 問答 Chatbot，使用本地 Ollama 模型搭配 ChromaDB 向量資料庫，讓使用者可以用自然語言查詢 `docs` 資料夾中的文本資訊（預設為檔名為 `profile.md`）。

## 技術架構

- **LLM**：Ollama（本地部署）
- **Embedding**：`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- **向量資料庫**：ChromaDB
- **框架**：LangChain
- **套件管理**：uv

## 前置需求

- Python >= 3.12
- [Ollama](https://ollama.com/) 已安裝並運行
- [uv](https://docs.astral.sh/uv/) 套件管理工具

## 快速開始

### 1. 安裝依賴

```bash
uv sync
```

### 2. 設定環境變數

```bash
cp .env.example .env
```

編輯 `.env`，填入你使用的 Ollama 模型名稱。

### 3. 準備 Profile 文件

```bash
cp docs/profile.example.md docs/profile.md
```

編輯 `docs/profile.md`，填入你的個人資訊。

### 4. 啟動問答系統

```bash
uv run python main.py
```

首次啟動會自動建立向量資料庫，之後啟動時可選擇是否重新 embedding。

## 專案結構

```
rag-demo/
├── main.py                    # 主程式
├── docs/
│   ├── profile.md             # 個人 Profile（不納入版控）
│   └── profile.example.md     # Profile 範本
├── chroma_db/                 # ChromaDB 向量資料庫（自動產生）
├── .env                       # 環境變數（不納入版控）
├── .env.example               # 環境變數範本
└── pyproject.toml             # 專案設定與依賴
```
