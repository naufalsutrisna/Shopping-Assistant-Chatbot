# 🛍️ AI-Powered Shopping Assistant

## 🚀 Overview
The **AI-Powered Shopping Assistant** is a chatbot that helps users find products based on their descriptions, budget, and preferences. It leverages **LLMs (Large Language Models)** and **vector search** to provide accurate product recommendations.

## 📂 Dataset
We use the **Amazon Product Dataset** from Kaggle:
[Amazon Product Dataset](https://www.kaggle.com/datasets/lokeshparab/amazon-products-dataset/data)

## 🛠️ Tech Stack
- **Python**
- **Streamlit** (for UI)
- **FastAPI** (for API backend)
- **FAISS** (for efficient similarity search)
- **Sentence-Transformers** (for text embeddings)
- **Torch (PyTorch)**
- **Transformers** (for NLP models)
- **LangChain** (for LLM-based applications)
- **OpenAI API** (optional for LLM-powered responses)
- **Pandas** (for data handling)

## 🏗️ Installation & Setup
### 1️⃣ Create a Virtual Environment
```bash
python3 -m venv streamlit-env
source streamlit-env/bin/activate  # On Windows, use `streamlit-env\Scripts\activate`
```

### 2️⃣ Install Dependencies
```bash
pip install streamlit sentence-transformers faiss-cpu torch transformers langchain openai
```

### 3️⃣ Run the Backend API
```bash
uvicorn search_api:app --host 0.0.0.0 --port 8000 --reload
```

### 4️⃣ Run the Streamlit App (Frontend)
```bash
streamlit run app.py
```

## 🔥 Features
✅ **Natural Language Search** – Users can describe what they need, and the chatbot finds matching products.
✅ **Budget Filtering** – Search results respect user-defined price limits.
✅ **FAISS for Efficient Search** – Enables fast and scalable similarity search.
✅ **Interactive UI** – Built with Streamlit for a smooth user experience.

## 📌 Roadmap
- Implement more advanced NLP-based filtering
- Enhance UI with product images and details
- Create API, because currently, the model is called directly inside Streamlit without an API
