# capstomeproject-Agentic_AI-

---

🎓 AI Student Education & Financing Assistant

An AI-powered assistant designed to help students in India with college selection, scholarships, entrance exams, and education loans using advanced RAG (Retrieval-Augmented Generation) and LLM-based reasoning.


---

🚀 Features

🔍 Knowledge-Based Q&A (RAG pipeline using ChromaDB)

🧠 Smart Query Routing (retrieve / memory / tool)

🌐 Web Search Integration (DuckDuckGo for real-time info)

💬 Conversational Memory

📊 Answer Evaluation (Faithfulness Scoring)

⚡ Fast LLM Responses (Groq - LLaMA 3)



---

🏗️ System Architecture

User Query
   ↓
Router (Decides: Retrieve / Tool / Memory)
   ↓
Retriever (ChromaDB) OR Web Search
   ↓
LLM (Groq - LLaMA 3)
   ↓
Evaluation (Faithfulness Score)
   ↓
Response (Streamlit UI)


---

🛠️ Tech Stack

Frontend: Streamlit

LLM: Groq (LLaMA 3)

Embeddings: SentenceTransformers

Vector DB: ChromaDB

Agent Framework: LangGraph

Evaluation: RAGAS

Search Tool: DuckDuckGo



---

📂 Project Structure

├── capstone_streamlit.py   # Main Streamlit App
├── capstone.ipynb          # Development Notebook
├── requirements.txt        # Dependencies
├── README.md               # Project Documentation


---

⚙️ Installation

1️⃣ Clone the repository

git clone https://github.com/your-username/your-repo.git
cd your-repo

2️⃣ Create virtual environment

python -m venv venv
venv\Scripts\activate   # Windows

3️⃣ Install dependencies

pip install -r requirements.txt


---

🔑 Environment Setup

Create a .env file:

GROQ_API_KEY=your_api_key_here


---

▶️ Run the App

streamlit run capstone_streamlit.py


---

📊 Evaluation Metrics

The system evaluates responses using:

Faithfulness

Answer Relevancy

Context Precision


Fallback manual evaluation is used if RAGAS is unavailable.


---

💡 Example Use Cases

🎯 "Which engineering colleges are best under JEE Main?"

💰 "What scholarships are available for SC students?"

🏦 "How to apply for education loans in India?"

🌍 "Funding options for studying abroad?"



---

⚠️ Limitations

Depends on knowledge base quality

Web search may vary in accuracy

Requires internet & API access



---

🔮 Future Improvements

📈 Evaluation dashboard (metrics tracking)

🤖 Multi-agent system

🌍 Multilingual support

📱 Mobile app version



---



📜 License

This project is for academic and educational purposes.


---

👨‍💻 Author

Ujjwal Kumar
B.Tech CSE | AI Enthusiast


---

⭐ Support

If you like this project, give it a ⭐ on GitHub!
