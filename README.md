
# 🤖 AI Chatbot using LangChain + HuggingFace API (Streamlit)

This project is a **simple and powerful AI Chatbot** built using **LangChain**, **Streamlit**, and the **HuggingFace Inference API**.
It uses an LLM such as **Llama 3**, **Mistral**, or **Gemma** through HuggingFace to generate intelligent responses.
The chatbot provides a clean UI and works entirely on cloud API—no GPU required.

---

## 🚀 Features

### 💬 Intelligent Chatbot

* Uses HuggingFace Inference API
* Supports all HuggingFace text generation models
* Handles long conversations
* Fast and responsive

### 🧠 LangChain Integration

* Uses `LLMChain` + custom prompt
* Implements conversation memory
* Clean and modular code

### 🌐 Streamlit UI

* Professional minimal UI
* Real-time conversation
* Auto-scroll chat
* Mobile-friendly

### 🔐 No Local GPU Needed

All heavy work is done using **HuggingFace Inference API**.


---

## 🔐 Environment Setup

Create a `.env` file:

```
HUGGINGFACEHUB_API_TOKEN=your_api_key_here
```

Replace `your_api_key_here` with your HuggingFace token.

---

## ▶️ How to Run Locally

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/ai-chatbot-langchain.git
cd ai-chatbot-langchain
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Add HuggingFace API Key

Create `.env`:

```
HUGGINGFACEHUB_API_TOKEN=your_token_here
```

### 4️⃣ Run App

```bash
streamlit run app.py
```



---

## 📤 Deploying

### 🚀 Streamlit Cloud

1. Upload repo to GitHub
2. Go to [https://share.streamlit.io](https://share.streamlit.io)
3. Select repo
4. Add environment variable:

| Key                      | Value      |
| ------------------------ | ---------- |
| HUGGINGFACEHUB_API_TOKEN | your_token |

5. Deploy 🎉

### 🚀 HuggingFace Spaces

* Create Space → Streamlit
* Upload files
* Add secret token
* Deploy instantly

---

## 🔮 Future Enhancements

* Chat history saving
* Model selection dropdown
* Theme toggle
* Voice input

