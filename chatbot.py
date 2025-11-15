import streamlit as st
from huggingface_hub import InferenceClient
from langchain_core.language_models.llms import LLM
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_core.pydantic_v1 import Field
from dotenv import load_dotenv
import os

# Load API token from .env
load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not HF_TOKEN:
    st.error("❌ Hugging Face API token not found. Please add it to your .env file.")
    st.stop()

st.set_page_config(page_title="Q&A Chatbot", page_icon="🤖")
st.title("🤖 Q&A Chatbot")
st.markdown("Hello! 👋 I'm your chatbot. Ask me anything and I'll try to help you.")

MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"

# Base client
client = InferenceClient(token=HF_TOKEN, model=MODEL)

# ---- FIXED CUSTOM LLM ----
class HuggingFaceChatLLM(LLM):
    client: InferenceClient = Field(...)

    @property
    def _llm_type(self):
        return "huggingface-custom-llm"

    def _call(self, prompt: str, stop=None):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        try:
            response = self.client.chat_completion(
                messages=messages,
                max_tokens=256
            )
            return response.choices[0].message["content"].strip()
        except Exception as e:
            return f"❌ Failed to get response: {e}"


# Conversation memory
memory = ConversationBufferMemory(memory_key="history", return_messages=True)

# Model instance
model = HuggingFaceChatLLM(client=client)

prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{user_input}")
])

# Chain
chain = LLMChain(
    llm=model,
    prompt=prompt_template,
    memory=memory
)

# Session messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Chat input
if prompt_text := st.chat_input("Type your question here..."):
    st.session_state.messages.append({"role": "user", "content": prompt_text})
    
    with st.chat_message("user"):
        st.write(prompt_text)

    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            reply = chain.run(user_input=prompt_text)
            st.write(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})
