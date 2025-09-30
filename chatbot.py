import streamlit as st
from huggingface_hub import InferenceClient
from langchain.llms.base import LLM
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from pydantic import BaseModel, Field

HF_TOKEN = st.secrets["HUGGINGFACEHUB_API_TOKEN"]

if not HF_TOKEN:
    st.error("❌ Hugging Face API token not found in Streamlit secrets.")
    st.stop()

st.set_page_config(page_title="Q&A Chatbot", page_icon="🤖")
st.title("🤖 Q&A Chatbot")
st.markdown("Hello! 👋 I'm your chatbot powered by **Mistral-7B-Instruct**")

st.sidebar.success("✅ Model: mistralai/Mistral-7B-Instruct-v0.3")
st.sidebar.write("API Token loaded:", HF_TOKEN is not None)

MODEL = "mistralai/Mistral-7B-Instruct-v0.3"

# Corrected initialization
client = InferenceClient(token=HF_TOKEN, model=MODEL)

class HuggingFaceChatLLM(LLM, BaseModel):
    client: InferenceClient = Field(...)

    @property
    def _llm_type(self):
        return "huggingface-chat"

    def _call(self, prompt: str, stop=None) -> str:
        messages = [{"role": "system", "content": "You are a helpful assistant."}]
        messages.append({"role": "user", "content": prompt})
        response = self.client.chat_completion(messages=messages, max_tokens=256)
        return response.choices[0].message["content"]

memory = ConversationBufferMemory(memory_key="history", return_messages=True)

llm = HuggingFaceChatLLM(client=client)
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{user_input}")
])
chain = LLMChain(llm=llm, prompt=prompt_template, memory=memory)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt_text := st.chat_input("Type your message here..."):
    st.session_state.messages.append({"role": "user", "content": prompt_text})
    with st.chat_message("user"):
        st.write(prompt_text)

    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            reply = chain.run(user_input=prompt_text)
            st.write(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})
