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
st.markdown("Hello! 👋 I'm your chatbot. Ask me anything and I'll try to help you.")

MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
client = InferenceClient(token=HF_TOKEN, model=MODEL)

class HuggingFaceChatLLM(LLM, BaseModel):
    client: InferenceClient = Field(...)
    memory: ConversationBufferMemory = None

    @property
    def _llm_type(self):
        return "huggingface-chat"

    def _call(self, prompt: str, stop=None) -> str:
        messages = [{"role": "system", "content": "You are a helpful assistant."}]
        if self.memory:
            for msg in self.memory.buffer:
                role = "assistant" if msg.type == "ai" else "user"
                messages.append({"role": role, "content": msg.content})
        messages.append({"role": "user", "content": prompt})
        response = self.client.chat_completion(messages=messages, max_tokens=512)
        return response.choices[0].message["content"]

memory = ConversationBufferMemory(memory_key="history", return_messages=True)

llm = HuggingFaceChatLLM(client=client, memory=memory)
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

if prompt_text := st.chat_input("Type your question here..."):
    st.session_state.messages.append({"role": "user", "content": prompt_text})
    with st.chat_message("user"):
        st.write(prompt_text)

    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            reply = chain.run(user_input=prompt_text)
            st.write(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})

# Hidden reset chat (functionality preserved, button hidden)
hide_button_style = """
    <style>
    button[title="Reset Chat"] {
        display: none;
    }
    </style>
"""
st.markdown(hide_button_style, unsafe_allow_html=True)

if st.button("🔄 Reset Chat"):
    st.session_state.messages = []
    memory.clear()
    st.experimental_rerun()
