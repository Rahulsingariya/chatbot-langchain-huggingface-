import streamlit as st
from huggingface_hub import InferenceClient
from langchain.llms.base import LLM
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from pydantic import BaseModel, Field


HF_TOKEN = st.secrets["HUGGINGFACEHUB_API_TOKEN"]

if not HF_TOKEN:
    st.error("❌ Hugging Face API token not found. Please add it to Streamlit secrets.")
    st.stop()

st.set_page_config(page_title="Q&A Chatbot", page_icon="🤖")
st.title("🤖 Q&A Chatbot")
st.markdown("Hello! 👋 I'm your chatbot. Ask me anything and I'll try to help you.")


MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"


client = InferenceClient(token=HF_TOKEN, model=MODEL)

class HuggingFaceChatLLM(LLM, BaseModel):
    client: InferenceClient = Field(...)

    @property
    def _llm_type(self):
        return "huggingface-chat"

    def _call(self, prompt: str, stop=None) -> str:
        messages = [{"role": "system", "content": "You are a helpful assistant."}]
        messages.append({"role": "user", "content": prompt})

        try:
            response = self.client.chat_completion(messages=messages, max_tokens=256)
            answer = response.choices[0].message["content"]
            return answer.strip()
        except Exception as e:
            return "❌ Failed to respond"


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


if prompt_text := st.chat_input("Type your question here..."):
    st.session_state.messages.append({"role": "user", "content": prompt_text})
    with st.chat_message("user"):
        st.write(prompt_text)

    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            reply = chain.run(user_input=prompt_text)
            st.write(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})

