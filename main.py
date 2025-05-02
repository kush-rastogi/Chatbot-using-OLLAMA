import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Langsmith tracking (optional)
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACKING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Simple Q&A Chatbot With OLLAMA"

# Prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please respond to user queries."),
    ("user", "Question: {question}")
])

# Function to generate response
def generate_response(question, engine, temperature, max_tokens):
    llm = Ollama(model=engine)
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    answer = chain.invoke({'question': question})
    return answer

# Streamlit UI
st.set_page_config(page_title="Q&A Chatbot with Ollama", page_icon="💬")
st.title("💬 Enhanced Q&A Chatbot With OLLAMA")
st.markdown("Ask me anything! I'm powered by open-source models using LangChain + Ollama.")

# Sidebar controls
st.sidebar.header("Model Configuration")
llm = st.sidebar.selectbox("Select Open Source model", ["mistral", "llama2", "gemma"])
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7)
max_tokens = st.sidebar.slider("Max Tokens", 50, 300, 150)

# Show model info
st.sidebar.markdown(f"**Model Selected:** `{llm}`")
st.sidebar.markdown("Use sliders to control the response style.")

# Chat history using session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Clear chat button
if st.button("🗑️ Clear Chat"):
    st.session_state.chat_history = []

# User input
user_input = st.text_input("You:", placeholder="Type your question here...")

# If user enters a query
if user_input:
    with st.spinner("Thinking..."):
        response = generate_response(user_input, llm, temperature, max_tokens)
        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("Bot", response))

# Display chat history
for speaker, msg in st.session_state.chat_history:
    if speaker == "You":
        st.markdown(f"**🧑 You:** {msg}")
    else:
        st.markdown(f"**🤖 Bot:** {msg}")

