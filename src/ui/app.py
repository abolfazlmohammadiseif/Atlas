"""
Streamlit UI for the RAG Application.
Provides a chat interface and a sidebar for document ingestion.
"""

import requests
import streamlit as st

# FastAPI server URL
API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="centered")
st.title("🤖 Chat with your Documents (RAG)")

# --- Sidebar for Document Ingestion ---
with st.sidebar:
    st.header("📄 Add Knowledge")
    st.write("Paste text here to add it to the AI's knowledge base.")

    doc_text = st.text_area("Document Text", height=200)
    doc_source = st.text_input("Source (e.g., 'Policy v1')", value="User Input")

    if st.button("Ingest Document"):
        if doc_text.strip():
            with st.spinner("Processing and storing..."):
                try:
                    response = requests.post(
                        f"{API_URL}/ingest",
                        json={"text": doc_text, "metadata": {"source": doc_source}},
                        timeout=30,
                    )
                    if response.status_code == 200:
                        st.success("✅ Document ingested successfully!")
                    else:
                        st.error(f"❌ Error: {response.text}")
                except requests.exceptions.ConnectionError:
                    st.error("❌ Cannot connect to the API. Is FastAPI running?")
        else:
            st.warning("Please enter some text first.")

# --- Main Chat Interface ---
# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! I'm ready to answer questions based on your documents. \
                      What would you like to know?",
        }
    ]

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Display user message
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Fetch response from FastAPI backend
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(f"{API_URL}/ask", json={"question": prompt}, timeout=60)
                if response.status_code == 200:
                    answer = response.json().get("answer", "No answer provided.")
                    st.markdown(answer)
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                else:
                    error_msg = f"Error: {response.status_code}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
            except requests.exceptions.ConnectionError:
                error_msg = "Cannot connect to the backend API. Please start the FastAPI server."
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
