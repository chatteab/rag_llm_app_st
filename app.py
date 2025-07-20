import streamlit as st
import os
import dotenv
import uuid

from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage, AIMessage

from rag_methods import load_doc_to_db, load_url_to_db, stream_llm_response, stream_llm_rag_response

dotenv.load_dotenv()

st.set_page_config(
    page_title="RAG LLM app?", 
    layout="centered", 
    initial_sidebar_state="expanded"
)


# --- Header ---
st.html("""<h2 style="text-align: center;" <i> LLM RAG APP </i></h2>""")

# --- Initial Setup ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "rag_sources" not in st.session_state:
    st.session_state.rag_sources = []

if "az_openai_api_key" not in st.session_state:
    st.session_state.az_openai_api_key = ""

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there! How can I assist you today?"}
        ]
if "cloud_key_availavle" in st.session_state:
    st.session_state.cloud_key_available = False

def cloud_key():
        st.session_state.cloud_key_available = True

# --- Side Bar LLM API Tokens ---
with st.sidebar:
    if "AZ_OPENAI_API_KEY" in os.environ:
        default_openai_api_key = os.getenv("AZ_OPENAI_API_KEY") if os.getenv("AZ_OPENAI_API_KEY") is not None else ""  # only for development environment, otherwise it should return None
        with st.popover("🔐 Azure OpenAI"):
            openai_api_key = st.text_input(
            "Introduce your OpenAI API Key (https://platform.openai.com/)",
            value=default_openai_api_key,
            type="password",
            key="openai_api_key"
            )
        cloud_key()
    else:
        with st.popover("🔐 Azure OpenAI"):
            openai_api_key = st.text_input(
            "Introduce your OpenAI API Key (https://platform.openai.com/)",
            type="password",
            key="openai_api_key",
            on_change=cloud_key
            )
    st.session_state.az_openai_api_key = openai_api_key

# --- Main Content ---
# Checking if the user has introduced the OpenAI API Key, if not, a warning is displayed
if st.session_state.cloud_key_available:
    missing_openai = openai_api_key == "" or openai_api_key is None
    if missing_openai:
        st.write("#")
        st.warning("⬅️ Please introduce an API Key to continue...")
    else:
        # Sidebar
        with st.sidebar:
            st.divider()
            models = ["Azure OpenAI"]
            st.selectbox(
                "Select a Model", 
                options=models,
                key="model",
            )

            cols0 = st.columns(2)
            with cols0[0]:
                is_vector_db_loaded = ("vector_db" in st.session_state and st.session_state.vector_db is not None)
                st.toggle(
                    "Use RAG", 
                    value=is_vector_db_loaded, 
                    key="use_rag", 
                    disabled=not is_vector_db_loaded,
                )

            with cols0[1]:
                st.button("Clear Chat", on_click=lambda: st.session_state.messages.clear(), type="primary")

            st.header("RAG Sources:")
                
            # File upload input for RAG with documents
            st.file_uploader(
                "📄 Upload a document", 
                type=["pdf", "docx"],
                accept_multiple_files=True,
                on_change=load_doc_to_db,
                key="rag_docs",
            )

            # URL input for RAG with websites
            st.text_input(
                "🌐 Introduce a URL", 
                placeholder="https://example.com",
                on_change=load_url_to_db,
                key="rag_url",
            )

            with st.expander(f"📚 Documents in DB ({0 if not is_vector_db_loaded else len(st.session_state.rag_sources)})"):
                st.write([] if not is_vector_db_loaded else [source for source in st.session_state.rag_sources])

        # Main chat app
        llm_stream = AzureChatOpenAI(
            openai_api_version="2025-01-01-preview",
            openai_api_key=openai_api_key,
            azure_deployment="gpt-4o",
            temperature=0.3,
            streaming=True,
        )

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Your message"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""

                messages = [HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"]) for m in st.session_state.messages]

                if not st.session_state.use_rag:
                    st.write_stream(stream_llm_response(llm_stream, messages))
                else:
                    st.write_stream(stream_llm_rag_response(llm_stream, messages))
