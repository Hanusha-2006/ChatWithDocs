import os
import gc
import time
import tempfile
import hashlib
import chromadb
import streamlit as st
import re

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import Docx2txtLoader, TextLoader

# ADD at the very top of app.py (line 1)
from dotenv import load_dotenv
load_dotenv()

#  Cache heavy objects — survive reruns without reloading
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource
def get_llm():
    return ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)

# @st.cache_resource
# def get_text_splitter():
#     return RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
@st.cache_resource
def get_semantic_splitter():
    embeddings = get_embeddings()
    return SemanticChunker(
        embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=85,
    )

def split_documents(docs):
    semantic_splitter = get_semantic_splitter()
    safety_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=150,
    )
    semantic_chunks = semantic_splitter.split_documents(docs)
    final_chunks = safety_splitter.split_documents(semantic_chunks)
    return final_chunks

#  Initialize all session state keys safely (run once)
def init_session_state():
    defaults = {
        "messages": [],        # chat messages {role, content}
        "context_memory": [],  # all retrieved chunks across turns
        "rag_chain": None,
        "retriever": None,
        "last_file_key": None,
        "vectorstore": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

#  Format full chat history for the LLM
def format_chat_history(messages):
    if not messages:
        return "No previous conversation."
    lines = []
    for msg in messages:
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)



# Format all previously retrieved doc chunks

def format_context_memory(context_memory):
    if not context_memory:
        return "None yet."
    lines = []
    for entry in context_memory:
        lines.append(f"[Page {entry['page']}]: {entry['content']}")
    return "\n\n".join(lines)

# Build RAG chain

def build_chain(retriever):
    prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. Answer ONLY from the document context provided.
Do NOT use any outside knowledge.
If the answer is not found in the context, say "I don't find this in the document."
Always mention the page number your answer comes from.

--- Current Retrieved Context ---
{context}

--- Previously Retrieved Context (earlier in this conversation) ---
{context_memory}

--- Conversation History ---
{chat_history}

--- Current Question ---
{question}

Answer:
""")

    llm = get_llm()

    def format_docs(docs):
        return "\n\n".join(
            f"[Page {doc.metadata.get('page', '?')}]: {doc.page_content}"
            for doc in docs
        )

    rag_chain = (
        {
            "context":        (lambda x: x["question"]) | retriever | format_docs,
            "question":       lambda x: x["question"],
            "chat_history":   lambda x: x["chat_history"],
            "context_memory": lambda x: x["context_memory"],
        }
        | prompt
        | llm
    )

    return rag_chain



#  Process uploaded PDFs

def process_documents(uploaded_files):
    # Release old vectorstore
    if st.session_state.vectorstore is not None:
        try:
            st.session_state.vectorstore._client.reset()
            del st.session_state.vectorstore
            st.session_state.vectorstore = None
            gc.collect()
            time.sleep(0.5)
        except Exception as e:
            print(f"⚠️ Could not release old store: {e}")

    text_splitter = get_semantic_splitter()
    embeddings = get_embeddings()
    all_splits = []

    for uploaded_file in uploaded_files:
        ext = os.path.splitext(uploaded_file.name)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix="ext") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        ext = os.path.splitext(uploaded_file.name)[1].lower()
        if ext == ".pdf":
            loader = PyPDFLoader(tmp_path)
        elif ext == ".docx":
            loader = Docx2txtLoader (tmp_path)
        elif ext == ".txt":
            loader = TextLoader (tmp_path, encoding="utf-8")
        else:
            st.warning(f"Unsupported file type: {uploaded_file.name}")
            os.unlink(tmp_path)
            continue
        docs = loader.load()
        splits = split_documents(docs)
        all_splits.extend(splits)
        os.unlink(tmp_path)

    collection_name = hashlib.md5(
        "".join([f.name for f in uploaded_files]).encode()
    ).hexdigest()[:12]

    client = chromadb.EphemeralClient()
    vectorstore = Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=embeddings,
    )
    vectorstore.add_documents(all_splits)

    st.session_state.vectorstore = vectorstore
    st.session_state.retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3, "fetch_k": 20, "lambda_mult": 0.7}
    )
    st.session_state.rag_chain = build_chain(st.session_state.retriever)

    # Preserve chat history — only add a note about new docs
    doc_names = ", ".join([f.name for f in uploaded_files])
    st.session_state.messages.append({
        "role": "assistant",
        "content": f"📄 Loaded: **{doc_names}**. Ask me anything about it!"
    })

# ----------------- Highlight Revlavent sentence -----------------
def highlight_relevant_sentence(chunk_text, query):
    """
    Finds the most relevant sentence in a chunk for the query
    and returns (before, highlight, after) for display.
    """
    # Split chunk into sentences
    sentences = re.split(r'(?<=[.!?])\s+', chunk_text.strip())
    
    if not sentences:
        return "", chunk_text, ""

    embeddings_model = get_embeddings()

    # Embed query and all sentences
    query_embedding = embeddings_model.embed_query(query)
    sentence_embeddings = embeddings_model.embed_documents(sentences)

    # Cosine similarity between query and each sentence
    def cosine_sim(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        mag_a = sum(x**2 for x in a) ** 0.5
        mag_b = sum(x**2 for x in b) ** 0.5
        return dot / (mag_a * mag_b + 1e-9)

    scores = [cosine_sim(query_embedding, se) for se in sentence_embeddings]
    best_idx = scores.index(max(scores))

    # Return context window: 1 sentence before and after the best match
    before = " ".join(sentences[max(0, best_idx - 1):best_idx])
    highlight = sentences[best_idx]
    after = " ".join(sentences[best_idx + 1: best_idx + 2])

    return before, highlight, after

# 🖥️ STREAMLIT UI

init_session_state()

st.title("📓 My NotebookLM")

# --- File Upload ---
uploaded_files = st.file_uploader(
    "Upload Documents",
    accept_multiple_files=True,
    type=["pdf", "docx", "txt"]
)

# Only re-index when files actually change
if uploaded_files:
    new_file_key = hashlib.md5(
        "".join([f.name for f in uploaded_files]).encode()
    ).hexdigest()[:12]

    if st.session_state.last_file_key != new_file_key:
        with st.spinner("⏳ Indexing your documents..."):
            process_documents(uploaded_files)
        st.session_state.last_file_key = new_file_key
        st.success(f"✅ {len(uploaded_files)} document(s) indexed!")

# --- Sidebar ---
with st.sidebar:
    st.header("💬 Chat History")
    st.write(f"**{len(st.session_state.messages)} message(s)** in memory")
    st.write(f"**{len(st.session_state.context_memory)} context chunk(s)** stored")

    if st.button("🗑️ Clear Chat + Context"):
        st.session_state.messages = []
        st.session_state.context_memory = []
        st.rerun()
    # ---- Download Chat History ----
    if st.session_state.messages:
        st.markdown("---")
        st.subheader("⬇️ Export Chat")

        # Build .txt content
        txt_lines = []
        for msg in st.session_state.messages:
            role = "User" if msg["role"] == "user" else "Assistant"
            txt_lines.append(f"{role}:\n{msg['content']}\n")
        txt_content = "\n".join(txt_lines)

        # Build .md content
        md_lines = []
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                md_lines.append(f"**🧑 User:**\n\n{msg['content']}\n")
            else:
                md_lines.append(f"**🤖 Assistant:**\n\n{msg['content']}\n")
        md_content = "\n---\n\n".join(md_lines)

        st.download_button(
            label="📄 Download as .txt",
            data=txt_content,
            file_name="chat_history.txt",
            mime="text/plain",
        )
        st.download_button(
            label="📝 Download as .md",
            data=md_content,
            file_name="chat_history.md",
            mime="text/markdown",
        )

    if st.session_state.messages:
        st.markdown("---")
        for msg in st.session_state.messages:
            icon = "🧑" if msg["role"] == "user" else "🤖"
            preview = msg["content"][:60] + "..." if len(msg["content"]) > 60 else msg["content"]
            st.caption(f"{icon} {preview}")

# --- Render all previous messages ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Chat Input ---
if st.session_state.rag_chain is None:
    st.info("👆 Please upload a PDF to get started.")
else:
    if query := st.chat_input("Ask anything about your documents..."):

        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        #  Fetch sources first (non-streaming, fast)
        sources = st.session_state.retriever.invoke(query)

        chat_history_str = format_chat_history(
            st.session_state.messages[:-1]
        )
        context_memory_str = format_context_memory(
            st.session_state.context_memory
        )

        #  Stream the response word by word
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""

            for chunk in st.session_state.rag_chain.stream({
                "question":       query,
                "chat_history":   chat_history_str,
                "context_memory": context_memory_str,
            }):
                # chunk is an AIMessageChunk — extract text content
                token = chunk.content if hasattr(chunk, "content") else str(chunk)
                full_response += token
                response_placeholder.markdown(full_response + "▌")  # blinking cursor effect

            # Final render without the cursor
            response_placeholder.markdown(full_response)

        #  Save completed response to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response
        })

        #  Save retrieved chunks to context memory
        for doc in sources:
            st.session_state.context_memory.append({
                "page": doc.metadata.get("page", "?"),
                "content": doc.page_content,
            })

        #  Sources expander
        with st.expander("📎 Sources used for this answer"):
            for i, doc in enumerate(sources):
                page = doc.metadata.get('page', '?')
                source = doc.metadata.get('source', 'uploaded file')
                filename = os.path.basename(source)

                # Get highlighted sentence
                before, highlight, after = highlight_relevant_sentence(
                    doc.page_content, query
                )

                st.markdown(f"**📄 {filename} — Page {page}**")

                # Display with highlight using markdown styling
                display_text = ""
                if before:
                    display_text += f"...{before} "
                display_text += f"**`{highlight}`**"   # highlighted sentence
                if after:
                    display_text += f" {after}..."

                st.markdown(
                    f"""
                    <div style="
                        background-color: #1e1e2e;
                        border-left: 4px solid #f0a500;
                        padding: 10px 15px;
                        border-radius: 4px;
                        font-size: 0.9em;
                        margin-bottom: 10px;
                        line-height: 1.6;
                    ">
                        {display_text}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                if i < len(sources) - 1:
                    st.divider()
                
