import streamlit as st
from rag import SimpleRAG

st.set_page_config(page_title="RAG From Scratch", layout="wide")

# --- Header ---
st.title("RAG From Scratch")
st.caption(
    "An interactive explorer for understanding Retrieval-Augmented Generation. "
    "Upload a PDF, tweak the parameters, and see exactly what happens at each step."
)

# --- Sidebar: controls ---
st.sidebar.header("1. Upload a PDF")
uploaded_file = st.sidebar.file_uploader("Choose a PDF", type=["pdf"])

st.sidebar.header("2. Chunking Parameters")
chunk_size = st.sidebar.slider("Chunk size (characters)", 200, 2000, 500, step=50)
chunk_overlap = st.sidebar.slider("Chunk overlap (characters)", 0, 200, 50, step=10)

st.sidebar.header("3. Retrieval Parameters")
k = st.sidebar.slider("k (number of chunks to retrieve)", 1, 10, 3)

st.sidebar.header("4. Model")
model = st.sidebar.text_input("Ollama model name", value="tinyllama")

# --- Initialise RAG ---
if "rag" not in st.session_state:
    st.session_state.rag = SimpleRAG(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    st.session_state.raw_text = None
    st.session_state.chunks = None

rag = st.session_state.rag

# --- Ingest PDF ---
if uploaded_file:
    file_bytes = uploaded_file.read()

    # Re-ingest if file or chunk params changed
    needs_reingest = (
        st.session_state.raw_text is None
        or uploaded_file.name != st.session_state.get("file_name")
        or rag.update_chunk_params(chunk_size, chunk_overlap)
    )

    if needs_reingest:
        with st.spinner("Extracting text and chunking..."):
            raw_text = rag.load_pdf_bytes(file_bytes)
            st.session_state.raw_text = raw_text
            st.session_state.file_name = uploaded_file.name
        with st.spinner("Embedding chunks..."):
            chunks = rag.ingest(raw_text)
            st.session_state.chunks = chunks

    # --- Step 1: Show chunks ---
    st.header("Step 1: Chunking")
    st.markdown(
        f"Your document was split into **{len(st.session_state.chunks)} chunks** "
        f"of up to **{chunk_size} characters** with **{chunk_overlap} character overlap**."
    )

    with st.expander(f"Browse all {len(st.session_state.chunks)} chunks"):
        for i, chunk in enumerate(st.session_state.chunks):
            st.markdown(f"**Chunk {i}** ({len(chunk)} chars)")
            st.code(chunk, language=None)

    # --- Step 2: Query ---
    st.header("Step 2: Query & Retrieve")
    question = st.text_input("Ask a question about your document:")

    if question:
        with st.spinner("Embedding query and searching..."):
            result = rag.ask(question, k=k, model=model)

        # --- Step 3: Show retrieved chunks ---
        st.header("Step 3: Retrieved Chunks")
        st.markdown(
            f"FAISS searched **{result['total_chunks']} chunk embeddings** "
            f"and returned the **{result['k']} nearest** by L2 distance."
        )

        cols = st.columns(min(k, 3))
        for i, (chunk, dist, idx) in enumerate(
            zip(result["chunks"], result["distances"], result["indices"])
        ):
            with cols[i % len(cols)]:
                st.metric(label=f"Chunk {idx}", value=f"Distance: {dist:.4f}")
                st.code(chunk, language=None)

        # --- Step 4: Show prompt ---
        st.header("Step 4: The Prompt Sent to the LLM")
        st.markdown(
            "This is the full text that gets sent to the model. "
            "Notice how the retrieved chunks are concatenated as context."
        )
        with st.expander("View full prompt"):
            st.code(result["prompt"], language=None)

        # --- Step 5: Show answer ---
        st.header("Step 5: LLM Response")
        st.markdown(f"**Model:** `{model}`")
        st.info(result["answer"])

else:
    st.info("Upload a PDF in the sidebar to get started.")