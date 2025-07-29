import streamlit as st
import requests
import re
import json
import time
import os

st.set_page_config(page_title="Chat With Your Code", layout="wide")

st.title("💬 Chat With Your Codebase")
st.markdown("Ask a question about your C/C++ codebase and get an intelligent response.")

# --- Session State for Query Log ---
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'uploaded_files_processed' not in st.session_state:
    st.session_state.uploaded_files_processed = False

# --- Sidebar for Parameters ---
st.sidebar.header("Configuration")

# File Uploader Section
st.sidebar.subheader("Upload Code Files")
uploaded_file = st.sidebar.file_uploader(
    "Upload C/C++/Header files",
    type=["c", "cpp", "h"],
    accept_multiple_files=True,
    help="Select one or more .c, .cpp, or .h files to upload and process."
)

if uploaded_file and not st.session_state.uploaded_files_processed:
    if st.sidebar.button("Process Uploaded Files"):
        st.sidebar.info("Processing files... This may take a moment.")
        
        backend_upload_url = "http://localhost:8000/upload_code_file"
        all_uploads_successful = True
        
        for file in uploaded_file:
            files_to_send = {'file': (file.name, file.getvalue(), file.type)}
            try:
                upload_response = requests.post(backend_upload_url, files=files_to_send)
                if upload_response.status_code == 200:
                    st.sidebar.success(f"Successfully processed: {file.name}")
                else:
                    st.sidebar.error(f"Failed to process {file.name}: {upload_response.text}")
                    all_uploads_successful = False
            except requests.exceptions.ConnectionError:
                st.sidebar.error("⚠️ Could not connect to the backend API. Please ensure it's running at `http://localhost:8000`.")
                all_uploads_successful = False
                break
            except Exception as e:
                st.sidebar.error(f"⚠️ An unexpected error occurred during upload for {file.name}: {e}")
                all_uploads_successful = False
                break
        
        if all_uploads_successful:
            st.session_state.uploaded_files_processed = True
            st.sidebar.success("All selected files have been processed and indexed!")
            st.rerun() # Rerun to clear uploader and reflect new state
        else:
            st.sidebar.warning("Some files failed to process. Check logs for details.")

elif st.session_state.uploaded_files_processed:
    st.sidebar.success("Files already processed. You can now ask questions!")
    # Optionally, provide a way to clear processed state and upload new files
    if st.sidebar.button("Upload New Files (Clear Current Index)"):
        st.session_state.uploaded_files_processed = False
        # You might also want a backend endpoint to clear the ChromaDB collection here
        try:
            clear_response = requests.post("http://localhost:8000/clear_codebase")
            if clear_response.status_code == 200:
                st.sidebar.info("Codebase cleared. Upload new files.")
                st.session_state.chat_history = [] # Clear chat history too
                st.rerun()
            else:
                st.sidebar.error(f"Failed to clear codebase: {clear_response.text}")
        except requests.exceptions.ConnectionError:
            st.sidebar.error("Could not connect to backend to clear codebase.")


temperature = st.sidebar.slider(
    "LLM Temperature", 0.0, 1.0, 0.2, 0.05,
    help="Controls the creativity/randomness of the LLM. Lower values (e.g., 0.1-0.3) for more factual, higher values (e.g., 0.7-1.0) for more creative."
)

top_k = st.sidebar.slider(
    "Number of Chunks to Retrieve (Top K)", 1, 20, 5, 1,
    help="The maximum number of most similar code chunks to retrieve from the database."
)

similarity_threshold = st.sidebar.slider(
    "Similarity Threshold (Lower is More Similar)", 0.0, 2.0, 0.7, 0.01,
    help="Only chunks with a distance score below this threshold will be used."
)

# --- Main Content Area ---
main_col, sidebar_col = st.columns([3, 1])

with main_col:
    question = st.text_input("🔍 Ask your question about the codebase:", placeholder="e.g., What does the 'lprint_brother_get_status' function do?")

# API URL for querying
API_QUERY_URL = "http://localhost:8000/ask/"

# Ask button
if st.button("Ask") and question:
    if not st.session_state.uploaded_files_processed:
        st.warning("Please upload and process code files first in the sidebar.")
    else:
        start_time = time.time()
        with st.spinner("Thinking..."):
            try:
                payload = {
                    "query": question,
                    "temperature": temperature,
                    "top_k": top_k,
                    "similarity_threshold": similarity_threshold
                }
                response = requests.post(API_QUERY_URL, json=payload)
                end_time = time.time()
                response_time = end_time - start_time

                if response.status_code == 200:
                    data = response.json()
                    answer_text = data["answer"]
                    retrieved_context_info = data.get("retrieved_context", [])

                    # ✅ Stylish single output block
                    st.subheader("✅ Answer from LLM")
                    # Handle Mermaid diagrams if present in the answer
                    mermaid_regex = r"```mermaid\n([\s\S]*?)\n```"
                    mermaid_match = re.search(mermaid_regex, answer_text)

                    if mermaid_match:
                        mermaid_code = mermaid_match.group(1).strip()
                        answer_display_text = re.sub(mermaid_regex, '', answer_text).strip()
                        st.markdown(answer_display_text)

                        st.subheader("📊 Generated Diagram:")
                        st.components.v1.html(
                            f"""
                            <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
                            <div class="mermaid" id="mermaid-diagram-{len(st.session_state.chat_history)}">
                            {mermaid_code}
                            </div>
                            <script>
                            // Initialize Mermaid if not already, or render new diagram
                            if (typeof window.mermaidInitialized === 'undefined') {{
                                mermaid.initialize({{ startOnLoad: true }});
                                window.mermaidInitialized = true;
                            }} else {{
                                // Need a unique ID for each render call
                                let elementId = 'mermaid-diagram-{len(st.session_state.chat_history)}';
                                let element = document.getElementById(elementId);
                                if (element) {{
                                    mermaid.render('graphDiv-' + elementId, `{mermaid_code}`).then(({{svg, bindFunctions}}) => {{
                                        element.innerHTML = svg;
                                    }});
                                }}
                            }}
                            </script>
                            """,
                            height=500, scrolling=True
                        )
                    else:
                        st.code(answer_text, language="markdown") # Use st.code for raw text to preserve formatting

                    st.info(f"Response Time: {response_time:.2f} seconds")

                    # Add to chat history
                    st.session_state.chat_history.append({
                        "question": question,
                        "answer": answer_text,
                        "retrieved_context_info": retrieved_context_info,
                        "response_time": response_time
                    })

                    # --- Retrieved Code Context ---
                    with st.expander("📄 Show Retrieved Context Details"):
                        if retrieved_context_info:
                            for i, chunk in enumerate(retrieved_context_info):
                                st.markdown(f"**Chunk {i+1}**")
                                st.write(f"- **Type:** `{chunk.get('type', 'N/A')}`")
                                source_path = chunk.get('source', 'N/A')
                                start_line = chunk.get('start_line', -1)

                                if source_path != 'N/A' and start_line != -1:
                                    # Create a dummy path for display purposes as uploaded files don't have local file path directly
                                    display_path = os.path.basename(source_path) # Just show filename
                                    st.markdown(f"- **Source:** `{display_path}` (Line {start_line})")
                                else:
                                    st.write(f"- **Source:** `{source_path}`")
                                    st.write(f"- **Start Line:** `{start_line}`")

                                st.write(f"- **Function:** `{chunk.get('function_name', 'N/A')}`")
                                st.write(f"- **Similarity Distance:** `{chunk.get('distance', -1):.4f}`")
                                st.code(chunk.get("content", "No content found"), language="cpp")
                                st.markdown("---")
                        else:
                            st.info("No relevant code context was retrieved based on your query and settings.")
                else:
                    st.error(f"Error {response.status_code}: {response.text}")
            except requests.exceptions.ConnectionError:
                st.error("⚠️ Could not connect to the backend API. Please ensure it's running at `http://localhost:8000`.")
            except Exception as e:
                st.error(f"⚠️ An unexpected error occurred: {e}")

# --- Sidebar: Function Reference + History ---
with sidebar_col:
    st.subheader("📁 Function References")
    if st.session_state.chat_history:
        last_retrieved_info = st.session_state.chat_history[-1].get("retrieved_context_info", [])
        unique_functions = []
        seen = set()
        for chunk in last_retrieved_info:
            func = chunk.get("function_name")
            if func and func not in seen:
                unique_functions.append(f"{func} (Line: {chunk.get('start_line', 'N/A')})")
                seen.add(func)

        if unique_functions:
            st.markdown("---")
            st.markdown("**Functions in Retrieved Context:**")
            for func in unique_functions:
                st.markdown(f"- {func}")
        else:
            st.info("No specific functions found in the current context.")
    else:
        st.info("Ask a question to see function references.")

    st.subheader("💬 Query History")
    if st.session_state.chat_history:
        for i, entry in enumerate(reversed(st.session_state.chat_history)):
            with st.expander(f"Q: {entry['question'][:50]}..."):
                st.markdown(f"**Question:** {entry['question']}")
                st.markdown(f"**Answer:**")
                st.code(entry["answer"], language="markdown")
                st.markdown(f"**Response Time:** {entry.get('response_time', 0):.2f} seconds")
    else:
        st.info("Your chat history will appear here.")

st.markdown("---")
st.markdown("Powered by Google Gemini and ChromaDB")