import streamlit as st
import os
import time
import json
from pathlib import Path
import shutil
from dotenv import load_dotenv
import sys
import logging

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    Settings,
    PromptTemplate
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()

#Configuration Constants
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

DATA_DIRECTORY = Path("./news_articles")
PERSIST_DIR = Path("./persistent_storage")
LLAMAINDEX_STORAGE_DIR = PERSIST_DIR / "llamaindex_storage"
INDEXED_FILES_METADATA = PERSIST_DIR / "indexed_files_llamaindex.json"
FEEDBACK_LOG_FILE = PERSIST_DIR / "feedback_log_llamaindex.jsonl"
MODEL_CACHE_DIR = "./model_cache"

# --- Logo Path and Assistant Name ---
# IMPORTANT: This path is set to look for 'claire_logo.jpg'
# Ensure this file is in the 'static' folder, which is next to your app.py script.
LOGO_PATH = "static/claire_logo.jpg" # Reverted to string, as Path object conversion handled in HTML if needed
ASSISTANT_NAME = "CLAIRE"
# --- END Logo Path and Assistant Name ---

# --- Model & RAG Parameters ---
GROQ_MODEL_NAME = "llama3-8b-8192"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
INITIAL_RETRIEVAL_K = 5 # How many chunks retriever gets initially

# --- Ensure necessary directories exist at script startup ---
try:
    DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)
    PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    LLAMAINDEX_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    Path(MODEL_CACHE_DIR).mkdir(parents=True, exist_ok=True)
    Path("static").mkdir(parents=True, exist_ok=True) # Ensure 'static' folder exists for the logo
    logging.info("Required directories created or confirmed existing.")
except OSError as e:
    logging.error(f"Error creating directories: {e}")
    st.error(f"Fatal Error: Could not create required directories. Check permissions. Error: {e}", icon="🔥")
    st.stop()


def print_elapsed_time(start_time, task_name="Task"):
    """Logs and returns elapsed time."""
    end_time = time.time()
    elapsed = f"{end_time - start_time:.2f}"
    logging.info(f"---> {task_name} completed in {elapsed} seconds.")
    return elapsed

@st.cache_resource
def init_embedding_model(model_name):
    logging.info(f"Initializing embedding model (LlamaIndex): {model_name}...")
    t1=time.time()
    try:
        embed_model = HuggingFaceEmbedding(model_name=model_name, cache_folder=MODEL_CACHE_DIR)
        print_elapsed_time(t1, f"Embedding model '{model_name}' init")
        return embed_model
    except Exception as e:
        logging.error(f"Failed to initialize embedding model '{model_name}': {e}", exc_info=True)
        st.error(f"Failed embedding model init: {e}", icon="🔥")
        return None

@st.cache_resource
def init_llm(_api_key, model_name, temp=0.1):
    logging.info(f"Initializing LLM (LlamaIndex/Groq): {model_name}...")
    t1=time.time()
    if not _api_key:
        logging.error("Groq API Key is missing.")
        st.error("Groq API Key missing. Please set GROQ_API_KEY environment variable.", icon="🔥")
        return None
    try:
        llm = Groq(model=model_name, api_key=_api_key, temperature=temp)
        print_elapsed_time(t1, f"LLM '{model_name}' init")
        return llm
    except Exception as e:
        logging.error(f"Failed to initialize LLM '{model_name}': {e}", exc_info=True)
        st.error(f"Failed LLM init: {e}", icon="🔥")
        return None

# --- LlamaIndex Index Handling ---
def load_index(storage_dir, _embed_model):
    if not storage_dir.exists():
        logging.warning(f"Attempted to load index, but storage directory not found: {storage_dir}")
        return None
    logging.info(f"Loading LlamaIndex index from {storage_dir}...")
    t1=time.time()
    try:
        storage_context = StorageContext.from_defaults(persist_dir=str(storage_dir))
        index = load_index_from_storage(storage_context, embed_model=_embed_model)
        print_elapsed_time(t1, "LlamaIndex Index Load")
        return index
    except FileNotFoundError:
        logging.warning(f"Index directory '{storage_dir}' exists, but index files are missing.")
        return None
    except Exception as e:
        logging.error(f"Error loading LlamaIndex from {storage_dir}: {e}", exc_info=True)
        st.error(f"Error loading LlamaIndex index: {e}.", icon="⚠️")
        return None

def load_indexed_files_metadata(metadata_path):
    if metadata_path.exists():
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                logging.info(f"Loaded {len(data)} file paths from metadata: {metadata_path}")
                return set(data)
        except json.JSONDecodeError as e:
            logging.error(f"Error decoding JSON from metadata file {metadata_path}: {e}")
            st.error(f"Error reading metadata file {metadata_path}. File might be corrupted.", icon="⚠️")
        except Exception as e:
            logging.error(f"Error loading metadata file {metadata_path}: {e}", exc_info=True)
            st.error(f"Error loading metadata {metadata_path}: {e}", icon="⚠️")
    else:
        logging.info(f"Metadata file not found: {metadata_path}. Returning empty set.")
    return set()

def save_indexed_files_metadata(metadata_path, processed_files_set):
    try:
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(list(processed_files_set), f, indent=4)
        logging.info(f"Metadata saved to {metadata_path} ({len(processed_files_set)} files).")
    except Exception as e:
        logging.error(f"Error saving metadata to {metadata_path}: {e}", exc_info=True)
        st.error(f"Error saving metadata file {metadata_path}: {e}", icon="🔥")

def sync_index_with_directory_llama(data_dir, index_storage_dir, indexed_files_metadata_path, _embed_model, _node_parser):
    """Scans data_dir, compares to metadata, updates LlamaIndex index."""
    if not data_dir.is_dir():
        logging.error(f"Data directory not found: {data_dir}")
        st.error(f"Data directory not found: {data_dir}")
        return None, load_indexed_files_metadata(indexed_files_metadata_path)

    logging.info(f"Scanning data directory: {data_dir}...")
    t_scan=time.time()
    current_files_set = set(str(p.resolve()) for pattern in ["**/*.txt", "**/*.pdf"] for p in data_dir.glob(pattern))
    scan_duration = print_elapsed_time(t_scan, f"Directory Scan ({len(current_files_set)} files found)")

    processed_files_set = load_indexed_files_metadata(indexed_files_metadata_path)
    logging.info(f"Found {len(processed_files_set)} files in existing metadata.")

    new_files = list(current_files_set - processed_files_set)
    removed_files = list(processed_files_set - current_files_set) # Check for removed files too (optional handling)

    # --- Optional: Handle removed files ---
    if removed_files:
        logging.warning(f"{len(removed_files)} file(s) found in metadata but not in directory. Metadata will be updated, but index nodes remain.")
        final_metadata_set = processed_files_set - set(removed_files)
        save_indexed_files_metadata(indexed_files_metadata_path, final_metadata_set)
        processed_files_set = final_metadata_set # Use updated set for comparison

    index = None
    final_metadata_set = processed_files_set.copy()

    if not new_files and index_storage_dir.exists():
        logging.info("No new files detected. Loading existing index...")
        index = load_index(index_storage_dir, _embed_model)
    elif new_files:
        st.info(f"Found {len(new_files)} new file(s) to index...")
        logging.info(f"New files to index: {[Path(f).name for f in new_files]}")
        t_load = time.time()
        new_documents = []
        try:
            # Load only the new files
            new_docs_reader = SimpleDirectoryReader(input_files=new_files, required_exts=[".pdf", ".txt"])
            new_documents = new_docs_reader.load_data(show_progress=True)
        except Exception as e:
            logging.error(f"Error loading new files: {e}", exc_info=True)
            st.error(f"Error loading new documents: {e}", icon="🔥")

        load_duration = print_elapsed_time(t_load, f"Loading {len(new_files)} New Files")

        if new_documents:
            # Load existing index if it exists, otherwise it stays None
            index = load_index(index_storage_dir, _embed_model)
            t_idx = time.time()
            newly_indexed_paths_current_batch = set()
            try:
                logging.info("Parsing nodes from new documents...")
                # Parse only new documents
                nodes = _node_parser.get_nodes_from_documents(new_documents, show_progress=True)
                logging.info(f"Parsed {len(nodes)} nodes from new documents.")

                if nodes:
                    if index is not None:
                        logging.info(f"Inserting {len(nodes)} new nodes into existing index...")
                        index.insert_nodes(nodes)
                        logging.info(f"Successfully inserted {len(nodes)} nodes.")
                    else:
                        logging.info("Creating new index from parsed nodes...")
                        # Create index only with the new nodes if no index existed
                        index = VectorStoreIndex(nodes, embed_model=_embed_model, show_progress=True)
                        logging.info("New index created.")

                    if index is None:
                        # This should ideally not happen if creation/insertion was attempted
                        raise Exception("Index became None unexpectedly after creation/insertion attempt.")

                    # Get file paths from the metadata of the *actually processed* new nodes
                    newly_indexed_paths_current_batch = {
                        node.metadata.get('file_path')
                        for node in nodes
                        if node.metadata.get('file_path')
                    }

                    logging.info(f"Persisting index to {index_storage_dir}...")
                    index.storage_context.persist(persist_dir=str(index_storage_dir))
                    logging.info("Index persisted successfully.")

                    # Update the final metadata set with successfully indexed files
                    final_metadata_set.update(str(Path(p).resolve()) for p in newly_indexed_paths_current_batch if p)
                    # Save metadata *after* successful persist
                    save_indexed_files_metadata(indexed_files_metadata_path, final_metadata_set)
                else:
                    logging.warning("No nodes were parsed from the new documents (they might be empty or unparseable).")
                    # Ensure index is loaded if it exists, even if no new nodes added
                    if index is None:
                       index = load_index(index_storage_dir, _embed_model)

                print_elapsed_time(t_idx, "LlamaIndex Indexing/Update")

            except Exception as e:
                logging.error(f"Error during index update/creation: {e}", exc_info=True)
                st.error(f"Error during indexing/update: {e}", icon="🔥")
                # Attempt to return the index state before the failed operation
                existing_index_if_any = load_index(index_storage_dir, _embed_model)
                # Return the metadata *before* this failed batch
                return existing_index_if_any, processed_files_set
        else:
            logging.warning("No content could be loaded from the new file(s).")
            # Load existing index if it exists
            index = load_index(index_storage_dir, _embed_model)
    else: # No new files, directory might be empty or index just doesn't exist
        logging.info("No new files detected and/or index storage is empty. Trying to load index if present.")
        index = load_index(index_storage_dir, _embed_model)


    if index is None and not final_metadata_set:
        logging.info("Sync complete. No index exists and no files are tracked in metadata.")
    elif index is None:
           logging.warning("Sync complete. No index could be loaded or created, but metadata exists.")
    else:
           logging.info("Sync complete. Index is loaded/updated.")

    # Save metadata if it changed (covers adding new files, or removing files if handled)
    if final_metadata_set != processed_files_set:
        logging.info("Metadata changed, saving updated set.")
        save_indexed_files_metadata(indexed_files_metadata_path, final_metadata_set)

    return index, final_metadata_set

# --- LlamaIndex Chat Engine Setup ---
def get_chat_engine(_index, _llm, _k=INITIAL_RETRIEVAL_K):
    if _index is None or _llm is None:
        logging.warning("Cannot create chat engine: Index or LLM is None.")
        return None
    logging.info(f"Creating LlamaIndex Chat Engine (Retriever K={_k})...")
    t_chat_setup = time.time()
    try:
        retriever = _index.as_retriever(similarity_top_k=_k)
        node_postprocessors = []
        chat_engine = CondensePlusContextChatEngine.from_defaults(
            retriever=retriever,
            llm=_llm,
            node_postprocessors=node_postprocessors,
            verbose=True,
        )
        print_elapsed_time(t_chat_setup,"Chat Engine Setup")
        return chat_engine
    except Exception as e:
        logging.error(f"Failed to create chat engine: {e}", exc_info=True)
        st.error(f"Error creating chat engine: {e}", icon="🔥")
        return None


# --- Feedback Logging ---
def log_feedback(session_id, message_index, feedback_type, question, answer, sources):
    log_entry = {
        "session_id": session_id,
        "message_index": message_index,
        "feedback": feedback_type,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "question": question,
        "answer": answer,
        "sources": sources
    }
    try:
        with open(FEEDBACK_LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry) + '\n')
        logging.info(f"Feedback logged: Session {session_id}, Msg {message_index}, Type {feedback_type}")
    except Exception as e:
        logging.error(f"Error writing feedback log: {e}", exc_info=True)

# --- Streamlit App UI ---
def main():
    st.set_page_config(page_title="Doc Chat (LlamaIndex)", layout="wide", page_icon="🦙")

    # --- Custom CSS for Fonts and Colors ---
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');
        @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;700;900&display=swap'); /* Added 900 for stronger bold */

        html, body, [class*="st-"] {
            font-family: 'Roboto', sans-serif; /* Primary font for content */
        }
        h1, h2, h3, h4, h5, h6 {
            font-family: 'Montserrat', sans-serif; /* Headings with a different font */
            color: #007bff; /* A fresh blue for headings */
        }
        .st-emotion-cache-1c7y2kl { /* Streamlit container for main content */
            padding-top: 2rem;
        }
        .stButton>button {
            border: 2px solid #007bff; /* Blue border */
            border-radius: 8px; /* Rounded corners */
            color: #007bff; /* Blue text */
            background-color: transparent;
            font-weight: bold;
            padding: 0.6em 1.2em;
            transition: all 0.2s ease-in-out;
        }
        .stButton>button:hover {
            background-color: #007bff; /* Blue background on hover */
            color: white; /* White text on hover */
        }
        .stButton>button:active {
            transform: translateY(1px);
        }

        /* Custom styling for primary buttons like "Sync & Update Index" (targeting Streamlit's internal class) */
        .st-emotion-cache-nahz7x.e1nzilvr4 { /* This class might change with Streamlit updates, verify if needed */
            background-color: #28a745; /* A vibrant green for primary actions */
            color: white;
            border-radius: 8px;
            border: 2px solid #28a745;
            font-weight: bold;
            transition: all 0.2s ease-in-out;
        }
        .st-emotion-cache-nahz7x.e1nzilvr4:hover {
            background-color: #218838;
            border-color: #1e7e34;
        }

        /* Customize chat bubbles (user and assistant) */
        [data-testid="stChatMessage"] {
            background-color: #f0f2f6; /* Light gray background for chat area */
            border-radius: 10px;
            padding: 10px;
            margin-bottom: 10px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05); /* Subtle shadow */
        }
        /* Specific styling for user messages */
        [data-testid="stChatMessage"][data-state="user"] {
            background-color: #e0f2f0; /* Lighter green/blue for user message */
            border-left: 5px solid #28a745; /* Green accent bar */
        }
        /* Specific styling for assistant messages */
        [data-testid="stChatMessage"][data-state="assistant"] {
            background-color: #f0f6ff; /* Lighter blue for assistant message */
            border-left: 5px solid #007bff; /* Blue accent bar */
        }

        /* Adjust input field */
        .stTextInput>div>div>input {
            border-radius: 8px;
            border: 1px solid #ccc;
            padding: 10px;
            box-shadow: inset 0 1px 3px rgba(0,0,0,0.05);
        }

        /* Expander styling */
        [data-testid="stExpander"] { /* Targets the entire expander component */
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            margin-top: 15px;
            margin-bottom: 15px;
            background-color: #fbfbfb;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }
        [data-testid="stExpander"] > div:first-child { /* Targets the header of the expander */
            background-color: #e9ecef; /* Light gray for header */
            border-top-left-radius: 8px;
            border-top-right-radius: 8px;
            padding: 10px;
            font-weight: bold;
            color: #333;
        }
        [data-testid="stExpander"] > div:first-child p { /* Targets the text inside the expander header */
            font-weight: bold;
            color: #333;
        }

        /* Info/Success/Warning/Error boxes - ensuring they have consistent border-radius */
        .stAlert {
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 15px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Main Title - Using markdown for more control and a pop of color
    st.markdown(
        f"""
        <div style='display: flex; flex-direction: column; align-items: center;'>
            <img src='{LOGO_PATH}' alt='Infinite Logo' width='100'>
            <h1 style='text-align: center; color: #007bff; font-size: 3em; margin-bottom: 0.5em; margin-top: 0.5em;'>
                {ASSISTANT_NAME}
            </h1>
            <p style='text-align: center; color: #888; font-size: 0.9em; margin-bottom: 0.2em;'>
                Powered by Infinite Computer Solutions
            </p>
            <p style='text-align: center; color: #555; font-size: 1.1em; margin-bottom: 0.5em;'>
                Meet <strong style='color: #28a745; font-weight: 900;'>{ASSISTANT_NAME}</strong>, your smart companion for quick insights from documents.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # This warning is for debugging purposes. It will show if the logo file path is incorrect.
    # It now uses os.path.exists with the simpler string path.
    if not os.path.exists(LOGO_PATH):
        st.warning(f"Logo file not found at: {LOGO_PATH}. Please ensure it's in the 'static' folder and is a JPG file.", icon="🖼️")


    # Hiding this line as per user request:
    # st.markdown(f"Upload PDF/TXT files to the `{DATA_DIRECTORY.name}` folder (automatically created), then Sync Index.")


    # --- Initialize Resources ---
    embeddings = init_embedding_model(EMBEDDING_MODEL_NAME)
    llm = init_llm(GROQ_API_KEY, GROQ_MODEL_NAME)

    if not embeddings or not llm:
        st.error("Core models (Embeddings or LLM) failed to initialize. Cannot continue.", icon="🛑")
        st.stop()

    # --- Apply Global LlamaIndex Settings ---
    Settings.llm = llm
    Settings.embed_model = embeddings
    if not hasattr(Settings, 'node_parser') or not isinstance(Settings.node_parser, SentenceSplitter):
        Settings.node_parser = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        logging.info(f"Initialized Settings.node_parser (Chunk: {CHUNK_SIZE}, Overlap: {CHUNK_OVERLAP})")


    # --- Session State Initialization ---
    if "messages" not in st.session_state: st.session_state.messages = []
    if "llama_index" not in st.session_state: st.session_state.llama_index = None
    if "processed_files" not in st.session_state: st.session_state.processed_files = set()
    # Corrected: 'not not' to 'not'
    if "chat_engine" not in st.session_state: st.session_state.chat_engine = None
    if "session_id" not in st.session_state: st.session_state.session_id = os.urandom(8).hex()
    if "loaded_initial_index" not in st.session_state: st.session_state.loaded_initial_index = False

    # --- Load existing index/metadata on first proper run ---
    if not st.session_state.loaded_initial_index and embeddings:
        logging.info("Attempting initial load of LlamaIndex index and metadata...")
        st.session_state.llama_index = load_index(LLAMAINDEX_STORAGE_DIR, Settings.embed_model)
        st.session_state.processed_files = load_indexed_files_metadata(INDEXED_FILES_METADATA)
        if st.session_state.llama_index and llm:
            st.session_state.chat_engine = get_chat_engine(st.session_state.llama_index, Settings.llm)
            logging.info("Loaded existing LlamaIndex index and chat engine initialized on startup.")
        elif st.session_state.llama_index:
            logging.warning("Loaded existing index, but LLM wasn't ready for chat engine init.")
        else:
            logging.info("No persistent LlamaIndex found or loaded on startup.")
        st.session_state.loaded_initial_index = True


    # --- Sidebar ---
    with st.sidebar:
        st.header("📚 **Document Management**")
        st.subheader("1. Upload Files")
        st.markdown(f"Files are saved to local folder: **`{DATA_DIRECTORY.name}`**")
        uploaded_files = st.file_uploader("Upload PDF/TXT Files", type=["pdf", "txt"], accept_multiple_files=True, key="file_uploader")

        if uploaded_files:
            save_errors=0
            saved_count=0
            with st.spinner(f"Saving {len(uploaded_files)} file(s) to '{DATA_DIRECTORY.name}'..."):
                DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)
                for uploaded_file in uploaded_files:
                    save_path = DATA_DIRECTORY.joinpath(uploaded_file.name).resolve()
                    try:
                        with open(save_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        saved_count += 1
                        logging.info(f"Successfully saved uploaded file: {save_path}")
                    except Exception as e:
                        save_errors += 1
                        logging.error(f"Error saving uploaded file {uploaded_file.name} to {save_path}: {e}", exc_info=True)
                        st.error(f"Error saving {uploaded_file.name}: {e}", icon="🔥")

            if saved_count > 0:
                st.success(f"Successfully saved {saved_count} file(s).", icon="✅")
                st.info("Click 'Sync & Update Index' below to process new files.", icon="💡")
            if save_errors > 0:
                st.warning(f"Failed to save {save_errors} file(s). Check logs for details.", icon="⚠️")

        st.divider()
        st.subheader("2. Index Documents")
        num_indexed = len(st.session_state.processed_files)
        st.info(f"**{num_indexed}** file(s) currently tracked in the index metadata.", icon="📊")
        if st.session_state.processed_files:
            with st.expander("View Tracked Files"):
                st.json(sorted([Path(f).name for f in st.session_state.processed_files]))

        if st.button("🔄 **Sync & Update Index**", key="sync_button", type="primary",
                     help=f"Check '{DATA_DIRECTORY.name}' for new/removed files & update the index."):
            if not DATA_DIRECTORY.is_dir():
                st.error(f"Data directory missing:\n{DATA_DIRECTORY}")
                logging.error(f"Sync button clicked but data directory missing: {DATA_DIRECTORY}")
            else:
                with st.spinner("Checking directory and updating index... This may take a while for large files or first-time indexing."):
                    t_sync = time.time()
                    updated_index, updated_files_set = sync_index_with_directory_llama(
                        data_dir=DATA_DIRECTORY,
                        index_storage_dir=LLAMAINDEX_STORAGE_DIR,
                        indexed_files_metadata_path=INDEXED_FILES_METADATA,
                        _embed_model=Settings.embed_model,
                        _node_parser=Settings.node_parser
                    )
                    sync_duration = print_elapsed_time(t_sync, "LlamaIndex Sync Operation")

                    st.session_state.processed_files = updated_files_set
                    new_num_indexed = len(updated_files_set)

                    if updated_index is not None:
                        st.session_state.llama_index = updated_index
                        st.session_state.chat_engine = get_chat_engine(st.session_state.llama_index, Settings.llm)
                        if st.session_state.chat_engine:
                            st.success(f"Index synchronized ({sync_duration}s). Chat engine ready.", icon="✅")
                            logging.info(f"Sync successful. Index updated. {new_num_indexed} files tracked.")
                        else:
                            st.warning(f"Index synchronized ({sync_duration}s), but failed to create chat engine. Check logs.", icon="⚠️")
                            logging.warning("Sync completed but chat engine creation failed post-sync.")
                    elif not updated_files_set and not any(DATA_DIRECTORY.glob("**/*.*")):
                        st.warning("Sync complete. Data directory is empty. No index loaded.", icon="ℹ️")
                        st.session_state.llama_index = None
                        st.session_state.chat_engine = None
                        logging.info("Sync complete. Data directory empty, index cleared/not loaded.")
                    else:
                        st.error(f"Index sync completed ({sync_duration}s), but no index is available. Check logs. Files might be empty or unreadable.", icon="🔥")
                        st.session_state.chat_engine = get_chat_engine(st.session_state.llama_index, Settings.llm)
                        logging.error("Sync process finished but resulted in no usable index.")


        st.divider()
        if st.button("⚠️ **Clear All Indexed Data**", type="secondary",
                     help="Deletes the persistent index and metadata files. Does NOT delete uploaded files."):

            @st.dialog("Confirm Deletion")
            def show_confirm_dialog():
                st.warning("⚠️ **Are you sure?**\n\nThis will permanently delete the LlamaIndex storage directory (`persistent_storage/llamaindex_storage`) and the indexed files metadata file (`persistent_storage/indexed_files_llamaindex.json`).\n\nYour original uploaded files in `news_articles` will **NOT** be deleted.\n\nThis action cannot be undone.")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Yes, Delete Index & Metadata", type="primary"):
                        st.info("Attempting to clear indexed data...")
                        delete_success = True
                        log_messages = []
                        console_log_messages = []

                        if LLAMAINDEX_STORAGE_DIR.exists():
                            msg = f"Deleting index directory: `{LLAMAINDEX_STORAGE_DIR}`..."
                            log_messages.append(msg); console_log_messages.append(msg)
                            try:
                                shutil.rmtree(LLAMAINDEX_STORAGE_DIR)
                                console_log_messages.append(f"Successfully deleted directory: {LLAMAINDEX_STORAGE_DIR}")
                                log_messages.append("✅ Index directory deleted.")
                            except Exception as e:
                                console_log_messages.append(f"Error deleting directory {LLAMAINDEX_STORAGE_DIR}: {e}")
                                log_messages.append(f"🔥 Error deleting index directory: {e}"); delete_success = False
                        else:
                            msg = f"Index directory not found, skipping deletion: {LLAMAINDEX_STORAGE_DIR}"
                            log_messages.append(msg); console_log_messages.append(msg)

                        if INDEXED_FILES_METADATA.exists():
                            msg = f"Deleting metadata file: `{INDEXED_FILES_METADATA}`..."
                            log_messages.append(msg); console_log_messages.append(msg)
                            try:
                                os.remove(INDEXED_FILES_METADATA)
                                console_log_messages.append(f"Successfully deleted metadata file: {INDEXED_FILES_METADATA}")
                                log_messages.append("✅ Metadata file deleted.")
                            except Exception as e:
                                console_log_messages.append(f"Error deleting metadata file {INDEXED_FILES_METADATA}: {e}")
                                log_messages.append(f"🔥 Error deleting metadata file: {e}"); delete_success = False
                        else:
                            msg = f"Metadata file not found, skipping deletion: {INDEXED_FILES_METADATA}"
                            log_messages.append(msg); console_log_messages.append(msg)

                        for msg in console_log_messages:
                            if "Error" in msg: logging.error(msg)
                            else: logging.info(msg)

                        if delete_success:
                            st.session_state.llama_index = None
                            st.session_state.processed_files = set()
                            st.session_state.chat_engine = None
                            st.session_state["clear_status"] = ("success", "Indexed data and metadata cleared successfully!", log_messages)
                        else:
                            st.session_state.llama_index = load_index(LLAMAINDEX_STORAGE_DIR, Settings.embed_model)
                            st.session_state.processed_files = load_indexed_files_metadata(INDEXED_FILES_METADATA)
                            st.session_state.chat_engine = get_chat_engine(st.session_state.llama_index, Settings.llm)
                            st.session_state["clear_status"] = ("error", "Could not fully clear indexed data. Check logs.", log_messages)

                        st.rerun()

                with col2:
                    if st.button("Cancel"):
                        st.session_state["clear_status"] = ("info", "Clear data operation cancelled.", [])
                        logging.info("User cancelled data clearing operation.")
                        st.rerun()

            show_confirm_dialog()

        if "clear_status" in st.session_state:
            status_type, message, logs = st.session_state.pop("clear_status")
            if status_type == "success": st.success(message, icon="🗑️")
            elif status_type == "error": st.error(message, icon="⚠️")
            elif status_type == "info": st.info(message, icon="ℹ️")


    # --- Main Chat Area ---
    st.header(f"💬 Chat with **{ASSISTANT_NAME}**")

    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant": # Only try to show sources for assistant messages
                sources_display = message.get("sources", []) # Get processed sources for display

                # Display sources if available
                if sources_display:
                    st.markdown(f"<small style='color: #777;'>Sources: {', '.join(sources_display)}</small>", unsafe_allow_html=True)
                else:
                    # Only show this info if there were no sources for an assistant message
                    # and the message was generated *after* a user prompt (i.e., not the initial welcome message)
                    if i > 0 and st.session_state.messages[i-1]["role"] == "user":
                        st.info("No source documents found for this response.", icon="ℹ️")


            if message["role"] == "assistant" and i > 0 and st.session_state.messages[i-1]["role"] == "user":
                feedback_key_base = f"fb_{st.session_state.session_id}_{i}"
                user_question = st.session_state.messages[i-1]['content']
                cols = st.columns([0.5, 0.5, 8])
                with cols[0]:
                    if st.button("👍", key=f"{feedback_key_base}_up", help="Good answer"):
                        log_feedback(st.session_state.session_id, i, "up", user_question, message['content'], message.get('sources', []))
                        st.toast("Feedback recorded: 👍", icon="✅")
                with cols[1]:
                    if st.button("👎", key=f"{feedback_key_base}_down", help="Bad answer"):
                        log_feedback(st.session_state.session_id, i, "down", user_question, message['content'], message.get('sources', []))
                        st.toast("Feedback recorded: 👎", icon="❌")


    # Accept user input - Updated placeholder
    if prompt := st.chat_input(f"Ask {ASSISTANT_NAME} a question about the indexed documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        if st.session_state.chat_engine is None:
            logging.warning("Chat engine was None. Attempting to re-initialize.")
            if st.session_state.llama_index:
                st.session_state.chat_engine = get_chat_engine(st.session_state.llama_index, Settings.llm)

            if st.session_state.chat_engine is None:
                st.warning("Chat engine is not ready. Please ensure documents are indexed using the 'Sync & Update Index' button in the sidebar.", icon="⚠️")
                logging.warning("Chat input received, but chat engine is still None after check.")
                st.stop()
            else:
                logging.info("Chat engine re-initialized successfully.")

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.chat_engine.chat(prompt)
                    st.markdown(response.response)
                    raw_source_nodes = response.source_nodes # Store the raw nodes for detailed debugging if needed later
                    sources_for_display = [node.metadata.get('file_name', 'Unknown Source') for node in raw_source_nodes if node.metadata.get('file_name')]

                    # Append both the response and the sources/raw_source_nodes to session state
                    st.session_state.messages.append({"role": "assistant",
                                                       "content": response.response,
                                                       "sources": sources_for_display,
                                                       "source_nodes": raw_source_nodes}) # Kept raw nodes in state for potential future detailed display/logging

                    # Display sources immediately after generation (if any)
                    if sources_for_display:
                        st.markdown(f"<small style='color: #777;'>Sources: {', '.join(sources_for_display)}</small>", unsafe_allow_html=True)
                    else:
                        st.info("No source documents found for this response.", icon="ℹ️") # Inform user if no sources found

                except Exception as e:
                    logging.error(f"Error during chat engine response: {e}", exc_info=True)
                    st.error(f"An error occurred while generating the response: {e}", icon="🚫")
                    st.session_state.messages.append({"role": "assistant", "content": f"Sorry, an error occurred: {e}"})

if __name__ == '__main__':
    main()