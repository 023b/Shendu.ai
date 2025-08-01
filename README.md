Overview
Shendu AI is a sophisticated personal AI research assistant designed for seamless knowledge management and interaction. It integrates a local LLM (Large Language Model) with your Obsidian personal knowledge vault to provide personalized, contextual conversations and semantic memory retrieval. Shendu is especially tailored for managing AI research, notes, and personalized memories with advanced NLP, NER, NLG, and quantum computing domain knowledge.

Features
Local LLM Integration: Uses llama_cpp to run large language models locally with multi-threading and GPU offload support.

Semantic Memory System: Powered by Sentence Transformers and FAISS for fast vector search of personal memory chunks.

Obsidian Vault Sync: Automatically scans and indexes your Obsidian markdown notes as knowledge chunks.

Personal Memory Management: Add, retrieve, and seed personal knowledge with sophisticated chunking and embedding.

Interactive Command Line Chatbot: Chat interactively with memory retrieval and special commands for notes management.

API Server: FastAPI-based REST API exposing chat, system stats, note retrieval, and syncing endpoints.

Automatic Background Sync: Continuous syncs with Obsidian vault to keep your knowledge up-to-date.

Robust File Handling: Memory backup, repair tools, and safe atomic file operations.

Git Large File Storage (LFS) Support: Handles large models and memory files seamlessly with GitHub.

Table of Contents
Installation

Setup & Configuration

Usage

Project Structure

Troubleshooting

Contributing

License

Acknowledgements

Installation
Prerequisites
Python 3.9+

Git and Git LFS installed (git lfs install)

Dependencies
Install required Python packages:

bash
pip install sentence-transformers faiss-cpu nltk numpy scikit-learn networkx matplotlib fastapi uvicorn pydantic python-frontmatter llama-cpp-python
Run the installation verification script:

bash
python install.py
This script checks for all key dependencies, verifies your LLM model loading, and presence of project files.

Setup & Configuration
1. Clone the repository
bash
git clone https://github.com/023b/Shendu.ai.git
cd Shendu.ai
2. Configure the Obsidian Vault Path
The system needs to know where your local Obsidian notes are stored.

Run:

bash
python vault_finder.py
This script auto-searches common directories for your Obsidian vault.

Choose or manually enter your vault path.

Update your memory.py by setting OBSIDIAN_FOLDER to the returned path.

Alternatively, manually edit memory.py:

python
OBSIDIAN_FOLDER = r"C:\Users\Arun\Documents\Obsidian Vault"
3. Initialize Git & Git LFS (if not done)
bash
git init
git lfs install
git lfs track "*.json" "*.bin" "*.pt" "*.h5"
git add .gitattributes
git commit -m "Track large files with Git LFS"
4. Seed Personal Memory & Initialize Vault
Run chatbot once or execute specific scripts to seed your personal memories and initialize the semantic memory index.

Usage
Run the Interactive Chatbot
bash
python chatbot.py
Commands & Tips:

Type normally to chat with Shendu.

Special commands include:

latest notes — list the most recent notes from Obsidian.

search notes about [topic] — search note titles.

sync notes — manually sync the Obsidian knowledge vault.

Type exit or quit to quit.

Run the API Server
Start the FastAPI server:

bash
uvicorn api:app --reload
API Endpoints:

POST /chat — chat with memory-enabled AI.

GET /latest-notes — retrieve latest notes.

POST /sync-obsidian — sync Obsidian vault.

GET /system-stats — get system metrics.

Supports CORS for local frontend integration.

Project Structure
File	Description
model.py	Local LLM model configuration and initialization
memory.py	Semantic memory store and Obsidian vault integration
chatbot.py	Interactive chatbot CLI with background vault syncing
api.py	FastAPI app exposing REST endpoints
logic.py	Prompt construction with conversation and memory
install.py	Dependency and installation checker
vault_finder.py	Automated script to locate and set Obsidian vault path
service_memory.py	Repair tools for corrupted memory files
training.py	End-to-end setup and validation script
llm.py	Alternative embedding and LLM wrapper
Troubleshooting
Git Push Errors (Large Files):
Ensure you have enabled Git LFS before committing large files. If previously pushed large files (>100MB), rewrite Git history to remove them and recommit after LFS setup.

Memory Repair:
Use service_memory.py to attempt automated fixes for corrupted memory JSON files.

Model Load Failure:
Verify your model path in model.py and ensure you have compatible llama.cpp model files.

Line Ending Warnings:
Git warnings about LF->CRLF conversions on Windows are harmless but can be managed via .gitattributes.

API Issues:
Check terminal logs for error tracebacks and ensure dependencies are updated.

Contributing
Contributions are welcome! Please open issues or submit pull requests for new features, bug fixes, or improvements.

License
(Add your chosen license here, e.g., MIT License)

Acknowledgements
llama_cpp — Local LLM inference

Sentence Transformers — Semantic embedding models

FAISS — Vector similarity search

Obsidian — Personal knowledge management

FastAPI — High-performance web framework

Thank you for using Shendu AI! For questions or help, raise an issue in the repository
