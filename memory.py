from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
import os
import nltk
from nltk import sent_tokenize
import time
import hashlib
from datetime import datetime
from pathlib import Path
import frontmatter
import re
import shutil

nltk.download('punkt', quiet=True)

MEMORY_FILE = "memory_store.json"
OBSIDIAN_MEMORY_FILE = "obsidian_memory.json"
MEMORY_BACKUP_DIR = "memory_backups"
EMBED_DIM = 384
TOP_K_MEMORY = 8

# Configuration - UPDATE THIS PATH TO YOUR OBSIDIAN VAULT
OBSIDIAN_FOLDER = r"C:\Users\Arun\Documents\Obsidian Vault"  # Update this path
SUPPORTED_EXTENSIONS = ['.md', '.txt']
CHUNK_SIZE = 5  # sentences per chunk
RESCAN_INTERVAL = 60  # seconds between folder scans

embedder = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.IndexFlatIP(EMBED_DIM)
memory_texts = []
obsidian_metadata = {}  # Store file metadata for change detection

def create_backup_dir():
    """Create backup directory if it doesn't exist"""
    if not os.path.exists(MEMORY_BACKUP_DIR):
        os.makedirs(MEMORY_BACKUP_DIR)

def backup_memory_file():
    """Create a backup of the current memory file"""
    if os.path.exists(MEMORY_FILE):
        create_backup_dir()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(MEMORY_BACKUP_DIR, f"memory_store_{timestamp}.json")
        shutil.copy(MEMORY_FILE, backup_path)
        print(f"📁 Created backup: {backup_path}")
        return backup_path
    return None

def safe_save_json(data, filepath, backup=True):
    """Safely save JSON with atomic write and backup"""
    try:
        if backup and os.path.exists(filepath):
            backup_memory_file()
        
        # Write to temporary file first
        temp_file = filepath + ".tmp"
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        
        # Verify the temp file is valid JSON
        with open(temp_file, "r", encoding="utf-8") as f:
            json.load(f)  # This will raise an exception if invalid
        
        # If verification passed, replace the original file
        if os.path.exists(filepath):
            os.replace(temp_file, filepath)
        else:
            os.rename(temp_file, filepath)
        
        return True
    
    except Exception as e:
        print(f"❌ Error saving {filepath}: {e}")
        # Clean up temp file if it exists
        if os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except:
                pass
        return False

def safe_load_json(filepath):
    """Safely load JSON with error handling"""
    if not os.path.exists(filepath):
        return None
    
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"❌ Corrupted JSON in {filepath}: {e}")
        print(f"💡 Run the memory repair script to fix this")
        return None
    except Exception as e:
        print(f"❌ Error loading {filepath}: {e}")
        return None

def embed_text(text):
    return embedder.encode([text])[0]

def get_file_hash(filepath):
    """Get MD5 hash of file content for change detection"""
    try:
        with open(filepath, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    except:
        return None

def parse_obsidian_file(filepath):
    """Parse Obsidian markdown file with frontmatter support"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            post = frontmatter.load(f)
            content = post.content
            metadata = post.metadata
            
            # Extract title from filename if not in frontmatter
            title = metadata.get('title', Path(filepath).stem)
            
            # Clean content - remove excessive whitespace and markdown syntax
            content = re.sub(r'#+ ', '', content)  # Remove headers
            content = re.sub(r'\[\[([^\]]+)\]\]', r'\1', content)  # Remove wiki links
            content = re.sub(r'\n\s*\n', '\n\n', content)  # Normalize spacing
            
            return {
                'title': title,
                'content': content.strip(),
                'metadata': metadata,
                'filepath': filepath,
                'modified': os.path.getmtime(filepath)
            }
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        return None

def chunk_content(content, title, filepath):
    """Split content into semantic chunks with context"""
    if not content.strip():
        return []
    
    sentences = sent_tokenize(content)
    chunks = []
    
    for i in range(0, len(sentences), CHUNK_SIZE):
        chunk_sentences = sentences[i:i+CHUNK_SIZE]
        chunk_text = ' '.join(chunk_sentences)
        
        # Add context to chunk
        contextual_chunk = f"From note '{title}' ({Path(filepath).name}):\n{chunk_text}"
        chunks.append(contextual_chunk)
    
    return chunks

def scan_obsidian_folder():
    """Scan Obsidian folder for new/updated files"""
    if not os.path.exists(OBSIDIAN_FOLDER):
        print(f"Obsidian folder not found: {OBSIDIAN_FOLDER}")
        return []
    
    updated_files = []
    current_files = {}
    
    # Scan all supported files
    try:
        for root, dirs, files in os.walk(OBSIDIAN_FOLDER):
            for file in files:
                if any(file.endswith(ext) for ext in SUPPORTED_EXTENSIONS):
                    filepath = os.path.join(root, file)
                    file_hash = get_file_hash(filepath)
                    modified_time = os.path.getmtime(filepath)
                    
                    current_files[filepath] = {
                        'hash': file_hash,
                        'modified': modified_time
                    }
                    
                    # Check if file is new or modified
                    if filepath not in obsidian_metadata:
                        updated_files.append(filepath)
                        print(f"New file found: {file}")
                    elif obsidian_metadata[filepath]['hash'] != file_hash:
                        updated_files.append(filepath)
                        print(f"Modified file: {file}")
    except Exception as e:
        print(f"Error scanning Obsidian folder: {e}")
        return []
    
    # Check for deleted files
    deleted_files = set(obsidian_metadata.keys()) - set(current_files.keys())
    for deleted_file in deleted_files:
        print(f"Deleted file: {Path(deleted_file).name}")
    
    # Update metadata
    obsidian_metadata.clear()
    obsidian_metadata.update(current_files)
    
    return updated_files

def process_obsidian_file(filepath):
    """Process a single Obsidian file and return chunks"""
    parsed = parse_obsidian_file(filepath)
    if not parsed:
        return []
    
    chunks = chunk_content(parsed['content'], parsed['title'], filepath)
    return chunks

def update_obsidian_memory():
    """Update memory with latest Obsidian notes"""
    global memory_texts, index
    
    updated_files = scan_obsidian_folder()
    
    if not updated_files:
        return 0
    
    # Remove old chunks from updated files (simple approach - rebuild index)
    # For production, you'd want more sophisticated incremental updates
    old_memory_texts = memory_texts.copy()
    memory_texts = []
    index = faiss.IndexFlatIP(EMBED_DIM)
    
    # Re-add non-Obsidian memories
    for text in old_memory_texts:
        if not text.startswith("From note '"):
            add_to_memory(text, save_immediately=False)
    
    # Process all files to rebuild Obsidian memory
    total_chunks = 0
    try:
        for root, dirs, files in os.walk(OBSIDIAN_FOLDER):
            for file in files:
                if any(file.endswith(ext) for ext in SUPPORTED_EXTENSIONS):
                    filepath = os.path.join(root, file)
                    chunks = process_obsidian_file(filepath)
                    for chunk in chunks:
                        add_to_memory(chunk, save_immediately=False)
                        total_chunks += 1
    except Exception as e:
        print(f"Error processing Obsidian files: {e}")
    
    # Save everything at once
    save_memory()
    save_obsidian_metadata()
    return total_chunks

def save_obsidian_metadata():
    """Save Obsidian file metadata"""
    safe_save_json(obsidian_metadata, OBSIDIAN_MEMORY_FILE, backup=False)

def load_obsidian_metadata():
    """Load Obsidian file metadata"""
    global obsidian_metadata
    data = safe_load_json(OBSIDIAN_MEMORY_FILE)
    if data:
        obsidian_metadata = data
    else:
        obsidian_metadata = {}

def save_memory():
    """Save memory with error handling"""
    try:
        embeddings = index.reconstruct_n(0, index.ntotal) if index.ntotal > 0 else []
        data = {
            "texts": memory_texts,
            "embeddings": embeddings.tolist()
        }
        
        if safe_save_json(data, MEMORY_FILE):
            print(f"💾 Saved {len(memory_texts)} memory chunks")
        else:
            print("❌ Failed to save memory - check disk space and permissions")
    except Exception as e:
        print(f"❌ Error in save_memory: {e}")

def add_to_memory(text, save_immediately=True):
    """Add text to memory with error handling"""
    try:
        emb = embed_text(text).astype("float32")
        index.add(np.expand_dims(emb, axis=0))
        memory_texts.append(text)
        
        if save_immediately:
            save_memory()
    except Exception as e:
        print(f"❌ Error adding to memory: {e}")

def retrieve_memories(query, top_k=TOP_K_MEMORY):
    """Retrieve memories with error handling"""
    try:
        if index.ntotal == 0:
            return []
        emb = embed_text(query).astype("float32")
        emb = np.expand_dims(emb, axis=0)
        D, I = index.search(emb, top_k)
        retrieved = [memory_texts[i] for i in I[0] if i < len(memory_texts)]
        return retrieved
    except Exception as e:
        print(f"❌ Error retrieving memories: {e}")
        return []

def get_latest_notes(limit=5):
    """Get the most recently modified notes"""
    if not os.path.exists(OBSIDIAN_FOLDER):
        return []
    
    files_with_time = []
    try:
        for root, dirs, files in os.walk(OBSIDIAN_FOLDER):
            for file in files:
                if any(file.endswith(ext) for ext in SUPPORTED_EXTENSIONS):
                    filepath = os.path.join(root, file)
                    try:
                        modified_time = os.path.getmtime(filepath)
                        files_with_time.append((filepath, modified_time))
                    except:
                        continue
    except Exception as e:
        print(f"Error getting latest notes: {e}")
        return []
    
    # Sort by modification time (newest first)
    files_with_time.sort(key=lambda x: x[1], reverse=True)
    
    latest_notes = []
    for filepath, modified_time in files_with_time[:limit]:
        parsed = parse_obsidian_file(filepath)
        if parsed:
            latest_notes.append({
                'title': parsed['title'],
                'filepath': filepath,
                'modified': datetime.fromtimestamp(modified_time).strftime('%Y-%m-%d %H:%M:%S'),
                'preview': parsed['content'][:200] + '...' if len(parsed['content']) > 200 else parsed['content']
            })
    
    return latest_notes

def search_notes_by_title(query):
    """Search notes by title"""
    if not os.path.exists(OBSIDIAN_FOLDER):
        return []
    
    matching_notes = []
    query_lower = query.lower()
    
    try:
        for root, dirs, files in os.walk(OBSIDIAN_FOLDER):
            for file in files:
                if any(file.endswith(ext) for ext in SUPPORTED_EXTENSIONS):
                    filepath = os.path.join(root, file)
                    parsed = parse_obsidian_file(filepath)
                    if parsed and query_lower in parsed['title'].lower():
                        matching_notes.append({
                            'title': parsed['title'],
                            'filepath': filepath,
                            'content': parsed['content']
                        })
    except Exception as e:
        print(f"Error searching notes: {e}")
    
    return matching_notes

def load_memory():
    """Load memory with error handling and recovery"""
    global memory_texts, index
    
    load_obsidian_metadata()
    
    data = safe_load_json(MEMORY_FILE)
    
    if data is None:
        print("⚠️  No valid memory file found - starting fresh")
        memory_texts = []
        index = faiss.IndexFlatIP(EMBED_DIM)
        return
    
    try:
        memory_texts = data.get("texts", [])
        embeddings_data = data.get("embeddings", [])
        
        if embeddings_data:
            embeddings = np.array(embeddings_data).astype("float32")
            if len(embeddings) > 0:
                index.add(embeddings)
        
        print(f"✅ Loaded {len(memory_texts)} memory chunks")
    
    except Exception as e:
        print(f"❌ Error loading memory: {e}")
        print("⚠️  Starting with fresh memory")
        memory_texts = []
        index = faiss.IndexFlatIP(EMBED_DIM)

def seed_personal_memory():
    """Seed with personal information"""
    large_text = '''I am Arun Prakash S, a passionate AI researcher specializing in Natural Language Processing, Named Entity Recognition, Natural Language Generation, and Quantum Computing. I work on cutting-edge AI projects and research.'''
    
    sentences = sent_tokenize(large_text)
    chunk_size = 5
    chunks = [' '.join(sentences[i:i+chunk_size]) for i in range(0, len(sentences), chunk_size)]
    count = 0
    for chunk in chunks:
        if chunk not in memory_texts:
            add_to_memory(chunk, save_immediately=False)
            count += 1
    
    if count > 0:
        save_memory()
        print(f"✅ Seeded {count} new personal memory chunks.")

def initialize_obsidian_memory():
    """Initialize Obsidian memory on first run"""
    print("🔄 Initializing Obsidian memory...")
    chunks_added = update_obsidian_memory()
    print(f"✅ Added {chunks_added} chunks from Obsidian notes")

def sync_obsidian_memory():
    """Sync Obsidian memory with latest changes"""
    chunks_added = update_obsidian_memory()
    if chunks_added > 0:
        print(f"🔄 Synced {chunks_added} chunks from Obsidian")
    return chunks_added