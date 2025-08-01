from model import get_llama_model
from memory import (
    retrieve_memories, add_to_memory, seed_personal_memory, load_memory, 
    TOP_K_MEMORY, initialize_obsidian_memory, sync_obsidian_memory,
    get_latest_notes, search_notes_by_title
)
from logic import build_prompt
import time
import threading
import re

MAX_CONTEXT = 4096
SYNC_INTERVAL = 300  # Sync every 5 minutes

def background_sync():
    """Background thread to sync Obsidian memory"""
    while True:
        try:
            sync_obsidian_memory()
            time.sleep(SYNC_INTERVAL)
        except Exception as e:
            print(f"Background sync error: {e}")
            time.sleep(60)  # Wait 1 minute before retrying

def handle_special_commands(user_input):
    """Handle special commands for Obsidian integration"""
    user_input_lower = user_input.lower()
    
    # Command: latest notes
    if "latest notes" in user_input_lower or "recent notes" in user_input_lower:
        latest = get_latest_notes(5)
        if latest:
            response = "Your latest notes:\n\n"
            for note in latest:
                response += f"📝 **{note['title']}** (modified: {note['modified']})\n"
                response += f"   {note['preview']}\n\n"
            return response
        else:
            return "No notes found in your Obsidian vault."
    
    # Command: search notes
    search_match = re.search(r'(?:search|find).*notes?.*(?:about|on|titled)\s+["\']?([^"\']+)["\']?', user_input_lower)
    if search_match:
        query = search_match.group(1)
        matching_notes = search_notes_by_title(query)
        if matching_notes:
            response = f"Found {len(matching_notes)} notes matching '{query}':\n\n"
            for note in matching_notes[:3]:  # Show top 3 matches
                response += f"📝 **{note['title']}**\n"
                preview = note['content'][:300] + '...' if len(note['content']) > 300 else note['content']
                response += f"   {preview}\n\n"
            return response
        else:
            return f"No notes found matching '{query}'"
    
    # Command: sync notes
    if "sync" in user_input_lower and "notes" in user_input_lower:
        chunks_added = sync_obsidian_memory()
        return f"✅ Synced Obsidian memory. Added {chunks_added} chunks."
    
    return None

def chat():
    llm = get_llama_model()
    load_memory()
    seed_personal_memory()
    
    print("🔄 Initializing Obsidian integration...")
    initialize_obsidian_memory()
    
    # Start background sync thread
    sync_thread = threading.Thread(target=background_sync, daemon=True)
    sync_thread.start()
    
    conversation_history = []
    print("\n=== Shendu is Back online with Obsidian Integration ===")
    print("Type 'exit' to quit.")
    print("Special commands:")
    print("  - 'latest notes' - Show your most recent notes")
    print("  - 'search notes about [topic]' - Search notes by title")
    print("  - 'sync notes' - Manually sync Obsidian memory")
    print()

    while True:
        user_input = input(" commands: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break

        # Handle special commands
        special_response = handle_special_commands(user_input)
        if special_response:
            print(f"\n🔍 {special_response}")
            continue

        # Regular chat processing
        personal_memories = retrieve_memories(user_input, TOP_K_MEMORY)
        prompt = build_prompt(conversation_history, user_input, personal_memories)

        if len(prompt.split()) > MAX_CONTEXT:
            print("⚠️ Context too long, trimming conversation.")
            while len(prompt.split()) > MAX_CONTEXT and conversation_history:
                conversation_history.pop(0)
                prompt = build_prompt(conversation_history, user_input, personal_memories)

        print(" Shendu (Lemme think...)\n")
        start_time = time.time()

        stream = llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0.2,
            top_p=0.95,
            stream=True
        )

        assistant_reply = ""
        for chunk in stream:
            text = chunk["choices"][0]["delta"].get("content", "")
            print(text, end="", flush=True)
            assistant_reply += text

        print(f"\n\n Completed in {time.time() - start_time:.2f} sec\n")

        conversation_history.append({"role": "user", "content": user_input})
        conversation_history.append({"role": "assistant", "content": assistant_reply})

        # Add new personal info to memory
        if any(phrase in user_input.lower() for phrase in ["my name", "i am", "i like", "i prefer", "remember that", "my projects", "my goals", "my certifications"]):
            add_to_memory(user_input)

if __name__ == "__main__":
    chat()