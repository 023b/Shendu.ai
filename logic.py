def build_prompt(conversation_history, user_input, personal_memories):
    # Separate Obsidian notes from regular memories
    obsidian_memories = []
    regular_memories = []
    
    for memory in personal_memories:
        if memory.startswith("From note '"):
            obsidian_memories.append(memory)
        else:
            regular_memories.append(memory)
    
    # Build memory blocks
    regular_memory_block = "\n".join(regular_memories) if regular_memories else ""
    obsidian_memory_block = "\n".join(obsidian_memories) if obsidian_memories else ""
    
    # Build conversation block
    conversation_block = "\n".join(
        [f"<{msg['role']}>\n{msg['content']}\n</{msg['role']}>" for msg in conversation_history]
    )
    
    # Enhanced system prompt
    system_prompt = (
        "You are Shendu, Arun Prakash S's personal AI research assistant. "
        "You have access to Arun's personal information, research notes, and Obsidian knowledge vault. "
        "Always use this information to provide personalized, contextual responses. "
        "Focus on NLP, NER, NLG, Quantum Computing, and AI Research. "
        "When referencing information from Obsidian notes, mention the note title for context."
    )
    
    # Build the complete prompt
    prompt_parts = [
        f"<system>\n{system_prompt}\n</system>\n"
    ]
    
    # Add regular personal memories
    if regular_memory_block:
        prompt_parts.append(f"<personal_memory>\n{regular_memory_block}\n</personal_memory>\n")
    
    # Add Obsidian knowledge
    if obsidian_memory_block:
        prompt_parts.append(f"<knowledge_vault>\n{obsidian_memory_block}\n</knowledge_vault>\n")
    
    # Add conversation history
    if conversation_block:
        prompt_parts.append(f"{conversation_block}\n")
    
    # Add current user input
    prompt_parts.append(f"<user>\n{user_input}\n</user>\n\n<assistant>\n")
    
    return "".join(prompt_parts)