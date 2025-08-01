from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import psutil
import json
import time
from datetime import datetime
from pathlib import Path
from model import get_llama_model
from memory import (
    retrieve_memories, load_memory, TOP_K_MEMORY, get_latest_notes, 
    search_notes_by_title, sync_obsidian_memory, memory_texts
)
from logic import build_prompt


app = FastAPI(title="Shendu AI API", description="Enhanced AI API with Obsidian Integration")


# Configure CORS - MUST be right after app creation
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://127.0.0.1:8000"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


# Create static directory if it doesn't exist
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)


# Mount static files for CSS, JS, images
app.mount("/static", StaticFiles(directory="static"), name="static")


# Initialize model and memory
llm = get_llama_model()
load_memory()


class ChatRequest(BaseModel):
    history: list
    user_input: str
    use_memory: bool = True


class SystemStats(BaseModel):
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    network_io: Dict[str, int]
    temperature: Optional[float] = None
    timestamp: str


# Enhanced chat endpoint with better error handling
@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    try:
        print(f"Received chat request: {req.user_input}")
        personal_memories = []
        if req.use_memory:
            personal_memories = retrieve_memories(req.user_input, TOP_K_MEMORY)
        prompt = build_prompt(req.history, req.user_input, personal_memories)
        stream = llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0.2,
            top_p=0.95,
            stream=False
        )
        reply = stream["choices"][0]["message"]["content"]
        print(f"Generated reply: {reply[:100]}...")
        return {
            "reply": reply,
            "memories_used": len(personal_memories),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# System stats endpoint
@app.get("/system-stats")
async def get_system_stats():
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        network = psutil.net_io_counters()
        temp = None
        try:
            if hasattr(psutil, "sensors_temperatures"):
                temps = psutil.sensors_temperatures()
                if temps:
                    temp = list(temps.values())[0][0].current
        except:
            pass
        return SystemStats(
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            disk_percent=disk.percent,
            network_io={
                "bytes_sent": network.bytes_sent,
                "bytes_recv": network.bytes_recv
            },
            temperature=temp,
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Obsidian operations
@app.get("/latest-notes")
async def get_latest_notes_endpoint():
    try:
        print("Getting latest notes...")
        notes = get_latest_notes(10)
        print(f"Found {len(notes)} notes")
        return {"notes": notes}
    except Exception as e:
        print(f"Error getting latest notes: {str(e)}")
        return {"notes": [], "error": str(e)}


@app.post("/sync-obsidian")
async def sync_obsidian_endpoint():
    try:
        chunks_added = sync_obsidian_memory()
        return {"chunks_added": chunks_added, "timestamp": datetime.now().isoformat()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Add favicon endpoint to prevent 404 errors
@app.get("/favicon.ico")
async def get_favicon():
    return {"message": "No favicon configured"}


# Replace your entire HTML return statement with this minimal working version
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Shendu AI</title>
    <link href="https://fonts.googleapis.com/css2?family=Cinzel:wght@400;500;600;700&family=Inter:wght@300;400;500;600&display=swap" rel="stylesheet">
    <style>
        :root {
            --shendu-red: #B22222;
            --shendu-dark-red: #8B0000;
            --shendu-green: #228B22;
            --shendu-dark-green: #006400;
            --shendu-lime: #32CD32;
            --shendu-black: #0A0A0A;
            --shendu-dark-gray: #1A1A1A;
            --shendu-gray: #2A2A2A;
            --text-white: #FFFFFF;
            --text-light: #E0E0E0;
            --shadow-glow: 0 0 20px rgba(34, 139, 34, 0.3);
            --text-glow: 0 0 10px rgba(50, 205, 50, 0.5);
        }


        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }


        body {
            font-family: 'Inter', sans-serif;
            background: var(--shendu-black);
            color: var(--text-white);
            min-height: 100vh;
            background-image: 
                linear-gradient(45deg, rgba(178, 34, 34, 0.15) 0%, rgba(26, 26, 26, 0.8) 100%),
                url('file:///C:/Users/Arun/Documents/Shendu/shendu-bg.png');
            background-size: 630px 630px, cover;
            background-position: center right, center;
            background-repeat: no-repeat, repeat;
            background-attachment: fixed;
            overflow-x: hidden;
        }


        .mystical-overlay {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: 
                radial-gradient(circle at 20% 80%, rgba(178, 34, 34, 0.1) 0%, transparent 50%),
                radial-gradient(circle at 80% 20%, rgba(34, 139, 34, 0.1) 0%, transparent 50%);
            pointer-events: none;
            z-index: 0;
        }


        .container {
            position: relative;
            z-index: 1;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            display: grid;
            grid-template-columns: 1fr 350px;
            gap: 30px;
            min-height: 100vh;
        }


        .main-content {
            display: flex;
            flex-direction: column;
            gap: 20px;
        }


        .header {
            background: linear-gradient(135deg, var(--shendu-dark-red) 0%, var(--shendu-red) 100%);
            padding: 25px 30px;
            border-radius: 15px;
            border: 2px solid var(--shendu-green);
            box-shadow: var(--shadow-glow), inset 0 0 30px rgba(34, 139, 34, 0.1);
            text-align: center;
            position: relative;
            overflow: hidden;
        }


        .header::before {
            content: '';
            position: absolute;
            top: -2px;
            left: -2px;
            right: -2px;
            bottom: -2px;
            background: linear-gradient(45deg, var(--shendu-green), var(--shendu-lime), var(--shendu-green));
            border-radius: 15px;
            z-index: -1;
            animation: borderGlow 3s ease-in-out infinite alternate;
        }


        @keyframes borderGlow {
            0% { opacity: 0.7; }
            100% { opacity: 1; }
        }


        .header h1 {
            font-family: 'Cinzel', serif;
            font-size: 2.2em;
            font-weight: 700;
            text-shadow: var(--text-glow);
            margin-bottom: 10px;
            background: linear-gradient(45deg, var(--shendu-green), var(--shendu-lime), var(--shendu-green));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }


        .header p {
            font-size: 1.1em;
            opacity: 0.9;
            color: var(--text-light);
        }


        .chat-container {
            flex: 1;
            background: linear-gradient(135deg, var(--shendu-dark-gray) 0%, var(--shendu-gray) 100%);
            border-radius: 15px;
            border: 1px solid rgba(34, 139, 34, 0.3);
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5), inset 0 0 20px rgba(34, 139, 34, 0.05);
            overflow: hidden;
            display: flex;
            flex-direction: column;
        }


        .messages-area {
            flex: 1;
            padding: 25px;
            overflow-y: auto;
            max-height: 500px;
            scrollbar-width: thin;
            scrollbar-color: var(--shendu-green) var(--shendu-gray);
        }


        .messages-area::-webkit-scrollbar {
            width: 8px;
        }


        .messages-area::-webkit-scrollbar-track {
            background: var(--shendu-gray);
        }


        .messages-area::-webkit-scrollbar-thumb {
            background: var(--shendu-green);
            border-radius: 4px;
        }


        .message {
            margin-bottom: 20px;
            animation: messageSlide 0.4s ease-out;
        }


        @keyframes messageSlide {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }


        .user-message {
            text-align: right;
        }


        .user-message .message-bubble {
            background: linear-gradient(135deg, var(--shendu-red) 0%, var(--shendu-dark-red) 100%);
            border: 1px solid var(--shendu-green);
            margin-left: 20%;
        }


        .assistant-message .message-bubble {
            background: linear-gradient(135deg, var(--shendu-gray) 0%, #333 100%);
            border: 1px solid rgba(34, 139, 34, 0.4);
            border-left: 4px solid var(--shendu-green);
            margin-right: 20%;
        }


        .message-bubble {
            padding: 18px 22px;
            border-radius: 18px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
            word-wrap: break-word;
            line-height: 1.5;
            color: var(--text-white);
        }


        .message-meta {
            font-size: 0.75em;
            opacity: 0.7;
            margin-top: 5px;
            color: var(--shendu-lime);
        }


        .typing-indicator {
            display: flex;
            align-items: center;
            gap: 8px;
            color: var(--shendu-lime);
        }


        .typing-dots {
            display: flex;
            gap: 4px;
        }


        .typing-dot {
            width: 6px;
            height: 6px;
            background: var(--shendu-lime);
            border-radius: 50%;
            animation: typingPulse 1.4s infinite ease-in-out;
        }


        .typing-dot:nth-child(2) { animation-delay: 0.2s; }
        .typing-dot:nth-child(3) { animation-delay: 0.4s; }


        @keyframes typingPulse {
            0%, 60%, 100% { opacity: 0.3; transform: scale(0.8); }
            30% { opacity: 1; transform: scale(1); }
        }


        .input-section {
            padding: 25px;
            background: rgba(26, 26, 26, 0.8);
            border-top: 1px solid rgba(34, 139, 34, 0.2);
        }


        .input-container {
            display: flex;
            gap: 15px;
            align-items: flex-end;
        }


        #messageInput {
            flex: 1;
            padding: 18px 22px;
            background: var(--shendu-dark-gray);
            border: 2px solid rgba(34, 139, 34, 0.3);
            border-radius: 25px;
            color: var(--text-white);
            font-size: 16px;
            font-family: 'Inter', sans-serif;
            resize: none;
            transition: all 0.3s ease;
        }


        #messageInput:focus {
            outline: none;
            border-color: var(--shendu-green);
            box-shadow: 0 0 15px rgba(34, 139, 34, 0.4);
        }


        #messageInput::placeholder {
            color: rgba(224, 224, 224, 0.5);
        }


        #sendButton {
            padding: 18px 30px;
            background: linear-gradient(135deg, var(--shendu-green) 0%, var(--shendu-lime) 100%);
            color: var(--shendu-black);
            border: none;
            border-radius: 25px;
            font-weight: 600;
            font-size: 16px;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(34, 139, 34, 0.3);
            min-width: 120px;
        }


        #sendButton:hover:not(:disabled) {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(34, 139, 34, 0.5);
        }


        #sendButton:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }


        .sidebar {
            display: flex;
            flex-direction: column;
            gap: 20px;
        }


        .sidebar-section {
            background: linear-gradient(135deg, var(--shendu-dark-gray) 0%, var(--shendu-gray) 100%);
            border: 1px solid rgba(34, 139, 34, 0.3);
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.4);
        }


        .sidebar-title {
            font-family: 'Cinzel', serif;
            font-size: 1.3em;
            font-weight: 600;
            color: var(--shendu-green);
            margin-bottom: 20px;
            text-shadow: 0 0 8px rgba(34, 139, 34, 0.4);
            text-align: center;
        }


        .quick-action-btn {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 12px;
            width: 100%;
            padding: 15px 20px;
            margin-bottom: 12px;
            background: linear-gradient(135deg, var(--shendu-red) 0%, var(--shendu-dark-red) 100%);
            color: var(--text-white);
            border: 1px solid rgba(34, 139, 34, 0.4);
            border-radius: 10px;
            font-size: 14px;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.3s ease;
            text-align: center;
        }


        .quick-action-btn:hover {
            background: linear-gradient(135deg, #CD5C5C 0%, var(--shendu-red) 100%);
            border-color: var(--shendu-green);
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(34, 139, 34, 0.3);
        }


        .quick-action-btn:last-child {
            margin-bottom: 0;
        }


        .status-indicator {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 0.9em;
            color: var(--text-light);
        }


        .status-dot {
            width: 8px;
            height: 8px;
            background: var(--shendu-lime);
            border-radius: 50%;
            animation: statusPulse 2s infinite;
        }


        @keyframes statusPulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }


        .memory-counter {
            background: rgba(34, 139, 34, 0.1);
            border: 1px solid rgba(34, 139, 34, 0.3);
            border-radius: 8px;
            padding: 12px;
            text-align: center;
            font-size: 0.9em;
        }


        .memory-number {
            font-size: 1.5em;
            font-weight: 600;
            color: var(--shendu-green);
            display: block;
        }


        @media (max-width: 1024px) {
            .container {
                grid-template-columns: 1fr;
                gap: 20px;
            }
        }


        @media (max-width: 768px) {
            .container {
                padding: 10px;
            }
            
            .header h1 {
                font-size: 1.8em;
            }
            
            .messages-area {
                max-height: 400px;
            }
            
            .user-message .message-bubble {
                margin-left: 10%;
            }
            
            .assistant-message .message-bubble {
                margin-right: 10%;
            }
        }
    </style>
</head>
<body>
    <div class="mystical-overlay"></div>
    
    <div class="container">
        <div class="main-content">
            <div class="header">
                <h1>Shendu AI</h1>
                <p>Personal Assistant</p>
            </div>
            
            <div class="chat-container">
                <div class="messages-area" id="messages">
                    <div class="message assistant-message">
                        <div class="message-bubble">
                            <strong>Shendu:</strong> I am ready to assist you.
                        </div>
                        <div class="message-meta">System initialized</div>
                    </div>
                </div>
                
                <div class="input-section">
                    <div class="input-container">
                        <textarea 
                            id="messageInput" 
                            placeholder="Ask me about your research, notes, or any questions..."
                            rows="1"
                        ></textarea>
                        <button id="sendButton">
                            <span id="sendButtonText">Send</span>
                        </button>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="sidebar">
            <div class="sidebar-section">
                <div class="sidebar-title">Quick Actions</div>
                <button class="quick-action-btn" onclick="sendQuickMessage('show me my latest notes')">
                    Latest Notes
                </button>
                <button class="quick-action-btn" onclick="sendQuickMessage('sync my obsidian notes')">
                    Sync Obsidian
                </button>
                <button class="quick-action-btn" onclick="sendQuickMessage('what do you know about my research?')">
                    Research Overview
                </button>
                <button class="quick-action-btn" onclick="sendQuickMessage('help me organize my thoughts')">
                    Organize Thoughts
                </button>
            </div>
            
            <div class="sidebar-section">
                <div class="sidebar-title">System Status</div>
                <div class="status-indicator">
                    <div class="status-dot"></div>
                    <span>AI Online & Ready</span>
                </div>
                <div class="memory-counter">
                    <span class="memory-number" id="memoryCount">0</span>
                    <span>Memories Accessed</span>
                </div>
            </div>
        </div>
    </div>


    <script>
        const messagesContainer = document.getElementById('messages');
        const messageInput = document.getElementById('messageInput');
        const sendButton = document.getElementById('sendButton');
        const sendButtonText = document.getElementById('sendButtonText');
        const memoryCounter = document.getElementById('memoryCount');
        
        let conversationHistory = [];
        let isProcessing = false;
        let totalMemoriesUsed = 0;


        // Auto-resize textarea
        messageInput.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = Math.min(this.scrollHeight, 150) + 'px';
        });


        function sendMessage(customMessage) {
            if (isProcessing) return;
            
            const message = customMessage || messageInput.value.trim();
            if (!message) return;


            console.log('Shendu processes:', message);
            setProcessingState(true);
            
            addMessage('user', message);
            if (!customMessage) {
                messageInput.value = '';
                messageInput.style.height = 'auto';
            }
            
            const typingId = addTypingIndicator();


            fetch('/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    history: conversationHistory,
                    user_input: message,
                    use_memory: true
                })
            })
            .then(response => {
                if (!response.ok) throw new Error('Network error');
                return response.json();
            })
            .then(data => {
                removeTypingIndicator(typingId);
                addMessage('assistant', `Shendu: ${data.reply}`, data.memories_used);
                
                conversationHistory.push({ role: 'user', content: message });
                conversationHistory.push({ role: 'assistant', content: data.reply });
                
                if (data.memories_used) {
                    totalMemoriesUsed += data.memories_used;
                    updateMemoryCounter();
                }
            })
            .catch(error => {
                console.error('Error:', error);
                removeTypingIndicator(typingId);
                addMessage('assistant', 'Shendu: An error occurred. Please try again.');
            })
            .finally(() => {
                setProcessingState(false);
            });
        }


        function addMessage(sender, content, memoriesUsed) {
            const messageId = 'msg-' + Date.now();
            const messageDiv = document.createElement('div');
            messageDiv.id = messageId;
            messageDiv.className = `message ${sender}-message`;
            
            const bubbleDiv = document.createElement('div');
            bubbleDiv.className = 'message-bubble';
            bubbleDiv.innerHTML = content;
            
            const metaDiv = document.createElement('div');
            metaDiv.className = 'message-meta';
            const timestamp = new Date().toLocaleTimeString();
            metaDiv.textContent = memoriesUsed ? 
                `${timestamp} • ${memoriesUsed} memories used` : timestamp;
            
            messageDiv.appendChild(bubbleDiv);
            messageDiv.appendChild(metaDiv);
            messagesContainer.appendChild(messageDiv);
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
            
            return messageId;
        }


        function addTypingIndicator() {
            const typingId = 'typing-' + Date.now();
            const messageDiv = document.createElement('div');
            messageDiv.id = typingId;
            messageDiv.className = 'message assistant-message';
            
            messageDiv.innerHTML = `
                <div class="message-bubble">
                    <div class="typing-indicator">
                        Shendu is thinking...
                        <div class="typing-dots">
                            <div class="typing-dot"></div>
                            <div class="typing-dot"></div>
                            <div class="typing-dot"></div>
                        </div>
                    </div>
                </div>
            `;
            
            messagesContainer.appendChild(messageDiv);
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
            return typingId;
        }


        function removeTypingIndicator(typingId) {
            const element = document.getElementById(typingId);
            if (element) element.remove();
        }


        function setProcessingState(processing) {
            isProcessing = processing;
            sendButton.disabled = processing;
            sendButtonText.textContent = processing ? 'Processing...' : 'Send';
        }


        function updateMemoryCounter() {
            memoryCounter.textContent = totalMemoriesUsed;
        }


        function sendQuickMessage(message) {
            sendMessage(message);
        }


        // Event listeners
        sendButton.addEventListener('click', () => sendMessage());
        
        messageInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });


        console.log('Shendu AI Interface loaded');
    </script>
</body>
</html>'''




if __name__ == "__main__":
    import uvicorn
    print("Starting Shendu AI Server...")
    print("Access your AI at: http://localhost:8000")
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=False)
