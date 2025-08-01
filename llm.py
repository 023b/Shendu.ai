from llama_cpp import Llama
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
import os
import time
import nltk
nltk.download('punkt')




from nltk import sent_tokenize

# ---------------- Configuration ----------------
MODEL_PATH = r"C:\Users\Arun\Downloads\test\llama.cpp\models\mistral-7b-instruct-v0.1.Q4_K_M.gguf"
MAX_CONTEXT = 4096
N_THREADS = os.cpu_count() // 2 or 2
N_BATCH = 64
N_GPU_LAYERS = 20
TEMPERATURE = 0.2
TOP_P = 0.95
MEMORY_FILE = "memory_store.json"
EMBED_DIM = 384  # Using MiniLM-L6-v2
TOP_K_MEMORY = 8

# ---------------- Load Model ----------------
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=MAX_CONTEXT,
    n_threads=N_THREADS,
    n_batch=N_BATCH,
    n_gpu_layers=N_GPU_LAYERS,
    offload_kqv=True,
    main_gpu=0,
    verbose=True
)

# ---------------- Load Embedder ----------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ---------------- Memory Store ----------------
index = faiss.IndexFlatIP(EMBED_DIM)
memory_texts = []

# Load previous memory
if os.path.exists(MEMORY_FILE):
    with open(MEMORY_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
        memory_texts = data["texts"]
        embeddings = np.array(data["embeddings"]).astype("float32")
        if len(embeddings) > 0:
            index.add(embeddings)

# ---------------- Helper Functions ----------------
def embed_text(text):
    return embedder.encode([text])[0]

def save_memory():
    embeddings = index.reconstruct_n(0, index.ntotal) if index.ntotal > 0 else []
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "texts": memory_texts,
            "embeddings": embeddings.tolist()
        }, f, indent=2)

def add_to_memory(text):
    emb = embed_text(text).astype("float32")
    index.add(np.expand_dims(emb, axis=0))
    memory_texts.append(text)
    save_memory()

def retrieve_memories(query, top_k=TOP_K_MEMORY):
    if index.ntotal == 0:
        return []
    emb = embed_text(query).astype("float32")
    emb = np.expand_dims(emb, axis=0)
    D, I = index.search(emb, top_k)
    retrieved = [memory_texts[i] for i in I[0] if i < len(memory_texts)]
    return retrieved

# ---------------- Pre-seed Personal Memory ----------------
def seed_personal_memory():
    # This is your ultra-detailed LinkedIn breakdown
    large_text = """
        1. Personal & Professional Identity
Name: Arun Prakash S

Pronouns: He/Him

Headline: Recent AI Graduate | Seeking Machine Learning Engineer Roles | Open to R&D

Location: Chennai, Tamil Nadu, India

Contact: Available on profile (LinkedIn/GitHub)

Open To:

Jobs: AI Developer, AI/ML Researcher, Generative AI Engineer.

Volunteering: Education, Science & Technology.

2. Education
Dr. MGR Educational and Research Institute (2022–2025)
Degree: Bachelor of Computer Applications (BCA) with specialization in AI & Data Science.

Key Coursework:

Machine Learning Algorithms, Data Science, NLP, Distributed Systems.

Grade: 2nd Year (status as of profile snapshot).

Certifications (Chronologically Ordered):
IELTS Academic (6.5 Band, July 2025)

Programming in Python (SWAYAM MHRD, May 2024)

Technical Skill Training Program (Tamilnadu Advanced Technical Training Institute, Apr 2024)

AI & ML with Deep Learning (Techobytes, Mar 2024)

Artificial Intelligence A-Z 2023 (Udemy, Jan 2024) – Covered Q-learning, A3C, LLMs.

Data Analytics with AI (Dr. MGR, Nov 2023)

Data Science Excellence (:bitspace, Sep 2023)

Great Learning Certifications (Apr 2023):

AI with Python, Data Mining, Data Science (Python/R).

Pantech.AI Internship (Jun 2022)

Diplomas:

Junior Diploma in DTPP (Adobe Photoshop, 2018).

MS Office (2017).

Extracurricular:

Karate Brown Belt (3rd Grade, 2014).

Spoken Hindi (Vani Vikash Grade 1, 2015).

3. Work Experience
Cognitive Solutions & Technical Content Intern @ IBM (via ProLearn)
Duration: Nov–Dec 2024 (2 months, Hybrid).

Key Contributions:

Researched IBM Watson’s cognitive computing tools.

Created technical blogs/documentation on AI technologies.

AI Development Intern @ Pantech Solutions
Duration: Jan–Apr 2023 (4 months, Remote).

Key Contributions:

Implemented real-time AI/ML models in Python.

Optimized training pipelines and deployment workflows.

4. Technical Skills
Core Competencies:
AI/ML:

Frameworks: PyTorch, TensorFlow, Hugging Face Transformers.

Techniques: Transfer Learning, NLP (BERT, GPT-2), Sentiment Analysis, Computer Vision (OpenCV).

Quantum Computing: Qiskit, Hybrid Quantum-Classical Models, Quantum NLP.

Systems: Distributed Systems (ZeroMQ), BCI (Brain-Computer Interface), Real-time Object Detection.

Languages: Python (Primary), R, JavaScript (Node.js).

Tools: Flask, Docker, Git, IBM Watson, MediaPipe.

Endorsed Skills:
Natural Language Processing (NLP), Machine Learning, Quantum Computing, Distributed Systems, Data Analysis.

5. Projects (Detailed Breakdown)
A. High-Impact AI Projects
Distributed Healthcare Q&A System (Feb–Apr 2025)

Tech Stack: Facebook/OPT-1.3b transformer, ZeroMQ, HTML dashboard.

Achievements:

Scaled across 6 nodes for distributed training/inference.

Real-time monitoring via custom dashboard.

Quantum-Enhanced Language Model (Aug–Oct 2024)

Tech Stack: DistilBERT + Qiskit.

Innovation: Quantum circuits optimized training (85% accuracy on SST-2).

Brain-Computer Interface (BCI) Framework (Sep 2024–Present)

Modules: EEG signal processing, motor imagery classification, synthetic data generation.

Datasets: PhysioNet.

Signature Detection & Verification (Oct 2024)

Model: Fine-tuned VGG16 (PyTorch) for authentication.

B. Computer Vision & Real-Time Systems
Live Webcam Object Classification (Jun–Aug 2024)

Features: Custom-class training via webcam, OpenCV integration.

Hand Gesture Recognition (May–Dec 2022)

Tech: MediaPipe + OpenCV (11 gestures detected).

C. Quantum Computing Research
Quantum-Inspired Neural Network (Mar–Aug 2024)

Results: 92% accuracy (vs. 85% classical) with 20% faster training.

Learning Quantum Computing (May–Jul 2024)

Topics: Grover’s/Shor’s algorithms, QFT, Quantum ML (9-week curriculum).

D. NLP & Utilities
YouTube Comment Sentiment Analysis (Jun–Aug 2023)

Tools: VADER, YouTube API v3.

Personal AI Assistant (Jan–Apr 2023)

Model: Fine-tuned GPT-2 for local deployment (i5/8GB RAM).

6. Publications & Posts
LinkedIn Post (2 weeks ago):

Topic: DIY Home Network (3-tier VLAN, PoE cameras, NAS from pen drives).

Key Takeaways: Systems thinking, cost-redundancy tradeoffs, SPOF analysis.

Hashtags: #AI #HomeLab #DIYTech.

7. GitHub & Open Source
Profile: 229 followers, 225 connections.

Projects Hosted: Likely includes BCI, Quantum ML, and distributed systems code (exact repos not linked).

8. Analytics & Engagement
Profile Views: 77 (past week).

Post Impressions: 114 (past week).

Search Appearances: 22 (past week).

9. Personal Branding
Niche: Bridges theoretical AI with hands-on systems engineering (e.g., quantum computing + DIY networking).

Volunteering: Focus on education/tech accessibility.

10. Recommendations & Network
2nd-Degree Connections: AI/ML peers (e.g., Siva Prakash S, Shabbeer Md).

Skills Endorsed: Python, NLP, Machine Learning.

Key Takeaways
Interdisciplinary Expertise: AI/ML + Quantum Computing + Distributed Systems.

Hands-On Builder: From DIY home labs to production-level AI deployments.

Research-Driven: Focus on cutting-edge areas (BCI, Quantum NLP).

Career Goal: ML Engineer/Researcher in AI R&D or Generative AI.

Weaknesses:

Limited industry experience (only internships).

No explicit mention of publications/conference papers.

Opportunities:

Leverage quantum computing niche for research roles.

Showcase GitHub projects more prominently.
    """
    sentences = sent_tokenize(large_text)
    chunk_size = 5  # adjust for ~200-300 words
    chunks = [' '.join(sentences[i:i+chunk_size]) for i in range(0, len(sentences), chunk_size)]

    count = 0
    for chunk in chunks:
        if chunk not in memory_texts:
            add_to_memory(chunk)
            count += 1
    print(f"✅ Seeded {count} new personal memory chunks.")

# ---------------- Prompt Assembly ----------------
def build_prompt(conversation_history, user_input, personal_memories):
    memory_block = "\n".join(personal_memories)
    conversation_block = "\n".join(
        [f"<{msg['role']}>\n{msg['content']}\n</{msg['role']}>" for msg in conversation_history]
    )
    prompt = (
        f"<system>\nYou are Shendu, Arun Prakash S's personal AI research assistant. "
        f"Always use Arun's personal facts to personalize replies in NLP, NER, NLG, Quantum Computing, AI Research.\n</system>\n\n"
        f"<personal_memory>\n{memory_block}\n</personal_memory>\n\n"
        f"{conversation_block}\n"
        f"<user>\n{user_input}\n</user>\n\n"
        f"<assistant>\n"
    )
    return prompt

# ---------------- Chat Loop ----------------
def chat():
    conversation_history = []
    print("\n=== Shendu is Back online ===")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input(" commands: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break

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
            temperature=TEMPERATURE,
            top_p=TOP_P,
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

        if any(phrase in user_input.lower() for phrase in ["my name", "i am", "i like", "i prefer", "remember that", "my projects", "my goals", "my certifications"]):
            add_to_memory(user_input)

if __name__ == "__main__":
    seed_personal_memory()
    chat()
