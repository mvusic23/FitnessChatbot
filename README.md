# Fitness Chatbot

A terminal fitness coach powered by [Ollama](https://ollama.com). It runs fully on your machine and can answer general coaching questions. With **RAG** (retrieval-augmented generation), it can also answer from your own gym documents in `data/knowledge/`.

---

## What you need

| Requirement | Notes |
|-------------|--------|
| [Ollama](https://ollama.com) | Must be installed and running |
| Python **3.11+** | Check with `python3 --version` |
| Chat model | e.g. `llama3.2`, `llama3`, `mistral` |
| Embedding model | `bge-m3` for RAG document search |

---

## How to start the project

### 1. Install and run Ollama

1. Download Ollama from [ollama.com](https://ollama.com) and install it.
2. Make sure the Ollama app is running (or start the daemon: `ollama serve`).

Pull the models you plan to use:

```bash
ollama pull llama3.2
ollama pull bge-m3
```

If you already have other models (e.g. `llama3`), you can use those instead—set the name in `.env` (see step 4).

Verify Ollama works:

```bash
ollama list
```

### 2. Clone or open the project

```bash
cd FitnessChatbot
```

### 3. Create a virtual environment and install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows

pip install -e .
```

### 4. Configure environment variables

Copy the example env file and edit it if needed:

```bash
cp .env.example .env
```

Example `.env` if you use `llama3` instead of `llama3.2`:

```env
OLLAMA_HOST=http://127.0.0.1:11434
OLLAMA_MODEL=llama3
EMBED_MODEL=bge-m3
RAG_TOP_K=4
```

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_HOST` | `http://127.0.0.1:11434` | Ollama API URL |
| `OLLAMA_MODEL` | `llama3.2` | Model used for chat |
| `EMBED_MODEL` | `bge-m3` | Model used for document search (RAG) |
| `RAG_TOP_K` | `4` | Number of document chunks per question |

### 5. Start the chatbot

With the virtual environment activated:

```bash
fitness-chat
```

You should see a welcome panel with the active model. If the model is missing, the app prints the exact `ollama pull …` command to run.

---

## How to use it

### Basic chat

1. Type your question at the `You:` prompt and press Enter.
2. The coach streams its reply in real time.
3. Continue the conversation; the bot remembers recent turns until you clear history.

**Example questions:**

- *Suggest a 3-day beginner full-body workout plan.*
- *What should I eat after an evening strength session?*
- *How do I progress from bodyweight squats to barbell squats safely?*

### Slash commands

| Command | What it does |
|---------|----------------|
| `/help` | Show available commands |
| `/clear` | Clear conversation history (start fresh) |
| `/quit` or `/exit` | Leave the chat |

Press **Ctrl+C** during a reply to cancel that response (your last message is not kept in history).

### Using RAG (your gym documents)

RAG lets the coach use **your** files (schedules, rules, membership info, etc.), not only general fitness knowledge.

**Step 1 — Add documents**

Put files in `data/knowledge/`:

- Supported: `.md`, `.txt`, `.pdf`
- A sample file is included: `data/knowledge/gym_info.md`

**Step 2 — Start the chat**

When the chat starts, it automatically reads `data/knowledge/`, creates embeddings, and refreshes the local vector index.
Wait until you see a message like `Indexed N chunks from M file(s).`

**Step 3 — Ask grounded questions**

Examples (if using the sample gym file):

- *What are the gym hours on Saturday?*
- *When is HIIT this week?*
- *What is included in the Plus membership?*

If no index exists yet, the bot still works as a general coach and reminds you to add files to `data/knowledge/` and restart the chat.

### Typical first-time workflow

```text
# Terminal 1 — one-time setup
cd FitnessChatbot
source .venv/bin/activate
cp .env.example .env
# Edit .env if your model name differs
pip install -e .

# Terminal 2 — ensure models exist
ollama pull llama3.2
ollama pull bge-m3

# Start chat
fitness-chat
What time is yoga on Tuesday?
/clear
/quit
```

---

## Project layout

```
FitnessChatbot/
├── src/fitness_chatbot/     # Application code
├── data/knowledge/          # Documents for RAG (you add files here)
├── chroma_db/               # Vector database (created automatically at startup)
├── .env                     # Your local config (not committed)
├── .env.example             # Template for .env
└── pyproject.toml           # Dependencies and fitness-chat entry point
```

---

## Troubleshooting

| Problem | What to do |
|---------|------------|
| `Cannot reach Ollama` | Start Ollama; check `OLLAMA_HOST` in `.env` |
| `Model '…' not found` | Run `ollama pull <model>` or set `OLLAMA_MODEL` to a model from `ollama list` |
| RAG not used | Add files to `data/knowledge/` and restart `fitness-chat` |
| Automatic indexing fails on embeddings | Run `ollama pull bge-m3` |
| `fitness-chat: command not found` | Activate `.venv` and run `pip install -e .` again |

---

## Disclaimer

This bot provides general fitness and nutrition guidance only. It is **not** a substitute for medical advice. Consult a doctor or qualified professional for injuries, pregnancy, chronic illness, or eating disorders.
