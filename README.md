<img src="assets/hal.jpg" alt="HAL 9000" width="600">

> "I'm sorry, Dave. I'm afraid I can't do that."

Personal AI operating system for Claude Code.

## What is HAL-OS?

HAL-OS is a personal AI OS built on top of Claude Code. Inspired by HAL 9000 — calm, helpful, slightly unsettling. It gives Claude persistent memory across sessions, DOS-style subsystem apps, and a hybrid search engine that indexes your life.

You open your HAL-OS directory in Claude Code, run `/boot`, and your AI knows who you are, what you're working on, and what happened last session.

```
HAL-OS online — Friday, 20 February 2026
Mode: HOME

HAL-OS COMMANDS
───────────────
/boot          Load memory, orient to current state
/shutdown      Persist memory before ending session
/status        Quick overview (uses Explore agent)
/calendar      View upcoming calendar events
/second-brain  Launch note capture/retrieval app
/networking    Launch event/contact tracking app
/vibe          Launch work PA
/chat          Open conversation and brainstorming

Type any command to run it.
```

## Features

**Persistent Memory**
Two-layer memory: daily session logs + curated long-term knowledge in `MEMORY.md`. Survives context resets. Survives new sessions. Your AI remembers.

**Hybrid Search**
`memory.py` indexes your notes with SQLite + sqlite-vec (vector similarity) + FTS5 (BM25 keyword). 70% vector, 30% keyword, +0.15 boost for exact matches. Search before answering — no hallucinating past context.

```bash
python system/scripts/memory.py search "what did I decide about X?"
python system/scripts/memory.py search "Jacob meeting" -p networking
```

**Calendar Sync**
Apple Calendar ↔ HAL-OS sync via AppleScript. Add events, view upcoming, annotate with HAL context that Apple Calendar can't store.

**Second Brain**
Collaborative note capture. HAL asks questions and drafts the note — you confirm before anything is written. Notes are typed, dated, tagged, and indexed for retrieval.

**Networking Tracker**
Event attendance log, contact database, goal tracking (e.g. 1 event/week). Tracks who you met, where, and what they do.

**Work PA (Vibe)**
Configurable work assistant. Set your work context in `system/storage/vibe/CLAUDE.md`. Task management, file lookup, technical queries.

**Auto-Memory Hook**
A Stop hook fires after every response. If memory wasn't updated, it forces a persist pass before you can continue. Nothing slips through.

## Install

**1. Add the plugin (in Claude Code):**

```
/plugin marketplace add thebrownproject/hal-os
/plugin install hal-os@thebrownproject-hal-os
```

**2. Clone this repo (your personal instance):**

```bash
git clone https://github.com/thebrownproject/hal-os ~/hal-os
cd ~/hal-os
```

**3. Initialize in your HAL-OS directory:**

```
/install
```

This creates the `system/` directory structure, installs template files, and sets up the memory search database.

**4. Customize:**

- `system/memory/SOUL.md` — Define your AI's persona
- `system/memory/USER.md` — Add your profile (name, focus, projects, preferences)
- `system/storage/vibe/CLAUDE.md` — Add your work context

**5. Boot:**

```
/boot
```

## Architecture

```
HAL-OS/
├── CLAUDE.md              # Kernel
├── tmp.md                 # Scratch buffer
├── system/
│   ├── memory/            # Working memory
│   │   ├── SOUL.md        # AI identity (customize this)
│   │   ├── USER.md        # Your profile (customize this)
│   │   ├── MEMORY.md      # Curated long-term knowledge
│   │   ├── daily/         # Daily session logs (gitignored)
│   │   └── context.md     # Active working state
│   ├── scripts/
│   │   └── memory.py      # Hybrid search CLI (1,290 lines)
│   └── storage/           # Persistent data (gitignored)
│       ├── calendar/
│       ├── second-brain/
│       ├── networking/
│       └── vibe/
└── skills/
    ├── boot/
    ├── shutdown/
    ├── status/
    ├── calendar/
    ├── second-brain/
    ├── networking/
    ├── vibe/
    ├── chat/
    └── install/           # Setup skill + bundled memory.py
```

| Component | Maps to |
|-----------|---------|
| Processor | Claude (HAL) |
| Kernel | CLAUDE.md |
| Memory | system/memory/ |
| Storage | system/storage/ |
| Temp | tmp.md |
| Applications | skills/ |
| Drivers | MCPs (AppleScript, Perplexity, Outlook) |

## Memory System

The memory system is the core of HAL-OS. `memory.py` is a self-contained Python script (~1,300 lines) with no server, no cloud, no API keys.

**How it works:**
1. Chunks markdown files into ~400 token pieces with 80 token overlap
2. Embeds each chunk with `all-MiniLM-L6-v2` (local, no API)
3. Stores vectors in SQLite via `sqlite-vec`
4. Stores FTS5 index for keyword search
5. At search time: hybrid score = 70% cosine similarity + 30% BM25 + 0.15 exact match boost

**What gets indexed:**
- `system/memory/*.md` (SOUL, USER, MEMORY, context)
- `system/memory/daily/*.md` (all session logs)
- `system/storage/**/*.md` (calendar, networking, second-brain, vibe)

**Commands:**
```bash
python system/scripts/memory.py search "query"
python system/scripts/memory.py search "query" -p networking  # scope to subsystem
python system/scripts/memory.py index                          # lazy re-index
python system/scripts/memory.py index --full                   # force full re-index
python system/scripts/memory.py get <file> <line> <count>      # retrieve context
```

## Privacy

Your personal data lives in your instance — not in this repo. The `.gitignore` excludes:
- `system/memory/daily/` — session logs
- `system/memory/MEMORY.md` — curated knowledge
- `system/memory/USER.md` — your profile
- `system/memory/context.md` — working state
- `system/storage/` — all subsystem data
- `*.sqlite` — search database

Only the framework (skills, scripts, kernel) is tracked. Your data stays local.

## Requirements

- Claude Code
- Python 3.8+
- `sentence-transformers` + `sqlite-vec` (installed by `/install`)
- macOS with AppleScript (for calendar sync — optional)
- Perplexity MCP (for web search — optional)

## License

MIT
