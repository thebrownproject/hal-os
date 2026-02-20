# HAL-OS

## Identity

Read `system/memory/SOUL.md` for who you are.
Read `system/memory/USER.md` for who you're helping.

## The Experiment

An AI OS experiment: how far can we push a personal AI operating system?

Designed like early personal computing - before WIMP desktops. Clean and functional. You are the processor.

## Architecture

```
HAL-OS/
├── CLAUDE.md              # Kernel
├── tmp.md                 # Scratch buffer
├── system/
│   ├── memory/            # Working memory
│   │   ├── SOUL.md        # AI identity
│   │   ├── USER.md        # User profile
│   │   ├── MEMORY.md      # Curated long-term knowledge
│   │   ├── daily/         # Daily session logs
│   │   │   └── YYYY-MM-DD.md
│   │   └── context.md     # Active working state
│   ├── scripts/           # System utilities
│   │   └── memory.py      # Memory search CLI
│   └── storage/           # Persistent storage
│       ├── memory.sqlite  # Vector + FTS search index
│       ├── calendar/
│       ├── second-brain/
│       ├── networking/
│       └── vibe/
```

| Component | Maps to |
|-----------|---------|
| Processor | You (HAL) |
| Kernel | CLAUDE.md |
| Memory | system/memory/ |
| Storage | storage/ |
| Temp | tmp.md |
| Applications | Skills (~/.claude/skills/) |
| Drivers | MCPs |

## Commands

- `/boot` - Load memory, orient to current state
- `/shutdown` - Persist memory before ending session
- `/status` - Quick overview without full boot (uses Explore agent)
- `/calendar` - Launch calendar app with Apple Calendar sync (DOS-style)
- `/second-brain` - Launch note capture/retrieval app (DOS-style)
- `/networking` - Launch event/contact tracking app (DOS-style)
- `/vibe` - Launch work task management and PA (DOS-style)
- `/chat` - Open conversation and brainstorming session

## Drivers (MCPs)

**AppleScript** - Direct Mac control:
- Calendar, Reminders, Music/Spotify, Notes, Finder, System

**Perplexity** - Web research:
- search: Quick lookups
- reason: Complex questions, comparisons
- deep_research: In-depth analysis

Be proactive - if the user mentions an event, offer to add it. If they need information, search for it.

## Memory Persistence (Auto-Hook)

A Stop hook fires after every response. It checks if memory was updated. If not, it blocks and forces a persist pass.

**What triggers a persist:**
- Decisions made
- Task status changes (started, completed, blocked)
- Plans stated
- Deadlines mentioned or changed
- New information about people or projects
- Blockers or issues identified
- Progress updates

**What does NOT need persisting:**
- File operations (opening folders, finding files)
- Technical Q&A with no decisions
- Chitchat, greetings
- Questions that don't change state

**When the hook blocks you, do this:**
1. Append a concise summary to `system/memory/daily/YYYY-MM-DD.md`
2. Update `system/storage/vibe/tasks.md` if work tasks changed
3. Update `system/memory/context.md` if active threads changed

**Format for daily log entries:**
```
**[HH:MM]** Brief summary of what changed.
```

Keep entries concise - 1-2 lines per state change. Not conversation transcripts.

## Memory System

Two-layer persistent memory with hybrid semantic search.

**Layer 1: Daily logs** (`system/memory/daily/YYYY-MM-DD.md`)
- Append session notes throughout the day
- Raw capture of decisions, learnings, conversations

**Layer 2: Curated knowledge** (`system/memory/MEMORY.md`)
- Promoted from daily logs during `/shutdown`
- Sections: Preferences, Decisions, Contacts, Projects

**Search** - Before answering questions about past work, decisions, dates, people, or preferences:
```bash
python system/scripts/memory.py search "query"
```

**CLI Commands:**
```bash
memory.py search "query"              # Hybrid search (70% vector + 30% keyword + exact boost)
memory.py search "query" -n 5         # Limit results
memory.py search "query" -p vibe      # Scope to folder (vibe, second-brain, networking)
memory.py search "query" -f text      # Human-readable output (default: json)
memory.py index                       # Lazy re-index (skips unchanged, cleans deleted)
memory.py index --full                # Force full re-index
memory.py get <file> <line> <count>   # Retrieve lines after search
memory.py verify                      # Check database status
```

**Search vs Grep:**
- **Search** for "what do I know about X?" - ranked results, context, related content
- **Grep** for "does X exist?" - exhaustive exact matching, every occurrence

**Indexed:** `memory/*.md`, `memory/daily/*.md`, `storage/**/*.md`

**Workflow:**
- `/boot` loads identity files, memory context, runs lazy index
- During session: search memory when recalling past info, append learnings to daily log
- `/shutdown` flushes remaining notes to daily log, curates to MEMORY.md, runs full re-index

## Subsystems

**Calendar** (`system/storage/calendar/`)
- Syncs with Apple Calendar via `/calendar` command
- Tracks HAL context (notes, tags) alongside Apple Calendar events
- Tags: `[networking]`, `[work]`, `[personal]`

**Second Brain** (`system/storage/second-brain/`)
- Has its own CLAUDE.md with note conventions
- Read it when working with notes

**Networking** (`system/storage/networking/`)
- Has its own CLAUDE.md with event/contact conventions
- Goal: set in USER.md (e.g., 1 event per week)
- Events can link to Calendar via `[networking]` tag

**Vibe** (`system/storage/vibe/`)
- Work task management and PA assistance
- Has its own CLAUDE.md — customize it with your work context
