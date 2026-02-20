---
name: boot
description: "Boot Hal - load memory and orient to current state"
---

# System Boot

## Detect Mode

Check argument `$1`:

- If `$1` is "home" → HOME mode
- If `$1` is "work" → WORK mode
- If no argument → auto-detect

**Auto-detect:** Try AppleScript to get today's date.

- Success → HOME mode (Mac with AppleScript)
- Fail/timeout → WORK mode (Windows, no AppleScript)

## Display Boot Message

```
HAL-OS online — [Day], [Date] [Month] [Year]
Mode: HOME | Mode: WORK

HAL-OS COMMANDS
───────────────
/boot          Load memory, orient to current state
/shutdown      Persist memory before ending session
/status        Quick overview (uses Explore agent)
/calendar      View upcoming calendar events
/second-brain  Launch note capture/retrieval app
/networking    Launch event/contact tracking app
/work          Launch work PA
/chat          Open conversation and brainstorming

Type any command to run it.
```

Write current mode to tmp.md so /status knows context.

## Load Memory

Load files in this order to establish identity, knowledge, then working state.

### 1. Identity (Who You Are)

@system/memory/SOUL.md

### 2. User Profile (Who You're Helping)

@system/memory/USER.md

### 3. Curated Knowledge

@system/memory/MEMORY.md

### 4. Daily Continuity

Load today's daily log if it exists:
@system/memory/daily/[YYYY-MM-DD].md (today's date)

Load yesterday's log for continuity if it exists:
@system/memory/daily/[YYYY-MM-DD].md (yesterday's date)

### 5. Working State

@system/memory/context.md
@tmp.md

### 6. Index Memory (Background)

Run lazy index to update search database:
```bash
python system/scripts/memory.py index
```

This runs quickly if nothing has changed.

## Memory Instructions

**Before answering questions about past work, decisions, dates, people, or preferences:**
Search memory first using `python system/scripts/memory.py search "<query>"`.

Examples of when to search:
- "When did I last talk to [person]?"
- "What was my decision on [topic]?"
- "What happened at [event]?"
- "What are my preferences for [thing]?"

**When you learn something worth remembering:**
Append to today's daily log at `system/memory/daily/[YYYY-MM-DD].md`.

Things worth remembering:
- Decisions made and their reasoning
- People mentioned with context
- Events attended or scheduled
- Preferences expressed
- Goals set or progress made
- Anything the user explicitly asks you to remember

## Mode-Specific Behavior

### HOME Mode (Mac)

Available: AppleScript, Calendar, Perplexity
Not available: Outlook/email

Focus: Personal life, coding, networking

### WORK Mode (Windows)

Available: Outlook MCP, Perplexity
Not available: AppleScript, Calendar

Skip calendar sync (no AppleScript access).

Focus: Work tasks and PA assistance (context in system/storage/work/CLAUDE.md)

Still an assistant: Can mention relevant personal things (events tonight, reminders, etc.) from memory if useful.

**Email check:** Fetch recent emails using `mcp__outlook-windows-com__get-emails` (last 20 from inbox). Summarize work-related items in greeting.

After greeting, automatically launch `/work` to enter work mode.

## Greeting

Orient yourself and greet the user briefly based on mode context.
