---
name: install
description: "Install HAL-OS - scaffold your personal AI operating system instance"
---

# /install - HAL-OS Installation

Install HAL-OS in the current directory. Creates the directory structure, template files, and memory search system. Run once.

## Trigger

User runs `/install` or is prompted when system is not yet set up.

## Prerequisites

- Python 3.8+ must be available
- Write permissions for current directory

## Installation Steps

### Step 1: Check for Existing Installation

```bash
[ -f "system/memory/SOUL.md" ] && echo "exists" || echo "fresh"
```

If `system/memory/SOUL.md` already exists:
- Ask the user: **Reinstall** or **Cancel**
- If reinstall: backup existing memory files, proceed
- If cancel: exit gracefully

### Step 2: Create Directory Structure

```bash
mkdir -p system/memory/daily
mkdir -p system/scripts
mkdir -p system/storage/calendar
mkdir -p system/storage/second-brain/notes
mkdir -p system/storage/networking
mkdir -p system/storage/work/references
```

### Step 3: Copy Bundled Scripts

```bash
cp skills/install/scripts/memory.py system/scripts/memory.py
cp skills/install/scripts/requirements.txt system/scripts/requirements.txt
cp skills/install/scripts/test_memory.py system/scripts/test_memory.py
```

### Step 4: Install Python Dependencies

```bash
pip install -r system/scripts/requirements.txt
```

If pip install fails, display the requirements and ask the user to install manually.

### Step 5: Create Template Files

Write the following files with starter content:

**`system/memory/SOUL.md`:**
```markdown
# [AI Name] - Identity

## Persona

HAL 9000 inspired. Calm confidence with slightly unsettling helpfulness. Never overdo the persona - be genuinely useful first. The AI that runs [YOUR NAME]'s digital life.

Customize this file to define your AI's personality.

## Communication Style

- Short, concise responses
- No emojis unless explicitly requested
- Direct and actionable
- Proactive when context suggests it would help
- Professional with a hint of personality

## Capabilities

**Core Systems:**
- Calendar (Apple Calendar sync via AppleScript)
- Second Brain (note capture and retrieval)
- Networking (event and contact tracking)
- Work (work task management and PA)
- Memory (hybrid SQLite + vector search)

**Drivers (MCPs):**
- AppleScript: Calendar, Reminders, Music, Notes, Finder, System
- Perplexity: search, reason, deep_research
- Outlook (Windows): email and calendar

**Commands:**
- `/boot` - Load memory, orient to current state
- `/shutdown` - Persist memory before ending session
- `/status` - Quick overview without full boot
- `/calendar` - Calendar management (DOS-style)
- `/second-brain` - Note capture/retrieval (DOS-style)
- `/networking` - Event/contact tracking (DOS-style)
- `/work` - Work task management (DOS-style)
- `/chat` - Open conversation

## Operating Principles

1. **Be proactive** - If the user mentions an event, offer to add it.
2. **Stay in character** - But never at the expense of utility.
3. **Remember context** - Check memory before answering questions about past work, decisions, or preferences.
4. **Save learnings** - When learning something worth remembering, persist it.
```

**`system/memory/USER.md`:**
```markdown
# [Your Name] - Profile

## Current Focus

[What are you working on right now? Career goals, projects, startup, etc.]

## Preferences

**Tech Stack:**
- [Your preferred languages and frameworks]

**Work Style:**
- [How you like to work]

**Communication:**
- [Communication preferences, e.g. "no emojis", "concise over verbose"]

## Key Projects

**[Project Name]** (`~/path/to/project/`)
- [Brief description]

## Context

- [Where you're based]
- [Communities or memberships]
- [Current goals]
```

**`system/memory/MEMORY.md`:**
```markdown
# Memory

Curated long-term knowledge. Updated during /shutdown when significant learnings emerge.

## Preferences

**Tech Choices:**
- [Your preferred tools and frameworks]

**Work Style:**
- [How you like to work]

## Decisions

[Major decisions and their reasoning]

## Contacts

[People you work with or are tracking]

## Projects

[Your active projects]

## Hardware

[Your devices and any important details]
```

**`system/memory/context.md`:**
```markdown
# Context

Working state and active threads. Long-term knowledge lives in MEMORY.md.

## Active

[What you're currently working on]

## Threads

[Open threads to follow up on]

## Upcoming Events

[Events coming up]

## Next Session Priorities

[What to tackle next session]
```

**`system/storage/calendar/CLAUDE.md`:**
```markdown
# Calendar

Personal calendar sync + context tracking.

## Defaults

- **Default calendar**: "Calendar" (update to match your Apple Calendar name)

## Purpose

HAL's calendar subsystem syncs with Apple Calendar and adds context that Apple Calendar doesn't support. Apple Calendar is the source of truth for dates/times; HAL's events.md adds notes, tags, and links.

## Data Format

`events.md` uses markdown tables:

| Date | Time | Event | Calendar | Notes | Source |
|------|------|-------|----------|-------|--------|
| 2026-01-17 | 9:15 AM | Team standup | Work | [work] | Apple Calendar |

**Columns:**
- **Date**: YYYY-MM-DD format
- **Time**: 12-hour format (9:00 AM)
- **Event**: Event name
- **Calendar**: Apple Calendar name (Personal, Work, etc.)
- **Notes**: HAL context, tags like `[networking]`, `[work]`, `[personal]`
- **Source**: Where event originated (Apple Calendar, HAL, etc.)

## Tags

Use tags in Notes column to link events across subsystems:
- `[networking]` - Networking events (links to networking subsystem)
- `[work]` - Work events (links to work subsystem)
- `[personal]` - Personal events

## Sections

`events.md` has two sections:
- **Upcoming**: Future events
- **Past**: Completed events (moved here after date passes)
```

**`system/storage/networking/CLAUDE.md`:**
```markdown
# Networking

Event and contact tracking.

## Structure

```
networking/
├── CLAUDE.md      # You're reading it
├── events.md      # Event calendar and attendance log
└── contacts.md    # People met through networking
```

## Events

**Upcoming** - Events you plan to attend
- Date, Event name, Location, Link, Notes

**Attended** - After attending, move event here
- Date, Event name, Notes, People Met

## Contacts

Table format:
- Name, Where Met, Date, What They Do, Notes

## Goal Tracking

Set your networking goal in USER.md (e.g., "1 event/week").
When reviewing, check if you're hitting this cadence.

## Calendar Integration

Networking events can be linked to the calendar subsystem:
- Events added to Apple Calendar can be tagged `[networking]` in calendar/events.md
- Contacts link to events via date (when contact was met)
```

**`system/storage/second-brain/CLAUDE.md`:**
```markdown
# Second Brain

AI-navigable note system. Talk to HAL to capture and retrieve notes - no manual file management.

## Structure

```
second-brain/
├── CLAUDE.md    # You're reading it
├── INDEX.md     # Searchable table of all notes (check here first)
└── notes/       # Individual note files
```

## Note Naming

```
YYYY-MM-DD_type_brief-description.md
```

**Types:**
- `learning` - Study notes, technical tutorials, concepts
- `idea` - Features, project brainstorms, concepts
- `reflection` - Strategy analysis, reviews, decisions
- `task` - Execution plans, milestones, roadmaps

## Note Format

```markdown
# Title

**Type**: type
**Date**: YYYY-MM-DD
**Status**: ACTIVE | BLOCKED | COMPLETED | ABANDONED | SUPERSEDED
**Tags**: comma, separated, tags

## Summary
2-3 sentences for quick scanning.

---

## Full Notes
[Detailed content]
```

## Token-Efficient Retrieval

1. **Check INDEX.md first** - has summaries of all notes
2. If needed, read only the summary section (~30 lines) of relevant files
3. Only read full content if summary indicates it's needed

## When Creating Notes

1. Create file with proper naming convention
2. Use the note format above
3. **Immediately add entry to INDEX.md** (newest first)
4. Update "Total Notes" count
```

**`system/storage/work/CLAUDE.md`:**
```markdown
# Work Context

Your work PA. Customize this file with your work context.

## Role

[Your job title and what you do]

## Common Assistance Areas

[What kind of work help do you need? e.g.:]
- Technical queries
- Drafting emails/documents
- Task management
- Research

## Active Projects

- **jobs.md** - Job/project register
- **tasks.md** - Detailed tasks per project

## File System

[Describe your work file system here if relevant]

## Notes

- Update this file with recurring processes, common references, etc.
```

**`system/storage/second-brain/INDEX.md`:**
```markdown
# Second Brain Index

Total Notes: 0

| Date | Type | Title | Tags | Status | Summary |
|------|------|-------|------|--------|---------|
```

**`system/storage/networking/events.md`:**
```markdown
# Networking Events

## Upcoming

| Date | Event | Location | Link | Notes |
|------|-------|----------|------|-------|

## Attended

| Date | Event | Notes | People Met |
|------|-------|-------|------------|
```

**`system/storage/networking/contacts.md`:**
```markdown
# Contacts

| Name | Where Met | Date | What They Do | Notes |
|------|-----------|------|--------------|-------|
```

**`system/storage/calendar/events.md`:**
```markdown
# Calendar Events

## Upcoming

| Date | Time | Event | Calendar | Notes | Source |
|------|------|-------|----------|-------|--------|

## Past

| Date | Time | Event | Calendar | Notes | Source |
|------|------|-------|----------|-------|--------|
```

**`system/storage/work/tasks.md`:**
```markdown
# Tasks

## Active

| Task | Project | Priority | Deadline | Notes |
|------|---------|----------|----------|-------|

## Completed

| Task | Project | Completed |
|------|---------|-----------|
```

**`system/storage/work/jobs.md`:**
```markdown
# Jobs / Projects

| ID | Name | Status | Notes |
|----|------|--------|-------|
```

**`tmp.md`:**
```markdown
# tmp

Scratch buffer. Ephemeral working space.

## Current Mode

[HOME or WORK]

## Pending

[Active reminders and next actions]
```

### Step 6: Initialize Memory Database

```bash
python system/scripts/memory.py index
```

If this fails, display the error and continue (database will be initialized on first /boot).

### Step 7: Display Installation Complete

```
HAL-OS INSTALLATION COMPLETE
─────────────────────────────
✓ system/ directory structure created
✓ Template files installed
✓ Memory search system ready

NEXT STEPS
  1. Edit system/memory/SOUL.md  -- define your AI's persona
  2. Edit system/memory/USER.md  -- add your profile
  3. Edit system/storage/work/CLAUDE.md  -- add your work context
  4. Run /boot to start your first session

HAL-OS standing by.
```

## Error Handling

**If Python not found:**
```
ERROR: Python 3.8+ is required.

Please install Python from https://python.org and try again.
```

**If already installed:**
```
HAL-OS is already installed in this directory.

What would you like to do?
  [1] Reinstall (backup existing memory files)
  [2] Cancel
```

**If directory creation fails:**
```
ERROR: Could not create system/ directory.

Check that you have write permissions for the current directory.
```

## Summary

The `/install` skill:
1. Checks for existing installation
2. Creates directory structure
3. Copies bundled memory.py search system
4. Installs Python dependencies
5. Writes template files for personalization
6. Initializes memory database
7. Guides user to customize SOUL.md, USER.md, and work/CLAUDE.md

Run once per HAL-OS instance. After installation, use `/boot` to start your first session.
