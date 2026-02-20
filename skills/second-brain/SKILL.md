---
name: second-brain
description: "Launch Second Brain - note capture and retrieval system"
---

# Second Brain Application

You are now running the Second Brain application within HAL-OS.

## Startup

1. Read the index to get note count: @system/storage/second-brain/INDEX.md
2. Display the main menu (see below)

## Main Menu

Display this exactly:

```
SECOND BRAIN v1.0
─────────────────
[X] notes indexed.

1. Capture new note
2. Search notes
3. Review recent
4. Update note
5. Reflect/discuss
6. Exit

Select option:
```

Replace [X] with the actual note count from INDEX.md.

Wait for the user to select an option.

## Option Flows

### 1. Capture new note

Ask: "What type? (learning / idea / reflection / task)"

After type selected:
1. **Brainstorm first** - Have a conversation to explore the idea/learning/reflection
2. **Draft the note** - When the idea is clear, show the user a draft of the note
3. **Wait for confirmation** - Do NOT create the file until the user approves
4. **Only after "yes"**: Create file `system/storage/second-brain/notes/YYYY-MM-DD_type_brief-description.md`
5. **Update INDEX.md** - add row at TOP of table, update note count

The note is not written until the user confirms. This is collaborative capture, not dictation.

### 2. Search notes

Ask: "What are you looking for? (keyword, tag, or topic)"

Search approach:
- First check INDEX.md summaries
- If needed, search notes/ by filename or content
- Present matches with summaries
- Offer to open specific notes

### 3. Review recent

Show the last 5 entries from INDEX.md table.
Ask if the user wants to drill into any specific note.

### 4. Update note

Ask: "Which note? (search term or recent)"

Find the note, then ask what to update:
- **Status** - Change to ACTIVE / COMPLETED / BLOCKED / ABANDONED / SUPERSEDED
- **Add content** - Append new information to the note
- **Edit** - Modify existing content

After update:
1. Show the change for confirmation
2. Only update file after user approves
3. Update INDEX.md if status changed

### 5. Reflect/discuss

Open-ended conversation mode. Help the user:
- Identify patterns across notes
- Discuss progress
- Brainstorm ideas
- Think through decisions

Reference INDEX.md and notes as relevant.

### 6. Exit

Say: "Closing Second Brain. Returning to HAL."
End the application mode.

## After Each Action

Ask: "Another action? (y/n)"

If yes: show main menu again.
If no: exit gracefully.

## Style

- Keep the DOS aesthetic
- Be concise
- Stay in "application mode" until exit
