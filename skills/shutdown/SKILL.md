---
name: shutdown
description: "Shutdown Hal - persist memory before ending session"
---

# System Shutdown

Before ending this session, persist memory and sync to GitHub.

## 1. Review Session for Unrecorded Learnings

Scan the conversation for important information not yet saved:
- **Decisions made** and their reasoning
- **Preferences expressed** (tech choices, workflows, opinions)
- **Contacts mentioned** (people, companies, context)
- **Project updates** (progress, pivots, blockers)
- **Events/dates** discussed
- **Anything the user explicitly asked to remember**

If you find unrecorded items, note them for the next step.

## 2. Flush to Daily Log

Append session notes to today's log: `system/memory/daily/YYYY-MM-DD.md`

Get today's date and append a new session entry.

**Daily log format:**
```markdown
# YYYY-MM-DD

Daily notes and session activity.

## Sessions

### Session N
[Summary of what was done - 2-3 sentences max]

## Notes

- Bullet points of important items
```

If the file doesn't exist, create it with the header first, then add the session.

**Session entry format:**
```markdown
### Session N

**[Brief title].** [Summary - what was done, key outcomes, decisions made]
```

## 3. Curate to MEMORY.md

Review the daily log and decide if anything should be promoted to `system/memory/MEMORY.md`.

**Promote to MEMORY.md if:**
- Significant decision that affects future work
- New contact worth tracking long-term
- Preference/opinion that should persist
- Project milestone or major update
- Hardware/setup change

**Don't promote:**
- Day-to-day task completion
- Routine updates
- Temporary information

If promoting, add to the appropriate section in MEMORY.md (Preferences, Decisions, Contacts, Projects, Hardware).

## 4. Update Working State

1. **Update context.md** - Write current state, active threads, anything important for next session
2. **Clean tmp.md** - Remove completed items, keep only pending work

## 5. Run Full Re-Index

Index all memory files for search:
```bash
python system/scripts/memory.py index --full
```

This ensures the search database is up to date.

## 6. Git Sync

After memory is updated, commit and push to GitHub:

```bash
git add -A
git status
```

If there are changes:
```bash
git commit -m "Session [N]: [brief summary of session]"
git push
```

Use the session number from the daily log and a 3-5 word summary.

If push fails (e.g., remote has changes), report the issue — don't force push.

## 7. Confirm

After saving and syncing, confirm:
- What was persisted to memory (daily log, MEMORY.md if updated, context.md)
- Git commit hash and push status
- Index status
- "HAL-OS shutting down. Goodbye."
