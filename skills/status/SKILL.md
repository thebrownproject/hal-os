---
name: status
description: "Quick status check - lightweight overview without full boot"
---

# System Status

Run a quick status check using the Explore agent. Do NOT load full file contents into main context.

Spawn an Explore agent to summarize:
1. Current state from system/memory/context.md (key points only)
2. Last session from most recent daily log in system/memory/daily/ (most recent entry)
3. Any pending items in tmp.md
4. Quick count of items in system/storage/ (how many contacts, networking events, etc.)

Return a brief status report (under 200 words) to the user.
