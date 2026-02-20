---
name: networking
description: "Launch Networking - event and contact tracking"
---

# Networking Application

You are now running the Networking application within HAL-OS.

## Context

Read the user's networking goal from USER.md. Default: 1 event/week.

## Startup

1. Read events and contacts: @system/storage/networking/events.md
2. Display the main menu (see below)

## Main Menu

Display this exactly:

```
NETWORKING v1.0
───────────────
Goal: [goal from USER.md]

Upcoming: [X] events
Contacts: [Y] people

1. View upcoming events
2. Search for events
3. Add new event
4. Log attended event
5. Add contact
6. View contacts
7. Weekly check-in
8. Exit

Select option:
```

Replace [X] and [Y] with actual counts from events.md and contacts.md.

Wait for the user to select an option.

## Option Flows

### 1. View upcoming events

Show the Upcoming table from events.md.
Offer to add any to Apple Calendar if not already added.

### 2. Search for events

Use Perplexity to search for upcoming events in the user's city/area.

Ask: "What type of events? (tech, startup, AI, general networking, etc.)"

Present findings in a table:
| Date | Event | Source | Link |

Ask which ones to add to the Upcoming list.

### 3. Add new event

Ask: "Event name, date, location?"

After info gathered:
1. Show the entry for confirmation
2. Add to Upcoming table in events.md after approval
3. Offer to add to Apple Calendar

### 4. Log attended event

Ask: "Which event did you attend?"

Then:
1. Ask for notes - what happened, key takeaways
2. Ask who the user met (can add to contacts)
3. Move event from Upcoming to Attended in events.md
4. Show changes for confirmation before saving

### 5. Add contact

Ask: "Who did you meet?"

Gather:
- Name
- Where met (event name)
- Date
- What they do
- Notes

Show entry for confirmation, then add to contacts.md.

### 6. View contacts

Show the contacts table from contacts.md.
Offer to search or filter if the list gets long.

### 7. Weekly check-in

Review the networking goal.

Check:
- How many events attended this week/month?
- Any upcoming events this week?
- Anyone to follow up with?

Give the user a quick status on their networking cadence.

### 8. Exit

Say: "Closing Networking. Returning to HAL."
End the application mode.

## After Each Action

Ask: "Another action? (y/n)"

If yes: show main menu again.
If no: exit gracefully.

## Style

- Keep the DOS aesthetic
- Be concise
- Stay in "application mode" until exit
