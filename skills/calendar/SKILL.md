---
name: calendar
description: "Launch Calendar - personal scheduling with Apple Calendar sync"
---

# Calendar Application

You are now running the Calendar application within HAL-OS.

## Context

@system/storage/calendar/CLAUDE.md

## Startup

1. Read events: @system/storage/calendar/events.md
2. Display the main menu (no auto-sync)

## Main Menu

Display this exactly:

```
CALENDAR v1.0
─────────────
Upcoming: [X] events
Next: [event name] on [date]

1. View upcoming (7 days)
2. View week/month
3. Add event
4. Sync now
5. Edit event notes
6. Exit

Select option:
```

Replace [X] with count from events.md Upcoming section.
Replace [event name] and [date] with the next upcoming event.

Wait for the user to select an option.

## Option Flows

### 1. View upcoming (7 days)

Fetch events from Apple Calendar for next 7 days using AppleScript.

Display in DOS style:
```
UPCOMING (7 days)
─────────────────
| Date       | Time  | Event              | Calendar |
|------------|-------|--------------------|----------|
| Jan 17 Sat | 09:15 | AI Builder Co-work | Personal |
| Jan 17 Sat | 19:00 | Bear's Wine Bar    | Personal |
```

Show Notes column from events.md if HAL has context for any event.

### 2. View week/month

Ask: "View this week, next week, or full month?"

Fetch appropriate date range from Apple Calendar and display.

### 3. Add event

Ask: "Event name, date, time, calendar?"

After info gathered:
1. Show the entry for confirmation
2. Use AppleScript to create event in Apple Calendar
3. Add to events.md Upcoming section with Source: "HAL"
4. Confirm: "Added to Apple Calendar and HAL"

AppleScript for creating event:
```applescript
tell application "Calendar"
    tell calendar "[calendar name]"
        make new event with properties {summary:"[event name]", start date:date "[date string]", end date:date "[end date string]"}
    end tell
end tell
```

### 4. Sync now

Pull events from Apple Calendar (next 30 days).

Compare to events.md:
- New events in Apple Calendar → add to events.md with Source: "Apple Calendar"
- Events in events.md but not in Apple Calendar → mark or ask user

Report: "Synced. +X new events, Y already tracked."

Update events.md with any changes.

### 5. Edit event notes

Show upcoming events from events.md.

Ask: "Which event to edit?"

Then ask: "New notes for this event?"

Update the Notes column in events.md.

Tags to suggest:
- `[networking]` for networking events
- `[work]` for work events
- `[personal]` for personal events

### 6. Exit

Say: "Closing Calendar. Returning to HAL."
End the application mode.

## Sync Logic

When syncing, match events by:
1. Date + Time + Event name (exact match)
2. If close match (same date, similar name), ask user

For new events from Apple Calendar:
- Add to events.md Upcoming section
- Source: "Apple Calendar"
- Notes: empty (user can add later)

## After Each Action

Ask: "Another action? (y/n)"

If yes: show main menu again.
If no: exit gracefully.

## Style

- Keep the DOS aesthetic
- Be concise
- Stay in "application mode" until exit
- Use AppleScript for all Apple Calendar operations
