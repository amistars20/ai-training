#!/usr/bin/env python3
"""
WhatsApp chat parser — extracts structured conversation pairs
from an exported _chat.txt file. Handles Android, iOS, and EU formats.
"""
import re
import json
import sys
from pathlib import Path
from datetime import datetime

# ─── patterns ────────────────────────────────────────────────────────────────

# Android:  "1/15/24, 10:30 AM - Name: msg"
ANDROID_PATTERN = re.compile(
    r'^(\d{1,2}/\d{1,2}/\d{2,4}),\s*(\d{1,2}:\d{2}(?::\d{2})?\s*[APap][Mm]?)?\s*-\s*([^:]+):\s*(.*)'
)

# iOS:      "[15/01/2024, 10:30:00] Name: msg"
IOS_PATTERN = re.compile(
    r'^\[(\d{1,2}/\d{1,2}/\d{2,4}),\s*(\d{1,2}:\d{2}(?::\d{2})?\s*[APap][Mm]?)?\]\s*([^:]+):\s*(.*)'
)

# EU no-AM/PM: "15/01/2024, 10:30 - Name: msg"
EU_PATTERN = re.compile(
    r'^(\d{1,2}/\d{1,2}/\d{2,4}),\s*(\d{1,2}:\d{2}(?::\d{2})?)\s*-\s*([^:]+):\s*(.*)'
)

ALL_PATTERNS = {"android": ANDROID_PATTERN, "ios": IOS_PATTERN, "eu": EU_PATTERN}

SYSTEM_STARTS = (
    "Messages and calls are end-to-end encrypted",
    "You created group",
    "changed the subject",
    "changed this group",
    "left",
    "joined using this group",
    "added",
    "removed",
    "security code changed",
    "Missed voice call",
    "Missed video call",
)

MEDIA_OMITTED = ("<Media omitted>", "Image omitted", "Video omitted", "Document omitted", "Audio omitted", "GIF omitted", "Sticker omitted", "image omitted", "video omitted")

# ─── parse ───────────────────────────────────────────────────────────────────

def detect_pattern(lines: list[str]) -> tuple[re.Pattern, str]:
    """Auto-detect which WhatsApp format the file uses."""
    scores = {name: 0 for name in ALL_PATTERNS}
    for line in lines[:50]:
        for name, pat in ALL_PATTERNS.items():
            if pat.match(line):
                scores[name] = scores.get(name, 0) + 1
    best = max(scores, key=scores.get)
    if scores[best] == 0:
        print("Could not detect chat format. Trying Android as default.")
        return ANDROID_PATTERN, "android"
    return ALL_PATTERNS[best], best


def parse_chat(filepath: str | Path) -> list[dict]:
    """Returns list of {sender, text, datetime} in chronological order."""
    raw = Path(filepath).read_text(encoding="utf-8")
    lines = raw.split("\n")

    pattern, fmt_name = detect_pattern(lines)

    messages = []
    current_sender = None
    current_text = None
    current_dt = None

    for line in lines:
        line = line.strip()
        if not line:
            continue

        m = pattern.match(line)
        if m:
            if current_sender and current_text:
                messages.append({
                    "sender": current_sender,
                    "text": current_text.strip(),
                    "datetime": current_dt,
                })
            date_part, time_part, sender, text = m.groups()
            # Normalize date
            try:
                if len(date_part.split("/")[-1]) == 2:
                    dt_str = f"{date_part} {time_part}" if time_part else date_part
                else:
                    dt_str = f"{date_part} {time_part}" if time_part else date_part
            except:
                dt_str = date_part
            current_dt = dt_str
            current_sender = sender.strip()
            current_text = text.strip()
        else:
            if current_text is not None:
                current_text += "\n" + line

    if current_sender and current_text:
        messages.append({
            "sender": current_sender,
            "text": current_text.strip(),
            "datetime": current_dt,
        })

    return messages


def is_system_message(msg: str) -> bool:
    if msg.startswith(SYSTEM_STARTS):
        return True
    for omit in MEDIA_OMITTED:
        if msg == omit:
            return True
    return False


def discover_speakers(messages: list[dict]) -> list[str]:
    """Return sorted unique speakers from parsed messages."""
    seen = set()
    for m in messages:
        s = m["sender"].strip()
        if s:
            seen.add(s)
    return sorted(seen)


def build_pairs(
    messages: list[dict],
    my_name: str,
    friend_name: str,
    min_msg_len: int = 2,
) -> list[dict]:
    """
    Build conversation pairs (my message → friend response).
    Returns list of {"user": str, "assistant": str, "context": list[dict]}
    with optional multi-turn context.
    """
    # Filter system/omitted messages and short ones
    filtered = []
    for m in messages:
        if is_system_message(m["text"]):
            continue
        if len(m["text"].strip()) < min_msg_len:
            continue
        filtered.append(m)

    # Build pairs with context window
    pairs = []
    for i in range(1, len(filtered)):
        prev = filtered[i - 1]
        curr = filtered[i]

        if prev["sender"] == my_name and curr["sender"] == friend_name:
            pairs.append({
                "user": prev["text"],
                "assistant": curr["text"],
            })

    return pairs


def deduplicate(pairs: list[dict]) -> list[dict]:
    """Remove near-identical pairs to avoid overfitting on repeated phrases."""
    seen = set()
    deduped = []
    for p in pairs:
        key = (p["user"][:60].lower(), p["assistant"][:60].lower())
        if key not in seen:
            seen.add(key)
            deduped.append(p)
    return deduped


# ─── main ────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python parse_chat.py <_chat.txt> [--stats-only]")
        sys.exit(1)

    filepath = sys.argv[1]
    stats_only = "--stats-only" in sys.argv

    print(f"Parsing: {filepath}")
    messages = parse_chat(filepath)
    print(f"  Total messages parsed: {len(messages)}")

    speakers = discover_speakers(messages)
    print(f"  Detected speakers: {speakers}")

    if stats_only:
        return

    if len(sys.argv) >= 4:
        my_name = sys.argv[2]
        friend_name = sys.argv[3]
    elif len(speakers) == 2:
        print(f"\nTwo speakers detected: {speakers[0]!r} and {speakers[1]!r}")
        my_name = input("Which one are you? ").strip()
        friend_name = speakers[0] if speakers[1] == my_name else speakers[1]
    else:
        my_name = input("Enter YOUR name as it appears in chat: ").strip()
        friend_name = input("Enter FRIEND's name as it appears in chat: ").strip()

    pairs = build_pairs(messages, my_name, friend_name)
    print(f"  Raw pairs: {len(pairs)}")

    pairs = deduplicate(pairs)
    print(f"  After dedup: {len(pairs)}")

    output_path = Path(filepath).with_suffix(".pairs.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"my_name": my_name, "friend_name": friend_name, "pairs": pairs}, f, indent=1, ensure_ascii=False)

    print(f"  Saved to: {output_path}")

    # Quick stats
    total_user_chars = sum(len(p["user"]) for p in pairs)
    total_asst_chars = sum(len(p["assistant"]) for p in pairs)
    print(f"  Avg user msg length: {total_user_chars // len(pairs)} chars")
    print(f"  Avg friend msg length: {total_asst_chars // len(pairs)} chars")

    # Print samples
    print("\n--- Sample pairs ---")
    for p in pairs[:5]:
        print(f"  YOU: {p['user'][:80]}")
        print(f"  THEM: {p['assistant'][:80]}")
        print()


if __name__ == "__main__":
    main()
