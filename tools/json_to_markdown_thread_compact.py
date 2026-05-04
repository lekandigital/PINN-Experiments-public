import json
from pathlib import Path
from datetime import datetime

SRC = Path('/Users/lekan/Downloads/project02_thread_extract.json')
OUT = Path('/Users/lekan/Downloads/project02_thread_compact.md')


def extract_content_blocks(content_list):
    blocks = []
    for c in content_list or []:
        if not isinstance(c, dict):
            continue
        ctype = c.get('type', 'unknown')
        text = c.get('text')
        if text is None:
            text = c.get('input_text')
        if text is None:
            text = c.get('output_text')
        if text is None:
            continue
        blocks.append((ctype, str(text)))
    return blocks


def main() -> None:
    data = json.loads(SRC.read_text(encoding='utf-8'))
    entries = data.get('entries', [])

    lines = []
    lines.append('# Project 02 Thread Extract (Compact)')
    lines.append('')
    lines.append(f"- Source JSONL: {data.get('source_file', '')}")
    lines.append(f"- Line range: {data.get('start_line_in_jsonl', '?')} to {data.get('end_line_in_jsonl', '?')}")
    lines.append(f"- Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append('')

    kept = 0
    for entry in entries:
        ts = entry.get('timestamp', 'n/a')
        payload = entry.get('payload')
        if not isinstance(payload, dict):
            continue

        ptype = payload.get('type')

        # Standard chat message entries
        if ptype == 'message':
            role = payload.get('role', 'unknown')
            blocks = extract_content_blocks(payload.get('content', []))
            if not blocks:
                continue
            kept += 1
            lines.append(f"## {role.title()} — {ts}")
            lines.append('')
            for ctype, text in blocks:
                lines.append(f"**{ctype}**")
                lines.append('')
                lines.append('```text')
                lines.append(text)
                lines.append('```')
                lines.append('')
            continue

        # Event stream messages with plain text payloads
        if ptype in {'agent_message', 'user_message'}:
            msg = payload.get('message')
            if not msg:
                continue
            kept += 1
            speaker = 'Assistant' if ptype == 'agent_message' else 'User'
            lines.append(f"## {speaker} ({ptype}) — {ts}")
            lines.append('')
            lines.append('```text')
            lines.append(str(msg))
            lines.append('```')
            lines.append('')
            continue

        # Optional thinking snippets (only if textual)
        if ptype == 'thinking':
            value = payload.get('value')
            text = None
            if isinstance(value, str):
                text = value
            elif isinstance(value, list):
                text = '\n'.join(str(v) for v in value if isinstance(v, str) and v.strip())
            if text and text.strip():
                kept += 1
                lines.append(f"## Thinking — {ts}")
                lines.append('')
                lines.append('```text')
                lines.append(text)
                lines.append('```')
                lines.append('')

    if kept == 0:
        lines.append('_No compact message content found in this extract._')

    OUT.write_text('\n'.join(lines), encoding='utf-8')
    print(str(OUT))
    print(f'kept_sections={kept}')


if __name__ == '__main__':
    main()
