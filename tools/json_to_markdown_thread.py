import json
from pathlib import Path
from datetime import datetime

SRC = Path('/Users/lekan/Downloads/project02_thread_extract.json')
OUT = Path('/Users/lekan/Downloads/project02_thread_extract.md')


def render_entry(entry_index: int, entry: dict) -> list[str]:
    lines: list[str] = []
    ts = entry.get('timestamp', 'n/a')
    etype = entry.get('type', 'unknown')
    payload = entry.get('payload')

    lines.append(f'## Entry {entry_index}')
    lines.append('')
    lines.append(f'- Timestamp: {ts}')
    lines.append(f'- Type: {etype}')
    if isinstance(payload, dict) and payload.get('type'):
        lines.append(f"- Payload type: {payload.get('type')}")
    lines.append('')

    rendered = False

    if isinstance(payload, dict) and payload.get('type') == 'message':
        role = payload.get('role', 'unknown')
        lines.append(f'### Message ({role})')
        lines.append('')
        for c in payload.get('content', []):
            ctype = c.get('type') if isinstance(c, dict) else None
            text = None
            if isinstance(c, dict):
                text = c.get('text') or c.get('input_text') or c.get('output_text')
            if ctype:
                lines.append(f'**{ctype}**')
            lines.append('')
            if text is not None:
                lines.append('```text')
                lines.append(str(text))
                lines.append('```')
            else:
                lines.append('```json')
                lines.append(json.dumps(c, ensure_ascii=False, indent=2))
                lines.append('```')
            lines.append('')
        rendered = True

    if not rendered and isinstance(payload, dict) and payload.get('type') in {'agent_message', 'user_message'}:
        lines.append(f"### {payload.get('type')}")
        lines.append('')
        msg = payload.get('message')
        if msg is not None:
            lines.append('```text')
            lines.append(str(msg))
            lines.append('```')
            lines.append('')
        rendered = True

    if not rendered:
        lines.append('### Raw Entry JSON')
        lines.append('')
        lines.append('```json')
        lines.append(json.dumps(entry, ensure_ascii=False, indent=2))
        lines.append('```')
        lines.append('')

    lines.append('---')
    lines.append('')
    return lines


def main() -> None:
    data = json.loads(SRC.read_text(encoding='utf-8'))
    entries = data.get('entries', [])

    lines: list[str] = []
    lines.append('# Project 02 Thread Extract')
    lines.append('')
    lines.append(f"- Source JSONL: {data.get('source_file', '')}")
    lines.append(f"- Line range: {data.get('start_line_in_jsonl', '?')} to {data.get('end_line_in_jsonl', '?')}")
    lines.append(f"- Entries: {data.get('entry_count', len(entries))}")
    lines.append(f"- Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append('')
    lines.append('---')
    lines.append('')

    for i, entry in enumerate(entries, start=1):
        lines.extend(render_entry(i, entry))

    OUT.write_text('\n'.join(lines), encoding='utf-8')
    print(str(OUT))
    print(f'entries={len(entries)}')


if __name__ == '__main__':
    main()
