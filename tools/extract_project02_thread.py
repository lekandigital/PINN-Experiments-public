import json
from pathlib import Path

SRC = Path('/Users/lekan/.codex/sessions/2026/04/25/rollout-2026-04-25T07-25-13-019dc462-f734-7501-855f-d8c205e821ca.jsonl')
OUT = Path('/Users/lekan/Downloads/project02_thread_extract.json')

START_MARKER = '## Current Project-02 State To Re-Verify'
END_MARKER = 'May I publish the markdown post to `/Users/lekan/Dev/1001ud.me/blog/content/Technical/Experimental/Physics-ML/Neural-Surrogates/02.wavepinn-nif-complexmedia-acoustic-field-demo.md`?'


def main() -> None:
    rows = []
    with SRC.open('r', encoding='utf-8') as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            blob = json.dumps(obj, ensure_ascii=False)
            rows.append((i, obj, blob))

    start_idx = None
    end_idx = None
    for idx, (line_no, obj, blob) in enumerate(rows):
        if start_idx is None and START_MARKER in blob:
            start_idx = idx
        if start_idx is not None and END_MARKER in blob:
            end_idx = idx
            break

    if start_idx is None:
        raise RuntimeError('START_MARKER_NOT_FOUND')
    if end_idx is None:
        raise RuntimeError('END_MARKER_NOT_FOUND')

    selected = rows[start_idx:end_idx + 1]
    payload = {
        'source_file': str(SRC),
        'start_marker': START_MARKER,
        'end_marker': END_MARKER,
        'start_line_in_jsonl': selected[0][0],
        'end_line_in_jsonl': selected[-1][0],
        'entry_count': len(selected),
        'entries': [r[1] for r in selected],
    }

    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print(str(OUT))
    print(f"entry_count={payload['entry_count']}")
    print(f"line_range={payload['start_line_in_jsonl']}..{payload['end_line_in_jsonl']}")


if __name__ == '__main__':
    main()
