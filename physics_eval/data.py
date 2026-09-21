"""Dataset loading and resumable JSONL helpers."""
import json
from pathlib import Path

REPO = 'yale-nlp/physics-verified'
SPLITS = ('validation', 'test')


def load_rows(split='test', data=None):
    """Rows from local JSONL files (`data`) or from the Hugging Face Hub; split may be 'validation', 'test', or 'all'."""
    if data:
        rows = []
        for path in data:
            rows += [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
        return rows
    from datasets import load_dataset
    splits = SPLITS if split == 'all' else (split,)
    return [dict(r) for s in splits for r in load_dataset(REPO, split=s)]


def read_jsonl(path):
    """Latest record per id; unparseable lines (e.g. from an interrupted write) are skipped."""
    out = {}
    if Path(path).exists():
        for line in Path(path).read_text().splitlines():
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            out[rec['id']] = rec
    return out
