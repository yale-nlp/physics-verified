#!/usr/bin/env python3
"""Re-judge a random sample of judged predictions with a second judge and report per-answer agreement.

  python tools/judge_crosscheck.py --predictions preds.jsonl --judged judged.jsonl \
      --judge-model SECOND_JUDGE --base-url URL [--n 200] [--out crosscheck.jsonl]
"""
import argparse
import asyncio
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from physics_eval.client import make_client  # noqa: E402
from physics_eval.data import load_rows, read_jsonl  # noqa: E402
from physics_eval.judge import judge_one  # noqa: E402


async def main(args):
    rows = {r['id']: r for r in load_rows(args.split, args.data)}
    preds, first = read_jsonl(args.predictions), read_jsonl(args.judged)
    ids = sorted(k for k, v in first.items() if 'judgment' in v and not v.get('skipped') and k in rows and k in preds)
    random.Random(args.seed).shuffle(ids)
    ids = ids[:args.n]
    client = make_client(args.base_url, args.api_key_env, 900)
    sem = asyncio.Semaphore(args.concurrency)

    async def one(pid):
        async with sem:
            return pid, await judge_one(client, args, rows[pid], preds[pid])

    agree = total = 0
    pairs, out = {}, []
    for pid, rec in await asyncio.gather(*(one(i) for i in ids)):
        if 'judgment' not in rec:
            continue
        a = [x['correct'] for x in first[pid]['judgment']['answers']]
        b = [x['correct'] for x in rec['judgment']['answers']]
        out.append({'id': pid, 'first': a, 'second': b, 'second_record': rec})
        for x, y in zip(a, b):
            total += 1
            agree += x == y
            pairs[f'{x}/{y}'] = pairs.get(f'{x}/{y}', 0) + 1
    if args.out:
        Path(args.out).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in out))
    print(f'{len(out)} problems, {total} answers: agreement {agree}/{total} = {agree / max(total, 1):.3f}')
    print('first/second verdict pairs:', pairs)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--predictions', required=True)
    p.add_argument('--judged', required=True)
    p.add_argument('--judge-model', required=True)
    p.add_argument('--base-url', required=True)
    p.add_argument('--api-key-env', default='OPENAI_API_KEY')
    p.add_argument('--split', default='all', choices=['validation', 'test', 'all'])
    p.add_argument('--data', nargs='*')
    p.add_argument('--n', type=int, default=200)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--concurrency', type=int, default=16)
    p.add_argument('--retries', type=int, default=4)
    p.add_argument('--max-tokens', type=int, default=32768)
    p.add_argument('--out')
    asyncio.run(main(p.parse_args()))
