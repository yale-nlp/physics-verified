"""Collect model responses in the PHYSICS-Verified response format."""
import json

from .client import make_client, run_all
from .data import load_rows, read_jsonl
from .prompts import SYSTEM_PROMPT


def messages_for(row):
    content = [{'type': 'text', 'text': row['questions']}] + (row['figures'] or [])
    return [{'role': 'system', 'content': SYSTEM_PROMPT}, {'role': 'user', 'content': content}]


async def predict(args):
    rows = load_rows(args.split, args.data)
    done = {k for k, v in read_jsonl(args.out).items() if v.get('error') is None and v.get('finish_reason') != 'abort'}
    todo = [r for r in rows if r['id'] not in done]
    print(f'{len(rows)} problems, {len(todo)} to answer', flush=True)
    client = make_client(args.base_url, args.api_key_env, args.timeout)
    params = {k: v for k, v in {'temperature': args.temperature, 'top_p': args.top_p, 'max_tokens': args.max_tokens,
                                'seed': args.seed}.items() if v is not None}
    extra = json.loads(args.extra_body)

    async def one(row):
        rec = {'id': row['id'], 'model': args.model, 'params': {**params, 'extra_body': extra}}
        try:
            r = await client.chat.completions.create(model=args.model, messages=messages_for(row), extra_body=extra or None, **params)
            choice = r.choices[0]
            if choice.finish_reason == 'abort':
                raise RuntimeError('request aborted by the server')
            msg = choice.message
            rec.update(response=msg.content or '', finish_reason=choice.finish_reason, error=None,
                       reasoning=getattr(msg, 'reasoning_content', None) or getattr(msg, 'reasoning', None),
                       usage=r.usage.model_dump() if r.usage else None)
        except Exception as e:  # recorded; rerun the command to retry
            rec.update(response=None, error=f'{type(e).__name__}: {e}'[:500])
        return rec

    await run_all(todo, one, args.out, args.concurrency)
