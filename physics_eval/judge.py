"""LLM judge: one call per problem, one structured verdict per reference answer."""
import asyncio
import json

from .client import make_client, run_all
from .data import load_rows, read_jsonl
from .prompts import judge_prompt


def final_text(pred):
    """The visible answer; any reasoning block leaked into the content is dropped."""
    return (pred.get('response') or '').split('</think>')[-1].strip()


def parse_judgment(raw, n):
    """Validated judgment dict, or None when the judge output is malformed or has the wrong number of verdicts."""
    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        return None
    answers = data.get('answers') if isinstance(data, dict) else None
    if not isinstance(answers, list) or len(answers) != n:
        return None
    if any(not isinstance(a, dict) or str(a.get('correct', '')).strip().lower() not in ('yes', 'no') for a in answers):
        return None
    try:
        confidence = max(0, min(100, int(round(float(str(data.get('confidence', 100)).rstrip('%'))))))
    except ValueError:
        confidence = 100
    return {'answers': [{'index': i, 'extracted_final_answer': a.get('extracted_final_answer'), 'reasoning': a.get('reasoning'),
                         'correct': str(a['correct']).strip().lower() == 'yes'} for i, a in enumerate(answers)],
            'confidence': confidence}


def unanswered(n, reason):
    return {'confidence': 100, 'answers': [{'index': i, 'extracted_final_answer': None, 'reasoning': reason, 'correct': False}
                                           for i in range(n)]}


async def judge_one(client, args, row, pred):
    base = {'id': row['id'], 'model': pred.get('model'), 'judge_model': args.judge_model}
    response = final_text(pred)
    if pred.get('error') or not response:
        return {**base, 'skipped': 'no final answer', 'judgment': unanswered(len(row['final_answers']), 'no final answer')}
    prompt = judge_prompt(row['questions'], response, row['answer_labels'], row['final_answers'])
    for attempt in range(args.retries):
        try:
            r = await client.chat.completions.create(model=args.judge_model, messages=[{'role': 'user', 'content': prompt}],
                                                     response_format={'type': 'json_object'}, max_tokens=args.max_tokens)
            parsed = parse_judgment(r.choices[0].message.content, len(row['final_answers']))
            if parsed:
                return {**base, 'judgment': parsed, 'attempts': attempt + 1}
        except Exception:
            pass
        await asyncio.sleep(2 * (attempt + 1))
    return {**base, 'judge_error': True}


async def judge(args):
    rows = {r['id']: r for r in load_rows(args.split, args.data)}
    preds = read_jsonl(args.predictions)
    done = {k for k, v in read_jsonl(args.out).items() if 'judgment' in v}
    todo = [pid for pid in preds if pid in rows and pid not in done]
    print(f'{len(preds)} predictions, {len(todo)} to judge', flush=True)
    client = make_client(args.base_url, args.api_key_env, args.timeout)
    await run_all(todo, lambda pid: judge_one(client, args, rows[pid], preds[pid]), args.out, args.concurrency)
