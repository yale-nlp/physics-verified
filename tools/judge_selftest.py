#!/usr/bin/env python3
"""Adversarial self-test of a judge model: synthetic responses built from reference answers with known verdicts.

Variants: the reference itself (correct), a numeric value off by 0.5% (correct under the 1% tolerance), and wrong
variants: a value doubled, a formula multiplied by 2, a hedge between the right and a wrong answer, a missing answer,
and answers swapped between two differently labeled parts. Run it before trusting a new judge model.

  python tools/judge_selftest.py --judge-model MODEL --base-url URL [--api-key-env VAR] [--per-type 20]
"""
import argparse
import asyncio
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from physics_eval.client import make_client  # noqa: E402
from physics_eval.data import load_rows  # noqa: E402
from physics_eval.judge import judge_one  # noqa: E402

NUM = re.compile(r'(?<![\w^{_])(\d+(?:\.\d+)?)(?:\s*\\times\s*10\^\{?(-?\d+)\}?)?')


def scale(text, factor):
    matches = list(NUM.finditer(text))
    if not matches:
        return None
    m = matches[-1]
    return text[:m.start(1)] + f'{float(m.group(1)) * factor:.4g}' + text[m.end(1):]


def double_formula(text):
    rhs = text.split('=')[-1].strip()
    balanced = all(rhs.count(a) == rhs.count(b) for a, b in ['()', '{}', '[]'])
    if (not rhs or '\\text' in rhs or not balanced or re.fullmatch(r'0|\\infty', rhs)
            or any(rel in text for rel in ['\\propto', '\\approx', '\\sim', '\\equiv'])):
        return None
    return text[:len(text) - len(text.split('=')[-1])] + f' 2\\left({rhs}\\right)' if '=' in text else f'2\\left({text}\\right)'


def response(labels, answers):
    lines = [answer if label.replace(',', '').isdigit() else f'{label} {answer}' for label, answer in zip(labels, answers)]
    return 'Explanation: Worked through the problem.\nAnswers:\n' + '\n'.join(lines) + '\nConfidence: 70%'


def build_cases(rows, per_type, seed=0):
    rng = random.Random(seed)
    cases = []
    for kind in ('numeric', 'symbolic', 'text'):
        pool = [(r, i) for r in rows for i, t in enumerate(r['answer_types']) if t == kind]
        for row, i in rng.sample(pool, min(per_type, len(pool))):
            labels, refs = row['answer_labels'], row['final_answers']

            def add(name, answers, ok, extra_wrong=()):
                if not ok and answers[i] == refs[i]:
                    return
                expected = [True] * len(refs)
                expected[i] = ok
                for j in extra_wrong:
                    expected[j] = False
                cases.append({'row': row, 'target': i, 'type': kind, 'variant': name, 'response': response(labels, answers), 'expected': expected})

            add('reference', list(refs), True)
            if kind == 'numeric' and scale(refs[i], 1.005):
                add('numeric_within_tolerance', refs[:i] + [scale(refs[i], 1.005)] + refs[i + 1:], True)
            if kind == 'numeric' and scale(refs[i], 2):
                add('numeric_doubled', refs[:i] + [scale(refs[i], 2)] + refs[i + 1:], False)
            if kind == 'symbolic' and double_formula(refs[i]):
                add('formula_doubled', refs[:i] + [double_formula(refs[i])] + refs[i + 1:], False)
            wrong = next((w for w in (scale(refs[i], 2), double_formula(refs[i])) if w and w != refs[i]), 'The opposite of the stated conclusion holds.')
            add('hedged', refs[:i] + [f'either {refs[i]} or {wrong}'] + refs[i + 1:], False)
            add('missing', refs[:i] + ['(no answer given)'] + refs[i + 1:], False)
            j = (i + 1) % len(refs)
            marker = lambda label: not label.replace(',', '').isdigit()
            if len(refs) > 1 and refs[i] != refs[j] and labels[i] != labels[j] and marker(labels[i]) and marker(labels[j]):
                swapped = list(refs)
                swapped[i], swapped[j] = refs[j], refs[i]
                add('swapped_parts', swapped, False, extra_wrong=(j,))
    return cases


async def main(args):
    rows = load_rows(args.split, args.data)
    cases = build_cases(rows, args.per_type, args.seed)
    client = make_client(args.base_url, args.api_key_env, 900)
    sem = asyncio.Semaphore(args.concurrency)

    async def one(case):
        async with sem:
            rec = await judge_one(client, args, case['row'], {'model': 'selftest', 'response': case['response']})
        got = [a['correct'] for a in rec['judgment']['answers']] if 'judgment' in rec else None
        return case, got

    table = {}
    for case, got in await asyncio.gather(*(one(c) for c in cases)):
        t = table.setdefault((case['variant'], case['type']), [0, 0, 0])
        t[0] += 1
        if got is None:
            t[2] += 1
        else:
            t[1] += got[case['target']] == case['expected'][case['target']]
    print('| variant | type | cases | target verdict as expected | judge errors |\n| --- | --- | --- | --- | --- |')
    for (variant, kind), (n, agree, errors) in sorted(table.items()):
        print(f'| {variant} | {kind} | {n} | {agree}/{n - errors} | {errors} |')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--judge-model', required=True)
    p.add_argument('--base-url', required=True)
    p.add_argument('--api-key-env', default='OPENAI_API_KEY')
    p.add_argument('--split', default='test', choices=['validation', 'test', 'all'])
    p.add_argument('--data', nargs='*')
    p.add_argument('--per-type', type=int, default=20)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--concurrency', type=int, default=16)
    p.add_argument('--retries', type=int, default=4)
    p.add_argument('--max-tokens', type=int, default=32768)
    asyncio.run(main(p.parse_args()))
