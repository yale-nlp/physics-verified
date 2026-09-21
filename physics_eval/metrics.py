"""Scores: strict (all answers of a problem correct), answer-level, partial credit, HLE calibration error, bootstrap CIs."""
import random
from collections import defaultdict

from .data import load_rows, read_jsonl


def calib_err(confidence, correct, beta=100):
    """RMS calibration error as in HLE's run_judge_results.py (bins of `beta`; the last bin is skipped there too)."""
    pairs = sorted(zip(confidence, correct))
    n = len(pairs)
    if n < beta:
        return None
    bins = [[i * beta, (i + 1) * beta] for i in range(n // beta)]
    bins[-1][1] = n
    total = 0.0
    for lo, hi in bins[:-1]:
        conf, corr = [c for c, _ in pairs[lo:hi]], [k for _, k in pairs[lo:hi]]
        total += len(conf) / n * (sum(conf) / len(conf) - sum(corr) / len(corr)) ** 2
    return total ** 0.5


def summarize(problems):
    flags = [f for p in problems for f in p]
    return {'strict_accuracy': sum(all(p) for p in problems) / len(problems),
            'answer_accuracy': sum(flags) / len(flags),
            'partial_credit': sum(sum(p) / len(p) for p in problems) / len(problems)}


def bootstrap(problems, reps=2000, seed=0):
    rng = random.Random(seed)
    boots = [summarize([problems[rng.randrange(len(problems))] for _ in problems]) for _ in range(reps)]
    return {k: [sorted(b[k] for b in boots)[int(0.025 * reps)], sorted(b[k] for b in boots)[int(0.975 * reps) - 1]] for k in boots[0]}


def score_rows(rows, judged, reps=2000):
    """Every row counts; a problem without a judgment is wrong on all of its answers (as in HLE)."""
    problems, conf, by_domain, by_type = [], [], defaultdict(list), defaultdict(list)
    for row in rows:
        rec = judged.get(row['id'])
        flags = [a['correct'] for a in rec['judgment']['answers']] if rec and 'judgment' in rec else [False] * len(row['final_answers'])
        problems.append(flags)
        conf.append((rec['judgment']['confidence'] if rec and 'judgment' in rec else 100) / 100)
        by_domain[row['domain']].append(flags)
        for t, f in zip(row['answer_types'], flags):
            by_type[t].append(f)
    result = summarize(problems)
    result['ci95'] = bootstrap(problems, reps)
    result['calibration_error'] = calib_err(conf, [all(p) for p in problems])
    result['problems'], result['answers'] = len(problems), sum(len(p) for p in problems)
    result['unjudged'] = sum(row['id'] not in judged or 'judgment' not in judged[row['id']] for row in rows)
    result['strict_accuracy_by_domain'] = {d: summarize(ps)['strict_accuracy'] for d, ps in sorted(by_domain.items())}
    result['answer_accuracy_by_type'] = {t: sum(v) / len(v) for t, v in sorted(by_type.items())}
    return result


def score(args):
    return score_rows(load_rows(args.split, args.data), read_jsonl(args.judged), args.bootstrap)
