"""Command line: physics-eval predict | judge | score."""
import argparse
import asyncio
import json


def main(argv=None):
    p = argparse.ArgumentParser(prog='physics-eval', description='Evaluate models on PHYSICS-Verified (yale-nlp/physics-verified).')
    sub = p.add_subparsers(dest='cmd', required=True)
    for name in ('predict', 'judge', 'score'):
        s = sub.add_parser(name)
        s.add_argument('--split', default='test', choices=['validation', 'test', 'all'])
        s.add_argument('--data', nargs='*', help='local JSONL files instead of the Hub dataset')
        if name != 'score':
            s.add_argument('--base-url', required=True, help='OpenAI-compatible endpoint, e.g. http://localhost:8000/v1')
            s.add_argument('--api-key-env', default='OPENAI_API_KEY', help='environment variable holding the API key')
            s.add_argument('--out', required=True, help='output JSONL; reruns resume from it')
            s.add_argument('--concurrency', type=int, default=16)
            s.add_argument('--timeout', type=float, default=3600)
    sp = sub.choices['predict']
    sp.add_argument('--model', required=True)
    sp.add_argument('--temperature', type=float)
    sp.add_argument('--top-p', type=float)
    sp.add_argument('--max-tokens', type=int)
    sp.add_argument('--seed', type=int)
    sp.add_argument('--extra-body', default='{}', help='JSON merged into the request, e.g. \'{"top_k": 20, "chat_template_kwargs": {"enable_thinking": true}}\'')
    sj = sub.choices['judge']
    sj.add_argument('--predictions', required=True)
    sj.add_argument('--judge-model', required=True)
    sj.add_argument('--retries', type=int, default=4)
    sj.add_argument('--max-tokens', type=int, default=32768)
    ss = sub.choices['score']
    ss.add_argument('--judged', required=True)
    ss.add_argument('--bootstrap', type=int, default=2000)
    ss.add_argument('--out', help='write the metrics JSON here as well')
    args = p.parse_args(argv)
    if args.cmd == 'predict':
        from .predict import predict
        asyncio.run(predict(args))
    elif args.cmd == 'judge':
        from .judge import judge
        asyncio.run(judge(args))
    else:
        from .metrics import score
        text = json.dumps(score(args), indent=2)
        print(text)
        if args.out:
            open(args.out, 'w').write(text + '\n')


if __name__ == '__main__':
    main()
