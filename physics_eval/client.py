"""OpenAI-compatible async client and a bounded, resumable runner."""
import asyncio
import json
import os


def make_client(base_url, api_key_env='OPENAI_API_KEY', timeout=3600):
    from openai import AsyncOpenAI
    return AsyncOpenAI(base_url=base_url, api_key=os.environ.get(api_key_env, 'EMPTY'), timeout=timeout, max_retries=2)


async def run_all(items, worker, out_path, concurrency):
    """Run `worker` over items with bounded concurrency, appending each result to out_path as it finishes."""
    sem = asyncio.Semaphore(concurrency)

    async def bounded(item):
        async with sem:
            return await worker(item)

    with open(out_path, 'a') as f:
        for i, fut in enumerate(asyncio.as_completed([bounded(x) for x in items]), 1):
            f.write(json.dumps(await fut, ensure_ascii=False) + '\n')
            f.flush()
            if i % 25 == 0 or i == len(items):
                print(f'{i}/{len(items)} done', flush=True)
