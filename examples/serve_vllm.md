# Serving an open model with vLLM

Start an OpenAI-compatible server, then point `physics-eval predict --base-url http://localhost:8000/v1` at it. Use a reasoning parser so that the thinking text is returned separately from the final answer.

```bash
vllm serve Qwen/Qwen3.5-9B --served-model-name Qwen3.5-9B \
    --max-model-len 81920 --reasoning-parser qwen3 --limit-mm-per-prompt '{"image": 8}'

physics-eval predict --split all --model Qwen3.5-9B --base-url http://localhost:8000/v1 \
    --temperature 1.0 --top-p 0.95 --max-tokens 65536 --seed 0 --extra-body '{"top_k": 20}' --out qwen3.5-9b.jsonl
```

Settings used for the published results:

| Model | Reasoning parser | Request extras | Sampling |
| --- | --- | --- | --- |
| Qwen3.5-9B | `qwen3` | `{"top_k": 20}` | T 1.0, top_p 0.95 |
| Qwen3.8-27B | `qwen3` | `{"top_k": 20}` (reasoning effort defaults to `xhigh`) | T 1.0, top_p 0.95 |
| GLM-5.3-Flash | `glm47` | reasoning effort left at its maximum default | T 1.0, top_p 0.95 |
| gemma-4-31B-it | `gemma4` | `{"top_k": 64, "chat_template_kwargs": {"enable_thinking": true}}` | T 1.0, top_p 0.95 |
| DeepSeek-V4.1-Flash | `deepseek_v41` | `{"reasoning_effort": 100}` (the maximum; the default is 50) | T 1.0, top_p 0.95 |

All runs use `--max-model-len 81920`, `--max-tokens 65536`, and seed 0, with one sample per problem. GLM-5.3-Flash and DeepSeek-V4.1-Flash were served with tensor parallelism across four and eight GPUs respectively, with an fp8 KV cache; DeepSeek-V4.1-Flash needs vLLM 0.30 or newer. Several problems include figures, so serve vision-capable models with image input enabled.
