# physics-verified

Evaluation harness for **PHYSICS-Verified** ([`yale-nlp/physics-verified`](https://huggingface.co/datasets/yale-nlp/physics-verified)). The benchmark has 1,109 PhD-qualifying-exam physics problems with 2,797 independently scored answers. It is a cleaned, results-only version of [PHYSICS](https://arxiv.org/abs/2503.21821).

The protocol follows [Humanity's Last Exam](https://arxiv.org/abs/2501.14249), extended to problems that ask for several results. The model gives labeled final answers and a confidence. An LLM judge returns one verdict per reference answer. Scores are reported per problem and per answer.

## Install

```bash
git clone https://github.com/yale-nlp/physics-verified && cd physics-verified
pip install -e .
```

## Quick start

Any OpenAI-compatible endpoint works: a hosted API, or a local [vLLM](examples/serve_vllm.md) server.

```bash
# 1. answer every problem (resumable; rerun to retry failures)
physics-eval predict --split all --model MODEL --base-url http://localhost:8000/v1 \
    --temperature 1.0 --top-p 0.95 --max-tokens 65536 --out preds.jsonl

# 2. judge each reference answer
export JUDGE_API_KEY=...
physics-eval judge --split all --predictions preds.jsonl --judge-model JUDGE_MODEL \
    --base-url https://api.example.com --api-key-env JUDGE_API_KEY --out judged.jsonl

# 3. score
physics-eval score --split all --judged judged.jsonl --out metrics.json
```

Use `--extra-body` for model-specific request fields such as `top_k` or thinking switches. Use `--split validation|test|all` to choose the split. Pass `--data file.jsonl` to run on local files instead of the Hub.

## Protocol

**Response format** (`physics_eval/prompts.py`). The system prompt asks for three parts:
- `Explanation:`;
- `Answers:`, with one final answer per line in the order asked, prefixed by the part label when the question has one;
- `Confidence:`, a percentage.

The number of expected answers is not disclosed.

**Judge**. There is one call per problem. The judge receives the question, the response, and the labeled reference answers. It returns strict JSON with a verdict for every reference answer, under these rules:
- Algebraically equivalent symbolic answers are correct. Quantum states that differ only by a global phase are equivalent.
- Numeric answers are correct within 1% relative error, or when they agree to the reference's significant figures.
- Order-of-magnitude estimates are correct if they have the same order of magnitude.
- Units must be equivalent.
- A labeled answer must appear under its own part.
- Hedged, missing, or misplaced answers are wrong. A qualitative description is wrong when the reference gives a formula or value, unless the response states that formula or value for the same part.
- Where a part's answer line gives only a choice or a qualitative statement, the quantity stated in its explanation is graded.
- An answer that equals the reference once a convention the response itself states is applied is correct; an unstated convention, or one that still leaves a difference, does not make a wrong answer right.

Malformed judge output, or output with the wrong number of verdicts, is retried. A response with no final answer, for example one truncated at the token limit, is scored as wrong without a judge call.

**Metrics** (`physics_eval/metrics.py`):
- **Strict accuracy** (headline): a problem is correct only if all of its answers are correct.
- **Answer-level accuracy.**
- **Partial credit:** the mean fraction of correct answers per problem.
- **RMS calibration error**, computed as in HLE.
- 95% confidence intervals come from a bootstrap that resamples problems. Problems without a prediction count as wrong.

## Results

Validation and test combined, 1,109 problems and 2,803 answers. Each model gets one sample per problem at its maximum or default thinking setting, with a limit of 65,536 output tokens. The judge is GLM-5.3-Flash, served locally. Values are percentages with 95% bootstrap CIs.

| Model | Strict | Answer-level | Partial credit | Calibration error |
| --- | --- | --- | --- | --- |
| DeepSeek-V4.1-Flash (thinking) | 82.0 [79.8, 84.1] | 89.4 [87.7, 90.9] | 89.2 [87.6, 90.7] | 24.9 |
| Qwen3.8-27B (thinking) | 81.0 [78.7, 83.4] | 87.8 [85.8, 89.7] | 87.7 [86.0, 89.5] | 29.1 |
| GLM-5.3-Flash (thinking) | 80.1 [77.8, 82.4] | 87.8 [86.0, 89.6] | 87.4 [85.8, 89.1] | 21.3 |
| gemma-4-31B-it (thinking) | 72.7 [70.1, 75.4] | 83.1 [81.2, 85.0] | 82.5 [80.5, 84.4] | 47.0 |
| Qwen3.5-9B (thinking) | 69.5 [66.7, 72.1] | 79.7 [77.1, 82.0] | 78.9 [76.7, 81.0] | 45.5 |

The judge is also one of the models under test. Judged instead by `deepseek-flash`, GLM-5.3-Flash scores 75.4 [72.9, 77.9] strict, 84.3 answer-level and 83.8 partial credit, 4.7 points below its score under itself. On a shared sample of 200 problems from another model's predictions the same two judges differ by 5.5 strict points in the same direction, so the drop is the judges' general strictness, not a model favouring itself. Its rank is unchanged.

Every model states a confidence near 95% and none is close to that accurate, which is what the calibration column measures.

Every response and every judge verdict behind this table is published at [yale-nlp/physics-verified-model-outputs](https://huggingface.co/datasets/yale-nlp/physics-verified-model-outputs), including the models' thinking text and the judge's reasoning for each answer.

## Checking a judge

The judge is an LLM, so its verdicts are noisy. Before relying on a new judge model, run two checks:

```bash
# known-verdict synthetic cases: correct references, 0.5% numeric shifts, doubled values/formulas, hedges, omissions, swapped parts
python tools/judge_selftest.py --judge-model JUDGE --base-url URL --api-key-env JUDGE_API_KEY

# agreement with a second judge on a random sample of real verdicts
python tools/judge_crosscheck.py --predictions preds.jsonl --judged judged.jsonl \
    --judge-model SECOND_JUDGE --base-url URL2 --api-key-env KEY2 --n 200
```

Results for GLM-5.3-Flash, the judge used above:
- The self-test agreed on 202 of 204 perturbed answers and all 481 unperturbed ones; both misses accepted a hedge in a verbal answer.
- A cross-check against `deepseek-flash` on 200 random problems agreed on 445 of 476 answers (93.5%). Of the 31 disagreements, 29 are GLM calling an answer correct where `deepseek-flash` calls it wrong.
- Four independent reviewers, shown the judges unlabelled, decided those 31: GLM was right on 27. Almost every `deepseek-flash` error rejects a correct answer over a convention the response itself states (SI against Gaussian units, Kittel's tau = k_B T, B against mu_0 H, a stated coordinate frame, the question's own phase factor) or applies the 1% tolerance to an explicit estimate. GLM's own errors include accepting an answer that covered only one of two required branches.

Scores computed with different judges are not directly comparable: on this benchmark `deepseek-flash` scores the same predictions about five points lower than GLM-5.3-Flash.

## Tests

```bash
pip install -e ".[test]" && pytest -q
```

The tests run offline against a fake endpoint.

## Citation

If you use PHYSICS-Verified, please cite the original benchmark:

```bibtex
@misc{feng2025physicsbenchmarkingfoundationmodels,
  title={PHYSICS: Benchmarking Foundation Models on University-Level Physics Problem Solving},
  author={Kaiyue Feng and Yilun Zhao and Yixin Liu and Tianyu Yang and Chen Zhao and John Sous and Arman Cohan},
  year={2025},
  eprint={2503.21821},
  archivePrefix={arXiv},
  primaryClass={physics.ed-ph},
  url={https://arxiv.org/abs/2503.21821}
}
```

## License

MIT.
