"""Offline tests: no model or network calls (the OpenAI client is replaced by a fake)."""
import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from physics_eval import cli, judge, metrics, predict, prompts  # noqa: E402
import judge_selftest  # noqa: E402

ROWS = [
    {'id': 'p1', 'domain': 'Optics', 'questions': 'Q1 (a) x? (b) y?', 'solutions': '', 'final_answers': ['x = 1', 'y = 2'],
     'answer_labels': ['(a)', '(b)'], 'answer_types': ['numeric', 'numeric'], 'figures': None},
    {'id': 'p2', 'domain': 'Optics', 'questions': 'Q2', 'solutions': '', 'final_answers': ['E = mc^2'],
     'answer_labels': ['1'], 'answer_types': ['symbolic'], 'figures': None},
]


@pytest.fixture
def data(tmp_path):
    path = tmp_path / 'rows.jsonl'
    path.write_text(''.join(json.dumps(r) + '\n' for r in ROWS))
    return path


class FakeClient:
    """Mimics AsyncOpenAI.chat.completions.create for the predict and judge calls."""
    def __init__(self, reply):
        self.reply = reply
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    async def create(self, model, messages, **kwargs):
        content = self.reply(messages)
        msg = SimpleNamespace(content=content, reasoning_content=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason='stop')], usage=None)


def test_prompt_lists_references_and_tolerance():
    text = prompts.judge_prompt('Q', 'R', ['(a)', '(b)'], ['x = 1', 'y = 2'])
    assert '0. label (a): x = 1' in text and '1. label (b): y = 2' in text and '1% relative' in text


def test_parse_judgment_validates():
    ok = json.dumps({'answers': [{'correct': 'yes'}, {'correct': 'No'}], 'confidence': '85%'})
    assert [a['correct'] for a in judge.parse_judgment(ok, 2)['answers']] == [True, False]
    assert judge.parse_judgment(ok, 3) is None
    assert judge.parse_judgment('not json', 1) is None
    assert judge.parse_judgment(json.dumps({'answers': [{'correct': 'maybe'}]}), 1) is None


def test_final_text_drops_reasoning():
    assert judge.final_text({'response': '<think>x</think>\nAnswers: 1'}) == 'Answers: 1'


def test_metrics_and_missing_counts_wrong():
    judged = {'p1': {'judgment': {'answers': [{'correct': True}, {'correct': False}], 'confidence': 90}}}
    r = metrics.score_rows(ROWS, judged, reps=50)
    assert r['strict_accuracy'] == 0 and r['answer_accuracy'] == 1 / 3 and r['partial_credit'] == 0.25 and r['unjudged'] == 1


def test_calibration_error_matches_hle():
    assert metrics.calib_err([0.5] * 10, [1] * 10) is None
    assert abs(metrics.calib_err([0.9] * 100 + [1.0] * 100, [0] * 100 + [1] * 100) - (0.5 * 0.81) ** 0.5) < 1e-9


def test_end_to_end_with_fake_endpoint(tmp_path, data, monkeypatch):
    preds, judged = tmp_path / 'preds.jsonl', tmp_path / 'judged.jsonl'
    monkeypatch.setattr(predict, 'make_client', lambda *a: FakeClient(lambda m: 'Explanation: e\nAnswers:\n(a) x = 1\n(b) y = 3\nConfidence: 80%'))
    cli.main(['predict', '--data', str(data), '--base-url', 'http://fake', '--model', 'm', '--out', str(preds)])
    verdict = json.dumps({'answers': [{'correct': 'yes'}, {'correct': 'no'}], 'confidence': 80})
    monkeypatch.setattr(judge, 'make_client', lambda *a: FakeClient(
        lambda m: verdict if '1. label (b)' in m[0]['content'] else json.dumps({'answers': [{'correct': 'yes'}], 'confidence': 80})))
    cli.main(['judge', '--data', str(data), '--base-url', 'http://fake', '--judge-model', 'j', '--predictions', str(preds), '--out', str(judged)])
    out = tmp_path / 'metrics.json'
    cli.main(['score', '--data', str(data), '--judged', str(judged), '--bootstrap', '20', '--out', str(out)])
    r = json.loads(out.read_text())
    assert r['strict_accuracy'] == 0.5 and r['answer_accuracy'] == 2 / 3 and r['unjudged'] == 0
    # resumable: a second predict run has nothing left to do
    cli.main(['predict', '--data', str(data), '--base-url', 'http://fake', '--model', 'm', '--out', str(preds)])
    assert len(preds.read_text().splitlines()) == 2


def test_selftest_builder():
    assert judge_selftest.scale(r'q = 6.7 \times 10^{-4}', 2) == r'q = 13.4 \times 10^{-4}'
    assert judge_selftest.double_formula(r'I \propto x') is None
    cases = judge_selftest.build_cases(ROWS, per_type=5)
    swapped = [c for c in cases if c['variant'] == 'swapped_parts']
    assert swapped and all(c['expected'] == [False, False] for c in swapped)
