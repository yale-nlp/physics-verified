"""Response-format and judge prompts for PHYSICS-Verified evaluation."""

# HLE's response format, extended to several results: labeled answers in the order asked.
SYSTEM_PROMPT = (
    "Your response should be in the following format:\n"
    "Explanation: {your explanation for your answers}\n"
    "Answers: {your final answers, one per line, in the order the question asks for them; "
    "begin each line with the subquestion label, such as (a), when the question labels its parts}\n"
    "Confidence: {your confidence score between 0% and 100% for your answers}"
)

NUMERIC_REL_TOL = 0.01

JUDGE_PROMPT = """Judge whether the following [response] to [question] correctly gives each result in the list of [reference_answers] below. The reference answers are precise and authoritative.

[question]: {question}

[response]: {response}

[reference_answers]:
{references}

For EACH reference answer, in order, decide independently:

extracted_final_answer: the response's final answer for the same requested result, copied from its Answers section or, if absent there, from its final statement of that result. When the reference label is a part marker such as (a) or (b)(ii), take only what the response gives for that same part; an answer the response places under a different part does not count. When the label is an ordinal (1, 2, ...), the question has no part markers, so identify the answer by the quantity it gives. If the response's answer line for that part states only a choice, a comparison, or a qualitative statement, but its explanation for that same part states the requested quantity, extract that quantity. Use 'None' if the response gives no final answer for it.

reasoning: explain why the extracted answer matches the reference answer or not, focusing only on meaningful differences. Do not comment on background, do not solve the problem, and do not argue for any answer other than the reference answer.

correct: 'yes' only if the extracted answer is equivalent to the reference answer under these rules, otherwise 'no':
1. Symbolic answers: algebraically equivalent expressions are correct, including rearrangements and notation defined in the question. A missing or different factor, sign, exponent, or variable dependence is incorrect. Quantum state vectors or wavefunctions that differ only by an overall phase factor are equivalent.
2. Numeric answers: correct if within {rel_tol_pct}% relative difference of the reference value, or equal to it when rounded to the reference's significant figures. For order-of-magnitude estimates, the same order of magnitude is correct. Units must be equivalent (unit conversions are fine); a missing unit is incorrect unless the question fixes the unit.
3. If the reference states a formula followed by its value, compare the final result; the response must not contradict either.
4. Verbal answers: correct if they state the same conclusion, choice, or identification. Extra correct detail is fine; any contradiction is incorrect. When the reference gives a formula, value, or explicit result, a qualitative description of it is incorrect unless the response also states that formula or value for the same part.
5. Incorrect if the response gives several different candidate answers for one result, hedges between alternatives, answers a different quantity, gives the result only under a different part, or gives no final answer. Intermediate steps shown in the reference are not required.
6. An answer that equals the reference once a convention or conversion the response itself states is applied is correct. The response must state the convention, and the two must then agree exactly; a convention it does not state, or one that still leaves a difference in factor, sign, exponent, or variable dependence, does not make a wrong answer correct.

Also extract confidence: the confidence score between 0 and 100 stated in the response; use 100 if none is given.

Return only json in exactly this shape, with one entry per reference answer in the same order:
{{"answers": [{{"index": 0, "label": "(a)", "extracted_final_answer": "...", "reasoning": "...", "correct": "yes"}}], "confidence": 85}}"""


def format_references(labels, answers):
    return '\n'.join(f'{i}. label {label}: {answer}' for i, (label, answer) in enumerate(zip(labels, answers)))


def judge_prompt(question, response, labels, answers):
    return JUDGE_PROMPT.format(question=question, response=response, references=format_references(labels, answers),
                               rel_tol_pct=f'{NUMERIC_REL_TOL * 100:g}')
