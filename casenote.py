"""casenote.py - Part C: Structured Prompting, 3-Gate Validation, Retry and Output Guardrails."""

import json
from retrieval import retrieve, MIN_SCORE
import mock_llm

VALID_PRIORITIES = {"low", "medium", "high"}
VALID_TEAMS = {"card_risk", "upi_risk", "merchant_ops", "customer_care"}
VALID_ACTIONS = {"approve", "hold_for_review", "block_and_notify"}
VALID_POLICIES = {"P1", "P2", "P3", "P4", "P5", "P6"}


def build_prompt(txn_id: str, fraud_probability: float, amount_inr: float, policy_id: str, policy_text: str) -> str:
    return f"""ROLE: Fraud Risk Policy Specialist at Razorpay.
TASK: Analyze the flagged payment against the provided merchant risk policy and output a standardized structured case note.
CONSTRAINTS:
1. Base your determination solely on the provided POLICY_TEXT.
2. Output exactly one valid JSON object and no surrounding conversational prose.
3. The JSON must strictly contain keys: summary, priority, recommended_team, recommended_action, policy_id.
OUTPUT FORMAT:
{{
  "summary": string,
  "priority": "low" | "medium" | "high",
  "recommended_team": "card_risk" | "upi_risk" | "merchant_ops" | "customer_care",
  "recommended_action": "approve" | "hold_for_review" | "block_and_notify",
  "policy_id": "P1" | "P2" | "P3" | "P4" | "P5" | "P6"
}}

POLICY_TEXT: {policy_text}

TXN_ID: {txn_id}
FRAUD_PROBABILITY: {fraud_probability}
AMOUNT_INR: {amount_inr}
POLICY_ID: {policy_id}"""


def validate(text: str, retrieved_policy_id: str):
    """Three-gate schema and domain validation."""
    # Gate 1: Valid JSON object
    try:
        note = json.loads(text)
        if not isinstance(note, dict):
            return None, 1
    except Exception:
        return None, 1

    # Gate 2: Required keys and types
    required = ["summary", "priority", "recommended_team", "recommended_action", "policy_id"]
    for k in required:
        if k not in note or not isinstance(note[k], str):
            return None, 2

    # Gate 3: Allowed enum domains and policy integrity
    if note["priority"] not in VALID_PRIORITIES:
        return None, 3
    if note["recommended_team"] not in VALID_TEAMS:
        return None, 3
    if note["recommended_action"] not in VALID_ACTIONS:
        return None, 3
    if note["policy_id"] not in VALID_POLICIES or note["policy_id"] != retrieved_policy_id:
        return None, 3

    return note, 0


def run_case(txn_id: str, fraud_probability: float, amount_inr: float, question: str, max_attempts: int = 3) -> dict:
    policy_id, policy_text, score = retrieve(question)
    if score < MIN_SCORE:
        return {
            "status": "no_policy_match",
            "attempts": 0,
            "failures": [],
            "case_note": None,
            "safe_default": False,
            "guardrail_override": False,
        }

    prompt = build_prompt(txn_id, fraud_probability, amount_inr, policy_id, policy_text)
    attempts = 0
    failures = []
    passed_note = None

    for _ in range(max_attempts):
        attempts += 1
        # Let ConnectionError bubble up to caller per task specification
        response_text = mock_llm.generate(prompt)
        note, gate = validate(response_text, policy_id)
        if gate == 0:
            passed_note = note
            break
        else:
            failures.append(gate)

    safe_default_used = False
    if passed_note is None:
        safe_default_used = True
        passed_note = {
            "summary": "Automated case note unavailable; manual review required.",
            "priority": "high",
            "recommended_team": "customer_care",
            "recommended_action": "hold_for_review",
            "policy_id": policy_id,
        }

    # Output guardrail: Do not approve high/medium risk transactions
    guardrail_override = False
    if passed_note["recommended_action"] == "approve" and fraud_probability >= 0.5:
        passed_note["recommended_action"] = "hold_for_review"
        guardrail_override = True

    return {
        "status": "ok",
        "attempts": attempts,
        "failures": failures,
        "case_note": passed_note,
        "safe_default": safe_default_used,
        "guardrail_override": guardrail_override,
    }


if __name__ == "__main__":
    with open("case_scenarios.json", "r") as f:
        scenarios = json.load(f)

    mock_llm.reset()
    print(f"{'Scenario':<14} | {'Policy':<6} | {'Att':<3} | {'Failures':<10} | {'Status':<6} | {'SafeDef':<7} | {'Final Action':<17} | Override")
    print("-" * 88)

    calls_total = 0
    for idx, sc in enumerate(scenarios, 1):
        res = run_case(sc["txn_id"], sc["fraud_probability"], sc["amount_inr"], sc["question"])
        calls_total += res["attempts"]
        sc_name = f"S{idx} ({sc['txn_id']})"
        action = res["case_note"]["recommended_action"] if res["case_note"] else "none"
        pol = res["case_note"]["policy_id"] if res["case_note"] else "none"
        print(f"{sc_name:<14} | {pol:<6} | {res['attempts']:<3} | {str(res['failures']):<10} | {res['status']:<6} | {'yes' if res['safe_default'] else 'no':<7} | {action:<17} | {'yes' if res['guardrail_override'] else 'no'}")

    print(f"\nTotal model calls made across scenarios: {calls_total}")