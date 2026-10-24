"""smoke_test.py - Part D: Automated Verification Suite for Q1-Q11."""

import time
import requests

BASE_URL = "http://127.0.0.1:8000"

BENIGN = {
    "merchant_category": "Grocery",
    "city": "Pune",
    "payment_mode": "UPI",
    "amount_inr": 1500,
    "hour_of_day": 13,
    "txn_count_last_24h": 0,
    "avg_amount_30d_inr": 1600,
    "distance_from_home_km": 3.0,
    "is_new_device": 0,
    "account_age_days": 900,
}

RISKY = {
    "merchant_category": "Electronics",
    "city": "Mumbai",
    "payment_mode": "Card",
    "amount_inr": 45000,
    "hour_of_day": 2,
    "txn_count_last_24h": 9,
    "avg_amount_30d_inr": 3000,
    "distance_from_home_km": 60.0,
    "is_new_device": 1,
    "account_age_days": 40,
}

BIG = {
    **BENIGN,
    "merchant_category": "Utilities",
    "payment_mode": "NetBanking",
    "amount_inr": 150000,
    "avg_amount_30d_inr": 250000,
    "distance_from_home_km": 0.0,
    "is_new_device": 1,
    "account_age_days": 2200,
}


def wait_for_service():
    print("Waiting for server startup at http://127.0.0.1:8000 ...")
    for _ in range(120):
        try:
            r = requests.get(f"{BASE_URL}/health", timeout=1)
            if r.status_code == 200:
                print("Server is healthy and ready.\n")
                return True
        except requests.RequestException:
            pass
        time.sleep(1)
    raise RuntimeError("Server did not start up within 120 seconds.")


def run_smoke_suite():
    wait_for_service()

    tests = [
        # Q1
        ("Q1", "happy", "GET", "/health", None, 200, lambda r: r.json() == {"status": "ok"}),
        # Q2
        ("Q2", "happy", "POST", "/score", {**BENIGN, "txn_id": "T800001"}, 200,
         lambda r: r.json().get("flagged") is False and r.json().get("decision") == "approve" and r.json().get("human_review") is False),
        # Q3
        ("Q3", "escalation", "POST", "/score", {**RISKY, "txn_id": "T800002"}, 200,
         lambda r: r.json().get("flagged") is True and r.json().get("decision") == "hold_for_review" and r.json().get("human_review") is True),
        # Q4
        ("Q4", "escalation", "POST", "/score", {**BIG, "txn_id": "T800003"}, 200,
         lambda r: r.json().get("flagged") is False and r.json().get("decision") == "hold_for_review" and r.json().get("human_review") is True),
        # Q5
        ("Q5", "edge", "POST", "/score", {**BENIGN, "txn_id": "T800004", "amount_inr": -5}, 400,
         lambda r: "error" in r.json() and r.json().get("field") == "amount_inr"),
        # Q6
        ("Q6", "edge", "POST", "/score", {k: v for k, v in {**BENIGN, "txn_id": "T800005"}.items() if k != "hour_of_day"}, 422,
         lambda r: True),
        # Q7
        ("Q7", "escalation", "POST", "/case", {**RISKY, "txn_id": "T900002", "question": "more than 6 UPI payments in 24 hours velocity alert"}, 200,
         lambda r: r.json().get("flagged") is True and r.json().get("attempts") == 2 and r.json().get("safe_default") is False and r.json().get("case_note", {}).get("policy_id") == "P2" and r.json().get("human_review") is True),
        # Q8
        ("Q8", "escalation", "POST", "/case", {**RISKY, "txn_id": "T900005", "question": "electronics merchant payment on an account younger than 30 days"}, 200,
         lambda r: r.json().get("attempts") == 3 and r.json().get("safe_default") is True and r.json().get("decision") == "hold_for_review" and r.json().get("human_review") is True),
        # Q9
        ("Q9", "edge", "POST", "/case", {**RISKY, "txn_id": "T900099", "question": "card payment above INR 50000 from a new device"}, 503,
         lambda r: "error" in r.json()),
        # Q10
        ("Q10", "happy", "POST", "/case", {**BENIGN, "txn_id": "T800001", "question": "card payment above INR 50000 from a new device"}, 200,
         lambda r: r.json().get("case_note") is None and r.json().get("attempts") == 0 and r.json().get("decision") == "approve" and r.json().get("human_review") is False),
        # Q11
        ("Q11", "edge", "POST", "/case", {**RISKY, "txn_id": "T800006", "question": "what is the cafeteria menu today"}, 200,
         lambda r: r.json().get("case_note") is None and r.json().get("attempts") == 0 and r.json().get("decision") == "hold_for_review" and r.json().get("human_review") is True),
    ]

    cat_counts = {"happy": 0, "escalation": 0, "edge": 0}
    cat_totals = {"happy": 0, "escalation": 0, "edge": 0}

    print(f"{'ID':<4} | {'Category':<11} | {'Exp':<4} | {'Obs':<4} | Status")
    print("-" * 42)

    for qid, cat, method, path, payload, exp_status, validator in tests:
        cat_totals[cat] += 1
        url = f"{BASE_URL}{path}"
        if method == "GET":
            resp = requests.get(url)
        else:
            resp = requests.post(url, json=payload)

        obs_status = resp.status_code
        status_pass = (obs_status == exp_status)
        content_pass = validator(resp)

        passed = status_pass and content_pass
        if passed:
            cat_counts[cat] += 1

        print(f"{qid:<4} | {cat:<11} | {exp_status:<4} | {obs_status:<4} | {'PASS' if passed else 'FAIL'}")

    print("\nSummary by Category:")
    for cat in ["happy", "escalation", "edge"]:
        print(f"  {cat}: {cat_counts[cat]}/{cat_totals[cat]} passed")

    m_resp = requests.get(f"{BASE_URL}/metrics")
    print(f"\nFinal GET /metrics Output:\n{m_resp.json()}")


if __name__ == "__main__":
    run_smoke_suite()