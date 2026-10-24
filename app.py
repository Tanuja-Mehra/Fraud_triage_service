"""app.py - Part D: FastAPI Production Triage Service."""

import time
import logging
from typing import Optional
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from prepare import add_features
from train import build_final_model
from casenote import run_case

# Set up standard logger
logging.basicConfig(
    filename="service.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("fraud_triage")

app = FastAPI(title="Razorpay Merchant Risk Desk Fraud Triage Service")

# Train & hydrate model artifact in-memory at startup
ARTIFACT = build_final_model()
MODEL = ARTIFACT["model"]
EXPECTED_COLUMNS = ARTIFACT["columns"]
TRAIN_MEDIAN = ARTIFACT["median"]
THRESHOLD = ARTIFACT["threshold"]

KNOWN_CATEGORIES = {
    "Grocery",
    "Electronics",
    "Travel",
    "Food Delivery",
    "Fashion",
    "Utilities",
}

# In-memory service metrics
METRICS = {
    "requests_total": 0,
    "success_total": 0,
    "error_total": 0,
    "human_review_total": 0,
    "total_latency_ms": 0.0,
}


@app.middleware("http")
async def observe_requests(request: Request, call_next):
    t0 = time.perf_counter()
    request.state.info = {}
    response = await call_next(request)

    # Middleware only updates state and logs for POST /score and POST /case
    if request.method == "POST":
        latency_ms = (time.perf_counter() - t0) * 1000.0
        status_code = response.status_code

        METRICS["requests_total"] += 1
        METRICS["total_latency_ms"] += latency_ms

        if status_code < 400:
            METRICS["success_total"] += 1
        else:
            METRICS["error_total"] += 1

        info = getattr(request.state, "info", {})
        if info.get("human_review", False):
            METRICS["human_review_total"] += 1

        # Determine log severity level
        safe_def = info.get("safe_default", False)
        override = info.get("override", False)

        if status_code >= 500:
            lvl = logging.ERROR
        elif status_code >= 400 or safe_def or override:
            lvl = logging.WARNING
        else:
            lvl = logging.INFO

        txn_id = info.get("txn_id", "UNKNOWN")
        prob = info.get("probability", None)
        prob_str = f" prob={prob:.4f}" if prob is not None else ""
        msg = f"endpoint={request.url.path} status={status_code} latency={latency_ms:.2f}ms txn_id={txn_id}{prob_str}"

        logger.log(lvl, msg)

    return response


class Txn(BaseModel):
    txn_id: str
    merchant_category: str
    city: str
    payment_mode: str
    amount_inr: float
    hour_of_day: int
    txn_count_last_24h: int
    avg_amount_30d_inr: float
    distance_from_home_km: Optional[float] = None
    is_new_device: int
    account_age_days: int


class CaseIn(Txn):
    question: str


def validate_input_bounds(t: Txn):
    """Custom input validation per Task 4."""
    if not (1.0 <= t.amount_inr <= 1000000.0):
        return "amount_inr", "amount_inr outside allowed range [1, 1000000]"
    if not (0 <= t.hour_of_day <= 23):
        return "hour_of_day", "hour_of_day outside 0 to 23"
    if t.txn_count_last_24h < 0:
        return "txn_count_last_24h", "txn_count_last_24h must be non-negative"
    if t.avg_amount_30d_inr <= 0:
        return "avg_amount_30d_inr", "avg_amount_30d_inr must be strictly positive"

    norm_category = t.merchant_category.strip().title()
    if norm_category not in KNOWN_CATEGORIES:
        return "merchant_category", f"Unknown merchant category: {norm_category}"

    return None, None


def score_transaction(t: Txn):
    """Executes feature engineering and tabular model inference for a transaction."""
    row = {
        "txn_id": t.txn_id,
        "merchant_category": t.merchant_category.strip().title(),
        "city": t.city.strip().title(),
        "payment_mode": t.payment_mode,
        "amount_inr": t.amount_inr,
        "hour_of_day": t.hour_of_day,
        "txn_count_last_24h": t.txn_count_last_24h,
        "avg_amount_30d_inr": t.avg_amount_30d_inr,
        "distance_from_home_km": t.distance_from_home_km,
        "is_new_device": t.is_new_device,
        "account_age_days": t.account_age_days,
    }

    df = pd.DataFrame([row])
    featured = add_features(df)
    encoded = pd.get_dummies(featured, dtype=int)

    # Impute missing distance using the training set median
    if (
        "distance_from_home_km" not in encoded
        or pd.isna(encoded.loc[0, "distance_from_home_km"])
    ):
        encoded.loc[0, "distance_from_home_km"] = TRAIN_MEDIAN

    # Reindex columns to align with training specification
    aligned = encoded.reindex(columns=EXPECTED_COLUMNS, fill_value=0)

    prob = float(MODEL.predict_proba(aligned)[0, 1])
    flagged = bool(prob >= THRESHOLD)
    decision = "hold_for_review" if flagged else "approve"

    # Task 5 output guardrail: Never auto-approve very large transactions on new devices
    large_payment_hold = False
    if not flagged and t.amount_inr >= 100000.0 and t.is_new_device == 1:
        decision = "hold_for_review"
        large_payment_hold = True

    human_review = bool(decision == "hold_for_review")

    return {
        "txn_id": t.txn_id,
        "fraud_probability": round(prob, 4),
        "flagged": flagged,
        "decision": decision,
        "human_review": human_review,
        "large_payment_hold": large_payment_hold,
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/score")
def score_endpoint(t: Txn, request: Request):
    err_field, err_msg = validate_input_bounds(t)
    if err_field:
        request.state.info = {"txn_id": t.txn_id}
        return JSONResponse(status_code=400, content={"error": err_msg, "field": err_field})

    res = score_transaction(t)
    request.state.info = {
        "txn_id": res["txn_id"],
        "probability": res["fraud_probability"],
        "human_review": res["human_review"],
        "override": res["large_payment_hold"],
    }
    return {
        "txn_id": res["txn_id"],
        "fraud_probability": res["fraud_probability"],
        "flagged": res["flagged"],
        "decision": res["decision"],
        "human_review": res["human_review"],
    }


@app.post("/case")
def case_endpoint(payload: CaseIn, request: Request):
    err_field, err_msg = validate_input_bounds(payload)
    if err_field:
        request.state.info = {"txn_id": payload.txn_id}
        return JSONResponse(status_code=400, content={"error": err_msg, "field": err_field})

    score_res = score_transaction(payload)
    flagged = score_res["flagged"]
    prob = score_res["fraud_probability"]

    case_note = None
    attempts = 0
    safe_def = False
    guardrail_override = score_res["large_payment_hold"]
    decision = score_res["decision"]

    if flagged:
        try:
            case_res = run_case(payload.txn_id, prob, payload.amount_inr, payload.question)
        except ConnectionError:
            request.state.info = {"txn_id": payload.txn_id, "probability": prob}
            return JSONResponse(
                status_code=503,
                content={"error": "case-note service temporarily unavailable, try again later"},
            )

        attempts = case_res["attempts"]
        safe_def = case_res["safe_default"]
        guardrail_override = case_res["guardrail_override"]

        if case_res["status"] == "no_policy_match":
            decision = "hold_for_review"
            case_note = None
        else:
            case_note = case_res["case_note"]
            # Flagged transactions cannot be approved
            if case_note["recommended_action"] == "approve":
                case_note["recommended_action"] = "hold_for_review"
                guardrail_override = True

            decision = case_note["recommended_action"]

    # Calculate final human review requirement
    human_review = bool(decision in ("hold_for_review", "block_and_notify") or safe_def)

    request.state.info = {
        "txn_id": payload.txn_id,
        "probability": prob,
        "human_review": human_review,
        "safe_default": safe_def,
        "override": guardrail_override,
    }

    return {
        "txn_id": payload.txn_id,
        "fraud_probability": prob,
        "flagged": flagged,
        "decision": decision,
        "human_review": human_review,
        "case_note": case_note,
        "attempts": attempts,
        "safe_default": safe_def,
        "guardrail_override": guardrail_override,
    }


@app.get("/metrics")
def metrics():
    req_total = METRICS["requests_total"]
    succ_total = METRICS["success_total"]
    err_total = METRICS["error_total"]
    hr_total = METRICS["human_review_total"]

    succ_rate = round(succ_total / req_total, 4) if req_total > 0 else 0.0
    err_rate = round(err_total / req_total, 4) if req_total > 0 else 0.0
    hr_rate = round(hr_total / req_total, 4) if req_total > 0 else 0.0
    avg_lat = round(METRICS["total_latency_ms"] / req_total, 2) if req_total > 0 else 0.0

    return {
        "requests_total": req_total,
        "success_total": succ_total,
        "error_total": err_total,
        "success_rate": succ_rate,
        "error_rate": err_rate,
        "avg_latency_ms": avg_lat,
        "human_review_total": hr_total,
        "human_review_rate": hr_rate,
    }