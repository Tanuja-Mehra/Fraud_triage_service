# Razorpay Merchant Risk Desk — Fraud Triage Service

## Overview
The Fraud Triage Service is an automated risk decision system protecting payment settlements across UPI, Card, NetBanking, and Wallet rails.
It cleans and engineers real-time tabular transaction streams and trains a balanced Random Forest classifier with defensible operational thresholds.
When high-risk transactions are flagged, the system uses TF-IDF cosine similarity to retrieve applicable merchant risk policies.
It queries a structured language-model pipeline with three validation gates, retry mechanisms, and automated safety overrides to draft queue notes.
A hardened FastAPI service exposes scoring and case endpoints with strict input validation, HTTP observability middleware, and live operational metrics.

## Reproduce
```bash
# 1. Setup clean virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install pinned dependencies
pip install -r requirements.txt

# 3. Generate raw synthetic datasets
python generate_data.py

# 4. Part A: Audit, clean, and feature engineer
python prepare.py

# 5. Part B: Train model, balance classes, select threshold
python train.py

# 6. Part C: Validate policy retrieval and structured case-notes
python retrieval.py
python casenote.py

# 7. Part D: Start service and run smoke test
rm -f service.log
uvicorn app:app --port 8000 &
python smoke_test.py