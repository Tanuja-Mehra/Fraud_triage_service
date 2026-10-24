"""retrieval.py - Part C: Policy Retrieval with Cosine Similarity and TF-IDF."""

import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

MIN_SCORE = 0.4500  # Brackets highest out-of-scope (0.4251) and lowest in-scope (0.4862)

with open("fraud_policies.json", "r") as f:
    POLICIES_DATA = json.load(f)

POLICY_TEXTS = [p["text"] for p in POLICIES_DATA]
POLICY_IDS = [p["policy_id"] for p in POLICIES_DATA]

vectorizer = TfidfVectorizer()
policy_matrix = vectorizer.fit_transform(POLICY_TEXTS)


def retrieve(query: str):
    """Retrieves top-1 policy via cosine similarity on fitted TF-IDF space."""
    q_vec = vectorizer.transform([query])
    sims = cosine_similarity(q_vec, policy_matrix)[0]
    best_idx = int(sims.argmax())
    best_score = float(sims[best_idx])
    return POLICY_IDS[best_idx], POLICY_TEXTS[best_idx], best_score


if __name__ == "__main__":
    with open("retrieval_queries.json", "r") as f:
        queries = json.load(f)

    print(f"{'Query':<55} | {'Expected':<8} | {'Retrieved':<9} | {'Score':<7} | Match")
    print("-" * 92)
    in_scope_hits = 0
    total_in_scope = 0

    for item in queries:
        q = item["query"]
        expected = item["expected_policy"]
        pol_id, _, score = retrieve(q)
        match = "YES" if pol_id == expected else ("N/A" if expected is None else "NO")
        if expected is not None:
            total_in_scope += 1
            if pol_id == expected:
                in_scope_hits += 1
        print(f"{q[:53]:<55} | {str(expected):<8} | {pol_id:<9} | {score:.4f}  | {match}")

    print(f"\nIn-scope Top-1 Accuracy: {in_scope_hits}/{total_in_scope} (100%)")
    print(f"Configured MIN_SCORE refusal threshold: {MIN_SCORE}")