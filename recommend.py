"""Step 3: personal recommenders, scored with your grader from evaluate.py.

    python recommend.py

Plan: build the SIMPLE version first (one average taste vector per customer),
score it, then build your clump version and see if it beats it.
The loading code is done for you. You fill in every TODO.
"""
import numpy as np
import pandas as pd

from evaluate import K, answers_per_customer, evaluate, load_transactions, popularity_baseline, split_by_time


def load_item_vectors():
    # Plumbing: image vectors + a lookup from article_id to its row in the matrix.
    emb = np.load("vectors/hm_emb.npy").astype(np.float32)
    ids = pd.read_parquet("vectors/hm_ids.parquet")
    row_of = dict(zip(ids.article_id, range(len(ids))))
    print(f"{emb.shape[0]:,} item vectors of size {emb.shape[1]}")
    return emb, ids.article_id.to_numpy(), row_of


def customer_histories(past, answers):
    # TODO A: return {customer_id: list of article_ids they bought, OLDEST first}.
    #   Same groupby idea as TODO 2, but keep order and duplicates? Your call.
    #   Hint: sort past by t_dat first, then groupby(...).article_id.apply(list).
    #   Speed tip: only keep customers who are in `answers` (they're the ones we score).

     scored = past[past.customer_id.isin(answers)]
     scored = scored.sort_values("t_dat")
     return scored.groupby("customer_id").article_id.apply(list).to_dict()

  


def candidate_pool(past, days):
    # TODO B: which items are we allowed to recommend?
    #   H&M has ~105k items, but many were no longer for sale in Sept 2020.
    #   Recommending them can never be a hit. Return article_ids sold in the last `days` days.
    #   Decision: how many days? (You already measured recency matters.)

    start = past.t_dat.max() - pd.Timedelta(days=days)
    recent = past[past.t_dat > start]
    return recent.article_id.unique()



def top_k_near(taste, pool_rows, pool_ids, emb, exclude, k=K):
    # TODO C: return the k article_ids in the pool closest to `taste`, skipping `exclude`.
    #   scores = emb[pool_rows] @ taste        -> one similarity per pool item
    #   order  = np.argsort(-scores)           -> best first
    #   Walk `order`, skip ids in `exclude`, stop at k.

        
    matrix = emb[pool_rows]
    scores = matrix @ taste
    order = np.argsort(-scores)
    result = []
    for article_id in pool_ids[order]:
        if article_id in exclude:
            continue          # skip it
        result.append(article_id)
        if len(result) == k:
            break
    return result


def mean_taste(history_ids, emb, row_of):
    # TODO D: the simple taste vector.
    #   Look up each bought item's vector, average them, renormalize to length 1.
    #   Skip ids missing from row_of (no image). Return None if nothing is left.

    rows = [row_of[a] for a in history_ids if a in row_of]
    if not rows:
        return None
    v = emb[rows].mean(axis=0)
    return v / np.linalg.norm(v)


def main():
    df = load_transactions()
    past, future = split_by_time(df)
    answers = answers_per_customer(future)
    emb, all_ids, row_of = load_item_vectors()

    histories = customer_histories(past, answers)
    popular = popularity_baseline(past, days=14, k=1000)
    pool_ids = np.array([a for a in popular if a in row_of])
    pool_rows = np.array([row_of[a] for a in pool_ids])
    print(f"Candidate pool: {len(pool_ids):,} items")

    fallback = popularity_baseline(past, days=14)  # for customers with no history

    def mean_recommender(customer):
        history = histories.get(customer, [])
        taste = mean_taste(history, emb, row_of)
        if taste is None:
            return fallback
        return top_k_near(taste, pool_rows, pool_ids, emb, exclude=set())

    # Start on a sample so each run takes about a minute, then run on everyone.
    print(f"Mean taste NDCG@{K}:", round(evaluate(mean_recommender, answers), 4))
    print(f"Popularity (14d) on the same sample:", round(evaluate(lambda c: fallback, answers), 4))

    # TODO E (next): clump_recommender, your design. Score it the same way.


if __name__ == "__main__":
    main()
