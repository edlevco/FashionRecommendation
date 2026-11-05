"""Step 2: the scoreboard. Scores a recommender on H&M's real purchases.

    python evaluate.py

You fill in every TODO. The loading code is done for you (it's plumbing).
Run it after each TODO. It should run without errors even half-finished.
"""
import numpy as np
import pandas as pd

K = 12  # how many items we recommend per customer


def load_transactions():
    # 31M rows. Only load the columns we need, with small types, or it eats ~10 GB of RAM.
    df = pd.read_csv(
        "data/hm/transactions_train.csv",
        usecols=["t_dat", "customer_id", "article_id"],
        dtype={"customer_id": "string", "article_id": "int32"},
        parse_dates=["t_dat"],
    )
    print(f"{len(df):,} purchases from {df.t_dat.min().date()} to {df.t_dat.max().date()}")
    return df


def split_by_time(df):
    # TODO 1: pick ONE cutoff date for everyone.
    #   past   = purchases before the cutoff (what the recommender is allowed to see)
    #   future = purchases on/after the cutoff (the answers)
    #   Decision: future = Week A (Sept 9-15), one week, the same length H&M's competition used.
    #   Week B (Sept 16-22) is left out entirely: it's the final exam, scored once at the end.
    cutoff = pd.Timestamp("2020-09-09")
    end = cutoff + pd.Timedelta(days=7)  # Sept 16, so `< end` keeps Sept 15
    past = df[df.t_dat < cutoff]
    future = df[(df.t_dat >= cutoff) & (df.t_dat < end)]
    print("past ends", past.t_dat.max().date(), "| future", future.t_dat.min().date(), "to", future.t_dat.max().date())
    return past, future


def answers_per_customer(future):
    # TODO 2: return a dict {customer_id: set of article_ids they bought in the future window}.
    return future.groupby("customer_id").article_id.apply(set).to_dict()


def ndcg_weight(slot):
    # NDCG's slot weight: drops fast right away (1.0, 0.63, 0.5, 0.43...).
    return 1 / np.log2(slot + 1)


def quad_weight(slot):
    # My slot weight: a downward parabola with its peak at slot 1, so the first few slots
    # stay near 1 and then attention fades. c = 1/144 puts the zero at slot 13, just past K.
    return 1 - (slot - 1) ** 2 / 144

def get_ideal(bought, weight, k):
    ideal_score = 0
    items = min(k, len(bought)) + 1
    for i in range(1,items):
        ideal_score += weight(i)

    return ideal_score



def ndcg_at_k(recommended, bought, k=K, weight=ndcg_weight):
    # TODO 3: YOUR formula. recommended = list of article_ids in slot order, bought = set.
    #   score = sum of slot weights for each hit / min(len(bought), k)
    #   weight = a function that takes a slot (counting from 1) and returns its weight.
    #   Note: real NDCG divides by the best possible weighted score, not just a count.
    #   Do yours first, then ask yourself if a perfect list scores exactly 1.0.
    total = 0.0
    ideal = get_ideal(bought, weight, k)
    for slot, item in enumerate(recommended[:k], start=1):

        if item in bought:
            total += weight(slot)


    return total / ideal


def popularity_baseline(past, days=None, k=K):
    # TODO 4: the lazy recommender. Return the K most bought article_ids, as a list.
    #   Decision: most popular over ALL past purchases, or only the last week? Try both.
    if days == None:
        recent = past

    else:
        start   = past.t_dat.max() - pd.Timedelta(days=days)
        recent = past[past.t_dat > start]
    top_ids = recent.article_id.value_counts().head(k).index.tolist()
    return top_ids
  


def evaluate(recommend_fn, answers, sample=None, weight=ndcg_weight):
    # Averages your score over every customer who bought something in the future window.
    customers = list(answers)
    if sample:
        rng = np.random.default_rng(0)
        customers = list(rng.choice(customers, size=min(sample, len(customers)), replace=False))
    scores = [ndcg_at_k(recommend_fn(c), answers[c], weight=weight) for c in customers]
    return float(np.mean(scores))


def main():
    df = load_transactions()
    past, future = split_by_time(df)
    answers = answers_per_customer(future)
    print(f"{len(answers):,} customers bought something in the future window")

    top = popularity_baseline(past)
    for name, weight in [("NDCG", ndcg_weight), ("Quad", quad_weight)]:
        score = evaluate(lambda customer: top, answers, weight=weight)
        print(f"Popularity baseline {name}@{K}: {score:.4f}")

    # TODO 5 (later): score the baseline separately for new customers (nothing in `past`)
    #   and returning ones. Which group is harder?


if __name__ == "__main__":
    main()
