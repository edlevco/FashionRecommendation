"""Step 4: the raw tables the ranker's features will come from.

    python features.py

Loads transactions, articles and customers, and prints the shape and first rows of each.
You fill in every TODO.
"""
import numpy as np
import pandas as pd
import torch


def load_transactions():
    # TODO 1: load data/hm/transactions_train.csv with columns t_dat, customer_id, article_id, price.
    #   Copy the trick from evaluate.py: usecols=[...], dtype={...}, parse_dates=[...].
    #   Decision: what dtype for price? (float32 is half the memory of the default float64.)

    tx = pd.read_csv("data/hm/transactions_train.csv", usecols=["t_dat", "customer_id", "article_id", "price"],
                        dtype={"customer_id": "string",
                               "article_id": "int32",
                               "price": "float32"},
                               parse_dates=["t_dat"])
    return tx


def load_articles():
    # TODO 2: load data/hm/articles.csv with only article_id, product_type_name, colour_group_name.
    #   article_id should match the transactions dtype, or joins later won't line up.
    
    articles = pd.read_csv("data/hm/articles.csv", usecols=["article_id", "product_type_name", "colour_group_name"],
                           dtype={"article_id": "int32",
                                  "product_type_name": "string",
                                  "colour_group_name": "string"})
    return articles


def load_customers():
    # TODO 3: load data/hm/customers.csv with only customer_id and age.
    #   customer_id should match the transactions dtype.
    #   Look at age after loading: are there missing values?

    customers = pd.read_csv("data/hm/customers.csv", usecols=["customer_id", "age"],
                            dtype={"customer_id": "string",
                                   "age": "float32"})  # float, not int: 15,861 ages are missing (NaN)
    return customers
 
    


def show(name, df):
    print(f"\n{name}: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(df.head())


def make_week(tx, cutoff):
    cutoff = pd.Timestamp(cutoff)
    end = cutoff + pd.Timedelta(days=7)                    # a date: 7 days after the cutoff
    history = tx[tx.t_dat < cutoff]                        # purchases before cutoff
    week = tx[(tx.t_dat >= cutoff) & (tx.t_dat < end)]     # >= keeps the cutoff day itself
    labels = week[["customer_id", "article_id"]].drop_duplicates()
    return history, labels

def make_candidates(history, labels, emb, row_of, n_popular=100, repurchase_days=28):
    last_day = history.t_dat.max()
    customers = pd.DataFrame({"customer_id": labels.customer_id.unique()})

    # source 1: popular (same for everyone)
    last_week = history[history.t_dat > last_day - pd.Timedelta(days=7)]
    top_items = last_week.article_id.value_counts().head(n_popular).index
    popular = customers.merge(pd.DataFrame({"article_id": top_items}), how="cross")

    # source 2: rebuys (personal)
    recent = history[history.t_dat > last_day - pd.Timedelta(days=repurchase_days)]
    rebuys = recent[recent.customer_id.isin(customers.customer_id)][["customer_id", "article_id"]]

    # source 3: NEW taste picks (personal)
    taste = taste_candidates(history, customers.customer_id, emb, row_of)

    cands = pd.concat([popular, rebuys, taste]).drop_duplicates().reset_index(drop=True)
    return cands


def add_popularity(cands, history, days):
    # How many times each article sold in the last `days` days of history
    last_day = history.t_dat.max()
    recent = history[history.t_dat > last_day - pd.Timedelta(days=days)]
    col = f"pop_{days}d"
    counts = recent.article_id.value_counts().rename(col)
    cands = cands.merge(counts, left_on="article_id", right_index=True, how="left")
    cands[col] = cands[col].fillna(0).astype("int32")      # rebuys that didn't sell recently -> 0
    return cands

def add_bought_before(cands, history):
    last_day = history.t_dat.max()
    mine = history[history.customer_id.isin(cands.customer_id)]
    summary = mine.groupby(["customer_id", "article_id"]).agg(
        times_bought=("t_dat", "count"),
        last_bought=("t_dat", "max"),
    ).reset_index()
    summary["days_since_bought"] = (last_day - summary.last_bought).dt.days
    summary = summary.drop(columns="last_bought")
    cands = cands.merge(summary, on=["customer_id", "article_id"], how="left")
    cands["times_bought"] = cands.times_bought.fillna(0)
    cands["days_since_bought"] = cands.days_since_bought.fillna(999)
    return cands

def add_customer(cands, history, customers):
    last_day = history.t_dat.max()
    mine = history[history.customer_id.isin(cands.customer_id)]

    # One row per customer: count, latest date, average price
    stats = mine.groupby("customer_id").agg(
        n_purchases=("t_dat", "count"),
        last_buy=("t_dat", "max"),
        avg_spend=("price", "mean"),
    ).reset_index()
    stats["days_since_last_buy"] = (last_day - stats.last_buy).dt.days
    stats = stats.drop(columns="last_buy")

    # Merge on customer only, so every row of that customer gets the same values
    cands = cands.merge(stats, on="customer_id", how="left")
    cands = cands.merge(customers, on="customer_id", how="left")   # adds age

    # Fill blanks
    cands["n_purchases"] = cands.n_purchases.fillna(0)
    cands["days_since_last_buy"] = cands.days_since_last_buy.fillna(999)
    cands["avg_spend"] = cands.avg_spend.fillna(cands.avg_spend.median())
    cands["age"] = cands.age.fillna(cands.age.median())
    return cands


def add_item(cands, history):
    last_day = history.t_dat.max()
    recent = history[history.t_dat > last_day - pd.Timedelta(days=30)]

    price = recent.groupby("article_id").price.mean().rename("item_price").reset_index()
    first = history.groupby("article_id").t_dat.min().rename("first_sold").reset_index()
    first["item_age_days"] = (last_day - first.first_sold).dt.days

    cands = cands.merge(price, on="article_id", how="left")
    cands = cands.merge(first[["article_id", "item_age_days"]], on="article_id", how="left")
    cands["item_price"] = cands.item_price.fillna(cands.item_price.median())
    cands["price_ratio"] = cands.item_price / cands.avg_spend
    return cands


def add_type_colour(cands, history, articles):
    mine = history[history.customer_id.isin(cands.customer_id)]
    mine = mine.merge(articles, on="article_id", how="left")   # each purchase gets its type + colour

    for col, name in [("product_type_name", "type_share"), ("colour_group_name", "colour_share")]:
        share = mine.groupby("customer_id")[col].value_counts(normalize=True).rename(name).reset_index()
        cands = cands.merge(articles[["article_id", col]], on="article_id", how="left")  # candidate's type
        cands = cands.merge(share, on=["customer_id", col], how="left")                 # look up the share
        cands[name] = cands[name].fillna(0)
        cands = cands.drop(columns=col)
    return cands


def load_item_vectors():
    # Image vectors + a lookup from article_id to its row in the matrix (same as recommend.py)
    emb = np.load("vectors/hm_emb.npy").astype(np.float32)
    ids = pd.read_parquet("vectors/hm_ids.parquet")
    row_of = dict(zip(ids.article_id, range(len(ids))))
    print(f"{emb.shape[0]:,} item vectors of size {emb.shape[1]}")
    return emb, ids.article_id.to_numpy(), row_of


def build_tastes(history, customer_ids, emb, row_of):
    # Each customer's taste = mean of their bought items' image vectors, scaled to length 1
    mine = history[history.customer_id.isin(customer_ids)]
    mine = mine[mine.article_id.isin(row_of)]                  # drop items with no image
    mine = mine.assign(row=mine.article_id.map(row_of))        # article_id -> row number
    rows_per_customer = mine.groupby("customer_id")["row"].apply(list)

    tastes, cust_row = [], {}    # taste vectors in order; customer_id -> their row in T
    for i, (cust, rows) in enumerate(rows_per_customer.items()):
        taste = emb[rows].mean(axis=0)
        taste = taste / np.linalg.norm(taste)
        tastes.append(taste)
        cust_row[cust] = i
    T = np.stack(tastes).astype(np.float32)
    return T, cust_row


def add_taste_sim(cands, history, emb, row_of):
    T, cust_row = build_tastes(history, cands.customer_id.unique(), emb, row_of)

    # C1: for every candidate row, look up the two row numbers
    c_idx = cands.customer_id.map(cust_row)   # customer -> row in T  (NaN if no history)
    i_idx = cands.article_id.map(row_of)      # item -> row in emb    (NaN if no image)

    # C2: keep only rows where both exist
    ok = c_idx.notna() & i_idx.notna()
    c = c_idx[ok].astype(int).to_numpy()
    i = i_idx[ok].astype(int).to_numpy()

    # C3: dot product in chunks of 500k rows (all at once would need ~24 GB)
    sims = np.zeros(len(c), dtype=np.float32)
    for start in range(0, len(c), 500_000):
        end = start + 500_000
        sims[start:end] = (T[c[start:end]] * emb[i[start:end]]).sum(axis=1)

    # D: 0 where we couldn't compute it
    cands["taste_sim"] = 0.0                  # 1. create the column
    cands.loc[ok, "taste_sim"] = sims         # 2. fill in the real values
    cands["has_taste"] = ok.astype(int)
    cands.loc[~ok, "taste_sim"] = cands.loc[ok, "taste_sim"].median()
    return cands

def taste_candidates(history, customer_ids, emb, row_of, n=50, pool_days=14, batch=4096):
    # For each customer: the n pool items whose image vector is closest to their taste.
    # Pool = items sold in the last `pool_days` days (that have an image vector).
    last_day = history.t_dat.max()
    recent = history[history.t_dat > last_day - pd.Timedelta(days=pool_days)]
    pool_ids = np.array([a for a in recent.article_id.unique() if a in row_of])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    P = torch.tensor(emb[[row_of[a] for a in pool_ids]]).to(device)   # pool items x 768

    T, cust_row = build_tastes(history, customer_ids, emb, row_of)
    custs = list(cust_row)                                   # dict order = row order in T

    top_rows = []
    for start in range(0, len(T), batch):                    # a few thousand customers at a time
        Tb = torch.from_numpy(T[start:start + batch]).to(device)
        scores = Tb @ P.T                                    # customers x pool: cosine sims
        top_rows.append(scores.topk(n, dim=1).indices.cpu().numpy())
    top_rows = np.concatenate(top_rows)                      # customers x n, positions in pool_ids

    return pd.DataFrame({
        "customer_id": pd.array(np.repeat(custs, n), dtype="string"),
        "article_id": pool_ids[top_rows.ravel()].astype("int32"),
    })


def add_labels(cands, labels):
    answers = labels[["customer_id", "article_id"]].copy()
    answers["label"] = 1                                  # every real purchase = 1
    cands = cands.merge(answers, on=["customer_id", "article_id"], how="left")
    cands["label"] = cands.label.fillna(0).astype("int8") # no match = didn't buy = 0
    return cands


def main():
    tx = load_transactions()
    articles = load_articles()
    customers = load_customers()
    show("transactions", tx)
    show("articles", articles)
    show("customers", customers)
    emb, all_ids, row_of = load_item_vectors()   # load once; make_candidates now needs them

    for cutoff in ["2020-09-02", "2020-09-09", "2020-09-16"]:   # train, validation, final test
        history, labels = make_week(tx, cutoff)
        print(cutoff, history.t_dat.max(), len(labels), labels.customer_id.nunique())

        cands = make_candidates(history, labels, emb, row_of)
        hits = labels.merge(cands, on=["customer_id", "article_id"], how="inner")
        print("candidates:", len(cands), "| recall:", round(len(hits) / len(labels), 3))
        #return                                             # TEMPORARY: stop here to just see recall

        cands = add_popularity(cands, history, days=7)
        cands = add_popularity(cands, history, days=30)
        print(cands.head())
        print(cands.describe())

        cands = add_bought_before(cands, history)
        print(cands[["times_bought", "days_since_bought"]].describe())

        cands = add_customer(cands, history, customers)
        cands = add_item(cands, history)
        cands = add_type_colour(cands, history, articles)
        print(cands.describe().T)

        cands = add_taste_sim(cands, history, emb, row_of)
        print(cands.taste_sim.describe())
        print("rows with taste_sim = 0:", (cands.taste_sim == 0).sum())

        cands = add_labels(cands, labels)
        print("positive rate:", cands.label.mean())
        cands.to_parquet(f"features_{cutoff}.parquet")


if __name__ == "__main__":
    main()
