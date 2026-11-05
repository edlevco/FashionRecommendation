import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from evaluate import load_transactions, split_by_time, answers_per_customer, ndcg_at_k

SKIP = ["customer_id", "article_id", "label"]
LOG_COLS = ["pop_7d", "pop_30d", "times_bought", "days_since_bought",
            "n_purchases", "days_since_last_buy", "item_age_days", "price_ratio"]
DEVICE = "cuda"


def load_week(path):                                   # unchanged
    df = pd.read_parquet(path)
    feature_cols = [c for c in df.columns if c not in SKIP]
    df[LOG_COLS] = np.log1p(df[LOG_COLS])
    X = torch.tensor(df[feature_cols].to_numpy(dtype="float32"))
    y = torch.tensor(df["label"].to_numpy(dtype="float32"))
    return df, X, y, feature_cols


def train_model(X_tr, y_tr, n_features, seed=0):       # = your model + loop lines
    torch.manual_seed(seed)
    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=4096, shuffle=True)
    model = nn.Sequential(
        nn.Linear(n_features, 64), nn.ReLU(),           # 15 -> n_features
        nn.Linear(64, 32), nn.ReLU(),
        nn.Linear(32, 1),
    ).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(5):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            loss = loss_fn(model(xb).squeeze(1), yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model


def validation_ndcg(model, X_v, df_v, answers):        # = your Piece A + Piece B lines
    model.eval()
    all_scores = []
    with torch.no_grad():
        for chunk in X_v.split(1_000_000):
            all_scores.append(model(chunk.to(DEVICE)).squeeze(1).cpu())
    df_v["score"] = torch.cat(all_scores).numpy()
    top = (df_v.sort_values(["customer_id", "score"], ascending=[True, False])
               .groupby("customer_id").head(12))
    recs = top.groupby("customer_id").article_id.apply(list).to_dict()
    return float(np.mean([ndcg_at_k(recs.get(c, []), b) for c, b in answers.items()]))


def main():
    # ---------- ONCE ----------
    df, X, y, feature_cols = load_week("features_2020-09-02.parquet")
    mean, std = X.mean(dim=0), X.std(dim=0)
    X = (X - mean) / std
    torch.manual_seed(0)
    keep_rows = (y == 1) | (torch.rand(len(y)) < 0.10)
    X_tr, y_tr = X[keep_rows], y[keep_rows]

    df_v, X_v, _, _ = load_week("features_2020-09-09.parquet")
    X_v = (X_v - mean) / std

    tx = load_transactions()
    past, future = split_by_time(tx)
    answers = answers_per_customer(future)
    n = len(feature_cols)

    # ---------- NOISE CHECK: same model, two seeds ----------
    base = validation_ndcg(train_model(X_tr, y_tr, n, seed=0), X_v, df_v, answers)
    again = validation_ndcg(train_model(X_tr, y_tr, n, seed=1), X_v, df_v, answers)
    print(f"ALL features: {base:.4f}  (seed 1: {again:.4f}, noise ~{abs(base - again):.4f})")

    # ---------- ABLATION: 15 runs ----------
    # results = []
    # for i, name in enumerate(feature_cols):
    #     keep = [j for j in range(n) if j != i]
    #     model = train_model(X_tr[:, keep], y_tr, n - 1)
    #     s = validation_ndcg(model, X_v[:, keep], df_v, answers)
    #     results.append((name, s, base - s))
    #     print(f"without {name:20s} {s:.4f}   drop {base - s:+.4f}")
    #
    # print("\nMOST IMPORTANT FIRST:")
    # for name, s, drop in sorted(results, key=lambda r: -r[2]):
    #     print(f"{name:20s} drop {drop:+.4f}")

    # ---------- FINAL EXAM: Sept 16-22, run ONCE ----------
    from evaluate import popularity_baseline
    FINAL = pd.Timestamp("2020-09-16")
    fut = tx[(tx.t_dat >= FINAL) & (tx.t_dat < FINAL + pd.Timedelta(days=7))]
    answers_f = answers_per_customer(fut)

    df_f, X_f, _, _ = load_week("features_2020-09-16.parquet")
    X_f = (X_f - mean) / std                                   # training's ruler, as always
    model = train_model(X_tr, y_tr, n, seed=0)
    ranker = validation_ndcg(model, X_f, df_f, answers_f)

    pop = popularity_baseline(tx[tx.t_dat < FINAL], days=14)   # bestsellers, same week
    pop_score = np.mean([ndcg_at_k(pop, b) for b in answers_f.values()])

    print(f"FINAL  ranker {ranker:.4f} | popularity {pop_score:.4f} | {ranker / pop_score:.1f}x")


if __name__ == "__main__":
    main()