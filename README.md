# FashionRecommendation: a two-stage fashion recommender in PyTorch

Predicts which H&M items each customer will buy next week. Candidates come from popularity, repurchases, and image-embedding retrieval, and a PyTorch MLP ranks them.

**Result on a held-out test week (Sept 16–22, 2020, 68,984 customers):**

| Model | NDCG@12 |
|---|---|
| Popularity baseline (top 12 sellers, last 14 days) | 0.0140 |
| **Two-stage ranker (this repo)** | **0.0422 (3.0x)** |

The test week was untouched until the very end and scored exactly once.

## Data

[H&M Personalized Fashion Recommendations](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations) (Kaggle). It has 31.8M transactions from Sept 2018 to Sept 2020, plus 1.37M customers and 105k articles with images. The data is for non-commercial use, so it is not included in this repo.

## Approach

```
purchase history ──► candidates (~150 per customer) ──► 15 features ──► MLP ranker ──► top 12
```

### 1. Image embeddings

Every article image is embedded with [Marqo-FashionSigLIP](https://huggingface.co/Marqo/marqo-fashionSigLIP), giving 105,100 × 768 L2-normalized vectors. A customer's **taste vector** is the normalized mean of the vectors of their past purchases.

### 2. Candidate generation

Scoring all 105k items for every customer is too slow, so each customer gets a shortlist from three sources:

| Source | Per customer | Recall (train week) |
|---|---|---|
| Top 100 sellers, last 7 days | 100 | |
| + own purchases, last 28 days | a few | 0.124 |
| + 50 nearest items to taste vector (pool: items sold in last 14 days) | 50 | 0.140 |

Taste retrieval is a batched GPU matrix multiply (`T @ P.T`) plus `torch.topk`, in chunks of 4,096 customers. Restricting the pool to recently sold items nearly doubled what this source added (+0.9 to +1.6 recall points) at the same candidate count.

### 3. Features (15 per customer–item pair)

- **Popularity:** sales last 7 and 30 days
- **Repurchase:** times bought, days since last bought
- **Customer:** number of purchases, days since last purchase, average spend, age
- **Item:** price, days since first sale, price relative to the customer's average spend
- **Style:** the customer's share of purchases in this item's product type and colour
- **Visual taste:** cosine similarity between the item's embedding and the customer's taste vector, plus a has-taste flag

Long-tailed counts are `log1p`-transformed. All features are z-scored using training-week statistics only.

### 4. Ranker

A PyTorch MLP (15 → 64 → 32 → 1, ReLU, 3,137 parameters), trained with `BCEWithLogitsLoss` and Adam (lr 1e-3) for 5 epochs at batch size 4,096. The label is 1 if the customer bought that item in the label week. With a positive rate of ~0.3%, the training set keeps all positives and a random 10% of negatives (~1.1M rows). Ranking only depends on score order, so downsampling doesn't hurt.

## Evaluation setup

Time-based split with no leakage. Features only ever use data before each cutoff.

| Week | Role |
|---|---|
| Sept 2–8 | training labels (history before Sept 2) |
| Sept 9–15 | validation: all tuning and ablation |
| Sept 16–22 | final test, scored once |

The metric is NDCG@12 against **all** of a customer's purchases that week, not only those inside the candidates, averaged over every customer who bought something.

## Results over time (validation week)

| Version | NDCG@12 |
|---|---|
| Popularity, all time | 0.0063 |
| Popularity, last 14 days (best window) | 0.0132 |
| Mean taste vector only (pure visual similarity) | 0.0126 |
| Ranker, popular + repurchase candidates | 0.0394 |
| Ranker, + taste candidates | 0.0408 |

Visual similarity on its own lost to bestsellers. Used as one feature inside a trained ranker, it became one of the strongest signals.

## Ablation: what does the model actually use?

Retrain without one feature at a time, then measure the validation NDCG drop. The seed-to-seed noise was about 0.0001.

| Feature removed | NDCG drop (popular + repurchase candidates) | NDCG drop (+ taste candidates) |
|---|---|---|
| days since bought (this item) | 0.0019 | 0.0013 |
| taste similarity | 0.0015 | 0.0007 |
| item age | 0.0012 | ~0 |
| popularity 7d / 30d | 0.0004 / 0.0003 | ~0 |

**Findings:**

- **Repurchase recency is the strongest signal.** H&M customers rebuy basics, and timing matters.
- **A signal used to select candidates loses ranking power.** Popularity barely helps rank because every candidate is already popular. Item age stopped mattering once taste candidates were restricted to recently sold items.
- **Correlated features mask each other** in leave-one-out ablation (for example times bought vs days since bought).

## Files

| File | What it does |
|---|---|
| `smoke_test.py` | checks the GPU, loads FashionSigLIP, labels one H&M photo |
| `embed_hm.py` | embeds all article images with FashionSigLIP, writing `vectors/hm_emb.npy` and `vectors/hm_ids.parquet` |
| `embed_hm_text.py` | embeds article descriptions as text vectors (exploration, not used by the ranker) |
| `embed_kagl.py` | embeds the KAGL product photo dataset (exploration, not used by the ranker) |
| `evaluate.py` | time split, NDCG@12, popularity baseline |
| `recommend.py` | first baseline: mean-taste-vector recommender |
| `features.py` | candidates + 15 features + labels per week, written to `features_<cutoff>.parquet` |
| `train.py` | scaling, downsampling, MLP training, validation NDCG, ablation, final test |

## Run it

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows; use source .venv/bin/activate elsewhere
pip install torch --index-url https://download.pytorch.org/whl/cu126   # CUDA build
pip install pandas numpy pyarrow open_clip_torch pillow tqdm
# unzip the Kaggle files into data/hm/ (articles.csv, customers.csv, transactions_train.csv, images/)
python smoke_test.py      # check the GPU and model load
python embed_hm.py        # GPU recommended
python evaluate.py        # popularity baselines
python features.py        # builds features_2020-09-02 / -09 / -16 parquet files (~13 min each, uses the GPU)
python train.py           # trains, validates, final test (the ablation loop is commented out; uncomment to rerun it)
```

## Next

- Multi-interest taste (several taste vectors per customer, clustered from purchases) as a candidate source
- Gradient-boosted ranker (LightGBM) as a comparison
- Using this pipeline in a swipe-to-shop app, retrained on swipes
