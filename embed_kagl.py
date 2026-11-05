"""Turns every KAGL product photo into a FashionSigLIP vector.

Test on 2,000 images first:   python embed_kagl.py --limit 2000
Then run on everything:       python embed_kagl.py

The first run downloads KAGL (about 44k images) into data/kagl.
Writes vectors/kagl_emb.npy and vectors/kagl_meta.parquet (gender, category,
color, season, usage...) in the same order.
"""
import argparse
from pathlib import Path

import numpy as np
import open_clip
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

MODEL = "hf-hub:Marqo/marqo-fashionSigLIP"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="only embed the first N images")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    model, _, preprocess = open_clip.create_model_and_transforms(MODEL)
    model = model.to("cuda").eval()

    ds = load_dataset("Marqo/KAGL", split="data", cache_dir="data/kagl")
    if args.limit:
        ds = ds.select(range(args.limit))
    print(f"Embedding {len(ds):,} images")

    Path("vectors").mkdir(exist_ok=True)
    ds.remove_columns("image").to_pandas().to_parquet("vectors/kagl_meta.parquet")

    def collate(rows):
        return torch.stack([preprocess(r["image"].convert("RGB")) for r in rows])

    loader = DataLoader(ds, batch_size=args.batch_size, collate_fn=collate)
    chunks = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for x in tqdm(loader, unit="batch"):
            emb = model.encode_image(x.to("cuda", non_blocking=True), normalize=True)
            chunks.append(emb.float().cpu().numpy().astype(np.float16))

    vectors = np.concatenate(chunks)
    np.save("vectors/kagl_emb.npy", vectors)
    print(f"Saved {vectors.shape[0]:,} vectors of size {vectors.shape[1]}")


if __name__ == "__main__":
    main()
