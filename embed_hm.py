"""Turns every H&M product photo into a FashionSigLIP vector.

Test on 2,000 images first:   python embed_hm.py --limit 2000
Then run on everything:       python embed_hm.py

Writes vectors/hm_emb.npy (one row per image) and vectors/hm_ids.parquet
(article_id, image path, and whether the image loaded) in the same order.
"""
import argparse
from pathlib import Path

import numpy as np
import open_clip
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

MODEL = "hf-hub:Marqo/marqo-fashionSigLIP"


class Images(Dataset):
    def __init__(self, paths, preprocess):
        self.paths, self.preprocess = paths, preprocess
        self.blank = preprocess(Image.new("RGB", (256, 256)))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        try:
            return self.preprocess(Image.open(self.paths[i]).convert("RGB")), i, True
        except Exception:
            return self.blank, i, False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="only embed the first N images")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    model, _, preprocess = open_clip.create_model_and_transforms(MODEL)
    model = model.to("cuda").eval()

    paths = sorted(Path("data/hm/images").rglob("*.jpg"))
    if args.limit:
        paths = paths[:args.limit]
    print(f"Embedding {len(paths):,} images")

    data = Images(paths, preprocess)
    loader = DataLoader(data, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True)

    with torch.no_grad():
        dim = model.encode_image(data.blank.unsqueeze(0).to("cuda")).shape[1]
    vectors = np.zeros((len(paths), dim), dtype=np.float16)
    ok = np.zeros(len(paths), dtype=bool)

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for x, idx, good in tqdm(loader, unit="batch"):
            emb = model.encode_image(x.to("cuda", non_blocking=True), normalize=True)
            vectors[idx.numpy()] = emb.float().cpu().numpy()
            ok[idx.numpy()] = good.numpy()

    Path("vectors").mkdir(exist_ok=True)
    np.save("vectors/hm_emb.npy", vectors)
    pd.DataFrame({
        "article_id": [int(p.stem) for p in paths],
        "path": [str(p) for p in paths],
        "ok": ok,
    }).to_parquet("vectors/hm_ids.parquet")
    print(f"Saved {vectors.shape[0]:,} vectors of size {dim}. Images that failed to load: {(~ok).sum()}")


if __name__ == "__main__":
    main()
