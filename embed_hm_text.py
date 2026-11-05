"""Turns each H&M product description into a FashionSigLIP text vector.

    python embed_hm_text.py

Writes vectors/hm_text_emb.npy and vectors/hm_text_ids.parquet in the same order.
"""
from pathlib import Path

import numpy as np
import open_clip
import pandas as pd
import torch
from tqdm import tqdm

MODEL = "hf-hub:Marqo/marqo-fashionSigLIP"


def main():
    model, _, _ = open_clip.create_model_and_transforms(MODEL)
    tokenizer = open_clip.get_tokenizer(MODEL)
    model = model.to("cuda").eval()

    arts = pd.read_csv("data/hm/articles.csv")
    texts = (arts.prod_name.fillna("") + ", " + arts.colour_group_name.fillna("") + " "
             + arts.product_type_name.fillna("") + ". " + arts.detail_desc.fillna("")).tolist()

    chunks = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for i in tqdm(range(0, len(texts), 256), unit="batch"):
            tokens = tokenizer(texts[i:i + 256]).to("cuda")
            chunks.append(model.encode_text(tokens, normalize=True).float().cpu().numpy().astype(np.float16))

    Path("vectors").mkdir(exist_ok=True)
    np.save("vectors/hm_text_emb.npy", np.concatenate(chunks))
    arts[["article_id"]].to_parquet("vectors/hm_text_ids.parquet")
    print(f"Saved {len(texts):,} text vectors")


if __name__ == "__main__":
    main()
