r"""Smoke test: checks the GPU, loads FashionSigLIP, and labels one H&M photo.

Run from C:\stylematch with the venv active:
    python smoke_test.py
"""
from pathlib import Path

import open_clip
import pandas as pd
import torch
from PIL import Image

MODEL = "hf-hub:Marqo/marqo-fashionSigLIP"
LABELS = ["a dress", "trousers", "a t-shirt", "a sweater", "a jacket",
          "shoes", "a bag", "underwear", "a skirt", "a hat"]


def main():
    print("GPU available:", torch.cuda.is_available())
    if not torch.cuda.is_available():
        print("Stop: PyTorch can't see your GPU. Reinstall torch with the CUDA command from pytorch.org.")
        return

    model, _, preprocess = open_clip.create_model_and_transforms(MODEL)
    tokenizer = open_clip.get_tokenizer(MODEL)
    model = model.to("cuda").eval()

    path = next(Path("data/hm/images").rglob("*.jpg"))
    print("Test image:", path)

    image = preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to("cuda")
    text = tokenizer(LABELS).to("cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        img = model.encode_image(image, normalize=True)
        txt = model.encode_text(text, normalize=True)
        probs = (100 * img @ txt.T).softmax(dim=-1)[0].float().cpu()

    print("Vector size:", img.shape[1])
    print("Model's guesses:")
    for label, p in sorted(zip(LABELS, probs.tolist()), key=lambda x: -x[1])[:5]:
        print(f"  {p:6.1%}  {label}")

    arts = pd.read_csv("data/hm/articles.csv", usecols=["article_id", "prod_name", "product_type_name"])
    row = arts[arts.article_id == int(path.stem)]
    if len(row):
        print("H&M says:", row.iloc[0].prod_name, "/", row.iloc[0].product_type_name)
    print("If the top guess matches what H&M says, everything works.")


if __name__ == "__main__":
    main()
