import argparse
import os
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPModel, CLIPProcessor

ImageFile.LOAD_TRUNCATED_IMAGES = True

DEFAULT_MODEL = "openai/clip-vit-base-patch32"


class Pairs(Dataset):
    def __init__(self, root, frame, text_column):
        self.root = Path(root)
        self.paths = frame["image_path"].tolist()
        self.texts = frame[text_column].tolist()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        image = Image.open(self.root / self.paths[i]).convert("RGB")
        return image, self.texts[i]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/crisismmd_clean.csv")
    parser.add_argument("--root", default="dataset/CrisisMMD_v2.0")
    parser.add_argument("--text-column", default="text_clean")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--out", default="embeddings")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def load_frame(csv_path, text_column, limit):
    frame = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    if text_column not in frame.columns:
        raise SystemExit(f"column {text_column} not found in {csv_path}")
    if limit:
        frame = frame.sample(n=min(limit, len(frame)), random_state=0).reset_index(drop=True)
    empty = int((frame[text_column] == "").sum())
    print(f"{len(frame)} rows, embedding column {text_column}, {empty} with empty text")
    return frame


def load_clip(model_name, device):
    model = CLIPModel.from_pretrained(model_name).to(device).eval()
    processor = CLIPProcessor.from_pretrained(model_name)
    return model, processor


def collate(batch, processor):
    images, texts = zip(*batch)
    return processor(
        text=list(texts),
        images=list(images),
        return_tensors="pt",
        padding=True,
        truncation=True,
    )


def embed_all(model, processor, loader, device):
    image_chunks, text_chunks = [], []
    for step, inputs in enumerate(loader):
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            image_emb = model.get_image_features(pixel_values=inputs["pixel_values"])
            text_emb = model.get_text_features(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )
        image_chunks.append(torch.nn.functional.normalize(image_emb, dim=-1).cpu().numpy())
        text_chunks.append(torch.nn.functional.normalize(text_emb, dim=-1).cpu().numpy())
        print(f"  batch {step + 1}/{len(loader)}", flush=True)
    return np.concatenate(image_chunks), np.concatenate(text_chunks)


def save_outputs(out_dir, text_column, frame, image_emb, text_emb):
    path = out_dir / f"crisismmd_{text_column}.npz"
    np.savez_compressed(
        path,
        image_emb=image_emb,
        text_emb=text_emb,
        image_id=np.array(frame["image_id"].tolist()),
    )
    print(f"saved {path} {image_emb.shape} {text_emb.shape}")


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device {device}")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    frame = load_frame(args.csv, args.text_column, args.limit)
    model, processor = load_clip(args.model, device)
    workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 4))
    loader = DataLoader(
        Pairs(args.root, frame, args.text_column),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=workers,
        collate_fn=partial(collate, processor=processor),
    )
    image_emb, text_emb = embed_all(model, processor, loader, device)
    save_outputs(out_dir, args.text_column, frame, image_emb, text_emb)


if __name__ == "__main__":
    main()