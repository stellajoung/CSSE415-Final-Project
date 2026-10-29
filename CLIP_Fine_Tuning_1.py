"""
Fine-tune a (layer-truncated) CLIP fusion classifier on CrisisMMD Task 1:
informative vs. not informative.

Expects a CSV with columns:
    image_path   path relative to --root
    text_clean   tweet text (name set with --text-column)
    text_info    Task 1 label, "informative" / "not_informative"  (--label-column)
    split        "train" / "dev" (or "val") / "test"  (--split-column)
    (optional) image_info  if --image-label-column is set, only rows where it equals the label are kept

Example:
    python finetune_clip_fusion.py --csv data/crisismmd_clean.csv \
        --root dataset/CrisisMMD_v2.0 --n-vision 6 --n-text 6 --epochs 5 --out runs/v6_t6

Outputs (in --out):
    metrics.json  all numbers from the run (merged across runs by compare_runs.py)
    best.pt       weights of the best epoch (chosen on the dev set)
"""
import argparse
import copy
import json
import os
import random
import time
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageFile
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset
from transformers import (CLIPConfig, CLIPModel, CLIPProcessor,
                          get_linear_schedule_with_warmup)

# Don't crash on partially downloaded / truncated image files.
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Maps the label strings in the CSV to class ids (0 = not informative, 1 = informative).
LABEL_MAP = {"not_informative": 0, "non_informative": 0, "not informative": 0,
             "non-informative": 0, "informative": 1}


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
class Pairs(Dataset):
    """Yields (PIL image, tweet text, integer label) for each image-text pair."""

    def __init__(self, root, frame, text_column):
        self.root = Path(root)                       # folder the image paths are relative to
        self.paths = frame["image_path"].tolist()
        self.texts = frame[text_column].tolist()
        self.labels = frame["y"].tolist()            # integer labels created in load_splits

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        # Images are opened lazily (one at a time) so the whole dataset isn't held in memory.
        image = Image.open(self.root / self.paths[i])
        if image.mode in ("P", "LA") or "transparency" in image.info:
            # Palette/transparent images: go through RGBA and flatten onto a white background.
            # Avoids PIL's "Palette images with Transparency" warning.
            image = image.convert("RGBA")
            background = Image.new("RGB", image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[-1])
            image = background
        else:
            image = image.convert("RGB")
        return image, self.texts[i], self.labels[i]


def collate(batch, processor):
    """Turns a list of (image, text, label) into one batch of model-ready tensors."""
    images, texts, labels = zip(*batch)
    # The CLIP processor resizes/normalizes the images and tokenizes the texts
    # (padding to the longest text in the batch, truncating at CLIP's 77-token limit).
    inputs = processor(text=list(texts), images=list(images), return_tensors="pt",
                       padding=True, truncation=True, max_length=77)
    inputs["labels"] = torch.tensor(labels, dtype=torch.long)
    return inputs


def load_splits(args):
    """Reads the CSV, converts labels to ints, and splits into train/dev/test frames."""
    frame = pd.read_csv(args.csv, dtype=str, keep_default_na=False)

    needed = [args.text_column, args.label_column, args.split_column, "image_path"]
    if args.image_label_column:
        needed.append(args.image_label_column)
    for col in needed:
        if col not in frame.columns:
            raise SystemExit(f"column '{col}' not found in {args.csv}; have {list(frame.columns)}")

    # Optional: keep only "agreed" pairs where the text label equals the image label.
    # Skip this (the default) if your CSV already contains only label-agreed rows.
    if args.image_label_column:
        text_lab = frame[args.label_column].str.strip().str.lower()
        image_lab = frame[args.image_label_column].str.strip().str.lower()
        before = len(frame)
        frame = frame[text_lab == image_lab].reset_index(drop=True)
        print(f"kept {len(frame)}/{before} rows where {args.label_column} == {args.image_label_column}")

    # Convert label strings to 0/1; stop if there is a label we don't recognize.
    labels = frame[args.label_column].str.strip().str.lower()
    unknown = set(labels) - set(LABEL_MAP)
    if unknown:
        raise SystemExit(f"unrecognized labels {unknown}; edit LABEL_MAP")
    frame["y"] = labels.map(LABEL_MAP)

    # Normalize split names ("val" -> "dev") and separate the three splits.
    frame["split_norm"] = frame[args.split_column].str.strip().str.lower().replace({"val": "dev"})
    splits = {s: frame[frame["split_norm"] == s].reset_index(drop=True) for s in ("train", "dev", "test")}

    for s, f in splits.items():
        print(f"{s}: {len(f)} rows, label counts {f['y'].value_counts().sort_index().to_dict()}")
        if len(f) == 0:
            raise SystemExit(f"no rows for split '{s}'")

    # Optional: shrink the training set for quick debugging runs.
    if args.limit:
        splits["train"] = splits["train"].sample(n=min(args.limit, len(splits["train"])),
                                                 random_state=0).reset_index(drop=True)
    return splits


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
def load_clip(model_name, n_vision, n_text):
    """Loads CLIP, optionally keeping only the first N layers of each tower."""
    config = CLIPConfig.from_pretrained(model_name)

    # Guard: asking for MORE layers than the checkpoint has would silently add randomly
    # initialized layers (huge memory use, and not pretrained). Only shrinking is allowed.
    for tower, n, full in (("vision", n_vision, config.vision_config.num_hidden_layers),
                           ("text", n_text, config.text_config.num_hidden_layers)):
        if n > full:
            raise SystemExit(f"--n-{tower}={n} but the pretrained {tower} tower has only {full} "
                             f"layers. Use a value <= {full}, or 0 to keep all layers.")

    if n_vision:
        config.vision_config.num_hidden_layers = n_vision   # shrink the image encoder
    if n_text:
        config.text_config.num_hidden_layers = n_text       # shrink the text encoder
    # Weights for the removed layers are simply not loaded.
    model = CLIPModel.from_pretrained(model_name, config=config)
    processor = CLIPProcessor.from_pretrained(model_name)
    return model, processor


# Number of embedding-sized vectors each fusion mode produces.
FUSION_PARTS = {"concat2": 2, "concat3": 3, "concat_diff": 3, "concat4": 4, "prod": 1}


class CLIPFusionClassifier(nn.Module):
    """CLIP image + text embeddings -> fused vector -> head -> 2 classes."""

    def __init__(self, clip, hidden=512, dropout=0.3, fusion="concat3", freeze_encoders=False):
        super().__init__()
        self.clip = clip
        self.fusion = fusion
        self.freeze_encoders = freeze_encoders
        dim = clip.config.projection_dim
        in_dim = FUSION_PARTS[fusion] * dim

        if hidden == 0:
            # Linear probe on the fused vector.
            self.head = nn.Linear(in_dim, 2)
        else:
            self.head = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, 2),
            )

        if freeze_encoders:
            for p in self.clip.parameters():
                p.requires_grad = False

    def encode(self, pixel_values, input_ids, attention_mask):
        img = self.clip.get_image_features(pixel_values=pixel_values)
        txt = self.clip.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
        return (nn.functional.normalize(img, dim=-1), nn.functional.normalize(txt, dim=-1))

    def fuse(self, img, txt):
        if self.fusion == "concat2":
            return torch.cat([img, txt], dim=-1)
        if self.fusion == "concat3":
            return torch.cat([img, txt, img * txt], dim=-1)
        if self.fusion == "concat_diff":
            return torch.cat([img, txt, (img - txt).abs()], dim=-1)
        if self.fusion == "concat4":
            return torch.cat([img, txt, img * txt, (img - txt).abs()], dim=-1)
        if self.fusion == "prod":
            return img * txt
        raise ValueError(self.fusion)

    def forward(self, pixel_values, input_ids, attention_mask):
        if self.freeze_encoders:
            self.clip.eval()
            with torch.no_grad():
                img, txt = self.encode(pixel_values, input_ids, attention_mask)
        else:
            img, txt = self.encode(pixel_values, input_ids, attention_mask)
        return self.head(self.fuse(img, txt))


# --------------------------------------------------------------------------
# Train / eval
# --------------------------------------------------------------------------
def to_device(batch, device):
    """Moves every tensor in a batch dict to the GPU/CPU."""
    return {k: v.to(device) for k, v in batch.items()}


def forward_batch(model, batch):
    """Runs the model on a batch (labels are not passed in; they're used for the loss)."""
    return model(batch["pixel_values"], batch["input_ids"], batch["attention_mask"])


@torch.no_grad()
def evaluate(model, loader, device):
    """Runs the model over a loader and returns loss, accuracy, precision, recall, and F1 scores."""
    model.eval()                                        # turns off dropout
    preds, golds, total_loss = [], [], 0.0
    for batch in loader:
        batch = to_device(batch, device)
        logits = forward_batch(model, batch)
        # Plain (unweighted) loss, summed so we can average per tweet at the end.
        total_loss += nn.functional.cross_entropy(logits, batch["labels"], reduction="sum").item()
        preds.extend(logits.argmax(dim=-1).cpu().tolist())      # predicted class
        golds.extend(batch["labels"].cpu().tolist())            # true class
    p, r, f1, _ = precision_recall_fscore_support(golds, preds, average="weighted", zero_division=0)
    _, _, f1_macro, _ = precision_recall_fscore_support(golds, preds, average="macro", zero_division=0)
    return dict(loss=total_loss / len(golds), acc=accuracy_score(golds, preds), precision=p,
                recall=r, f1_weighted=f1, f1_macro=f1_macro)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def set_seed(seed):
    """Seeds all random number generators for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    """Defines the command-line options."""
    ap = argparse.ArgumentParser()
    # Data locations and column names
    ap.add_argument("--csv", default="data/crisismmd_clean.csv")
    ap.add_argument("--root", default="dataset/CrisisMMD_v2.0")
    ap.add_argument("--text-column", default="text_clean")
    ap.add_argument("--label-column", default="text_info")        # Task 1 label column in your CSV
    # Leave empty if the CSV is already agreed-only; set to "image_info" to filter text==image here.
    ap.add_argument("--image-label-column", default="")
    ap.add_argument("--split-column", default="split")
    # Model options
    ap.add_argument("--model", default="openai/clip-vit-base-patch32")
    ap.add_argument("--n-vision", type=int, default=0, help="keep first N vision layers (0 = all 12)")
    ap.add_argument("--n-text", type=int, default=0, help="keep first N text layers (0 = all 12)")
    ap.add_argument("--freeze-encoders", action="store_true", help="train the fusion head only")
    # Training hyperparameters
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr-backbone", type=float, default=1e-5)   # small LR for pretrained CLIP
    ap.add_argument("--lr-head", type=float, default=1e-3)       # larger LR for the new head
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-ratio", type=float, default=0.1)   # fraction of steps for LR warmup
    ap.add_argument("--class-weights", action="store_true", help="weight loss by inverse class frequency")
    ap.add_argument("--select-metric", default="f1_weighted", choices=["acc", "f1_weighted", "f1_macro"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="subsample train rows (debugging)")
    ap.add_argument("--out", default=None,
                    help="output folder (default: runs/v<vision>_t<text>_e<epochs>_s<seed>)")
    ap.add_argument("--fusion", default="concat3",
                    choices=["concat2", "concat3", "concat_diff", "concat4", "prod"],
                    help="how image/text embeddings are combined before the head")
    ap.add_argument("--hidden", type=int, default=512, help="head hidden size (0 = linear head)")
    ap.add_argument("--dropout", type=float, default=0.3)
    args = ap.parse_args()

    # An argparse default can't depend on other arguments, so build the folder name here.
    # (0 layers means "keep all", shown as "all".)
    if args.out is None:
        args.out = (f"runs/v{args.n_vision or 'all'}_t{args.n_text or 'all'}"
                    f"_e{args.epochs}_s{args.seed}")
    return args


def main():
    args = parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"device {device}")
    start_time = time.time()

    # ---- Load data and model ----
    splits = load_splits(args)
    clip, processor = load_clip(args.model, args.n_vision, args.n_text)
    n_vis = len(clip.vision_model.encoder.layers)
    n_txt = len(clip.text_model.encoder.layers)
    print(f"vision layers: {n_vis}, text layers: {n_txt}")
    model = CLIPFusionClassifier(clip, hidden=args.hidden, dropout=args.dropout,
                             fusion=args.fusion,
                             freeze_encoders=args.freeze_encoders).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    # ---- Data loaders (shuffle only the training set) ----
    workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 4))
    make_loader = lambda split, shuffle: DataLoader(
        Pairs(args.root, splits[split], args.text_column), batch_size=args.batch_size,
        shuffle=shuffle, num_workers=workers, collate_fn=partial(collate, processor=processor))
    train_loader = make_loader("train", True)
    dev_loader, test_loader = make_loader("dev", False), make_loader("test", False)

    # ---- Optimizer and learning-rate schedule ----
    groups = [{"params": model.head.parameters(), "lr": args.lr_head}]
    if not args.freeze_encoders:
        groups.append({"params": model.clip.parameters(), "lr": args.lr_backbone})
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(args.warmup_ratio * total_steps), total_steps)

    # ---- Loss function ----
    weight = None
    if args.class_weights:
        counts = np.bincount(splits["train"]["y"].values, minlength=2)
        weight = torch.tensor(counts.sum() / (2 * counts), dtype=torch.float, device=device)
        print(f"class weights {weight.tolist()}")
    loss_fn = nn.CrossEntropyLoss(weight=weight)
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ---- Training loop ----
    best_score, best_epoch, best_state, history = -1.0, 0, None, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for step, batch in enumerate(train_loader, 1):
            batch = to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                loss = loss_fn(forward_batch(model, batch), batch["labels"])
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running += loss.item()
            if step % 50 == 0:
                print(f"  epoch {epoch} step {step}/{len(train_loader)} loss {running / step:.4f}", flush=True)

        # After each epoch, score on the dev set and log it (including dev loss).
        dev = evaluate(model, dev_loader, device)
        train_loss = running / len(train_loader)
        history.append(dict(epoch=epoch, train_loss=train_loss, **{f"dev_{k}": v for k, v in dev.items()}))
        print(f"epoch {epoch}: train loss {train_loss:.4f} | dev loss {dev['loss']:.4f} | "
              f"dev acc {dev['acc']:.4f} | dev f1 {dev['f1_weighted']:.4f}")
        if dev[args.select_metric] > best_score:
            best_score, best_epoch = dev[args.select_metric], epoch
            best_state = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})
            print(f"  new best dev {args.select_metric}={best_score:.4f}")

    # ---- Final evaluation: restore the best-on-dev weights and test once ----
    model.load_state_dict(best_state)
    test = evaluate(model, test_loader, device)
    print("TEST (best dev checkpoint): " + " ".join(f"{k}={v:.4f}" for k, v in test.items()))

    # ---- Save weights and a JSON record of the run ----
    minutes = (time.time() - start_time) / 60
    torch.save(best_state, out_dir / "best.pt")
    with open(out_dir / "metrics.json", "w") as fh:
        json.dump(dict(args=vars(args), n_vision_layers=n_vis, n_text_layers=n_txt,
                       n_params=n_params, minutes=minutes, best_epoch=best_epoch,
                       best_dev=best_score, select_metric=args.select_metric,
                       history=history, test=test), fh, indent=2)
    print(f"saved to {out_dir}")

    print("\nWhat to look at:")
    print("  1. Dev F1 (higher is better) - compare runs on this, not on training loss.")
    print("  2. Dev loss vs train loss - if train keeps falling while dev loss rises, it is overfitting.")
    print("  3. Use test scores only for the configuration you pick on dev (run compare_runs.py).")


if __name__ == "__main__":
    main()