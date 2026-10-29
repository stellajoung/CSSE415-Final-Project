"""Fine-tune a CLIP classifier on CrisisMMD Task 1 (informative vs not_informative).
 
Bottom ~80% of each CLIP encoder used stays frozen; the top ~20% of transformer
layers get LoRA adapters. A small MLP head is trained from scratch on the
projected embeddings.
 
One invocation trains three separate models, one after another:
  image       image tower only, head on img
  text        text tower only, head on txt
  image+text  both towers, fusion head on [img, txt, img*txt]
 
Each gets its own LoRA adapters and head, chosen on dev macro F1, saved under
--out-dir/<modality>/. Test metrics for all three are combined into
--out-dir/LoRA_test_best.csv in the same format as zeroshot_test_best.csv.
"""
import argparse
import json
import math
import random
from pathlib import Path
 
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from peft import LoraConfig, get_peft_model, load_peft_weights, set_peft_model_state_dict
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPModel, CLIPProcessor, get_cosine_schedule_with_warmup
 
LABELS = ["not_informative", "informative"]
MODALITIES = ["image", "text", "image+text"]
 
 
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="data/crisismmd_clean.csv")
    p.add_argument("--image-root", default=".", help="prefix joined to image_path")
    p.add_argument("--image-column", default="image_path")
    p.add_argument("--text-column", default="text_clean")
    p.add_argument("--label-column", default="image_info")
    p.add_argument("--all-rows", action="store_true", help="use disagreeing rows too")
    p.add_argument("--model", default="openai/clip-vit-base-patch32")
    p.add_argument("--modalities", nargs="+", choices=MODALITIES, default=MODALITIES,
                   help="which runs to do; all three by default")
    # LoRA placement / shape
    p.add_argument("--lora-frac", type=float, default=0.2, help="fraction of top layers with LoRA")
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lora-dropout", type=float, default=0.1)
    p.add_argument("--lora-mlp", action="store_true", help="also adapt fc1/fc2, not just attention")
    # training
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr-lora", type=float, default=1e-4)
    p.add_argument("--lr-head", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup", type=float, default=0.1, help="warmup fraction of total steps")
    p.add_argument("--class-weights", action="store_true", help="inverse-frequency CE weights")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="results/lora")
    return p.parse_args()
 
 
def uses_image(modality):
    return modality in ("image", "image+text")
 
 
def uses_text(modality):
    return modality in ("text", "image+text")
 
 
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
 
 
# ---------------------------------------------------------------- data
 
def load_frame(args):
    frame = pd.read_csv(args.csv, dtype=str, keep_default_na=False)
    if not args.all_rows:
        frame = frame[frame["agreed"] == "True"].reset_index(drop=True)
    frame["y"] = (frame[args.label_column] == "informative").astype(int)
    print(f"{len(frame)} rows, all_rows={args.all_rows}, informative share {frame['y'].mean():.3f}")
    print(frame.groupby("split")["y"].agg(["count", "mean"]).round(3).to_string())
    return frame
 
 
class CrisisDataset(Dataset):
    def __init__(self, frame, args, modality):
        self.load_image = uses_image(modality)
        self.paths = [str(Path(args.image_root) / p) for p in frame[args.image_column]]
        self.texts = frame[args.text_column].tolist()
        self.labels = frame["y"].tolist()
 
    def __len__(self):
        return len(self.labels)
 
    def __getitem__(self, i):
        # text-only runs never touch the image files
        image = Image.open(self.paths[i]).convert("RGB") if self.load_image else None
        return image, self.texts[i], self.labels[i]
 
 
def make_collate(processor, modality):
    def collate(batch):
        images, texts, labels = zip(*batch)
        enc = {}
        if uses_image(modality):
            enc.update(processor(images=list(images), return_tensors="pt"))
        if uses_text(modality):
            enc.update(processor(text=list(texts), return_tensors="pt",
                                 padding=True, truncation=True, max_length=77))
        enc["labels"] = torch.tensor(labels, dtype=torch.long)
        return enc
    return collate
 
 
# ---------------------------------------------------------------- model
 
def lora_target_regex(clip, frac, include_mlp, modality):
    """Regex matching attention (and optionally MLP) linears in the top `frac` of
    the layers of each tower this modality uses. Towers are handled separately
    since they can have different depths (e.g. ViT-L/14: 24 vision, 12 text)."""
    towers = []
    if uses_image(modality):
        towers.append(("vision_model", clip.config.vision_config))
    if uses_text(modality):
        towers.append(("text_model", clip.config.text_config))
    modules = "q_proj|k_proj|v_proj|out_proj" + ("|fc1|fc2" if include_mlp else "")
    parts = []
    for tower, cfg in towers:
        n = cfg.num_hidden_layers
        k = max(1, round(n * frac))
        top = list(range(n - k, n))
        print(f"{tower}: {n} layers, frozen 0-{n - k - 1}, LoRA on {top}")
        idx = "|".join(map(str, top))
        parts.append(rf"{tower}\.encoder\.layers\.({idx})\.(self_attn|mlp)\.({modules})")
    return r".*(" + "|".join(parts) + r")"
 
 
class CLIPClassifier(nn.Module):
    def __init__(self, clip, modality, hidden=512, dropout=0.2, n_classes=2):
        super().__init__()
        self.clip = clip
        self.modality = modality
        d = clip.config.projection_dim
        in_dim = 3 * d if modality == "image+text" else d
        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )
 
    def forward(self, batch):
        if uses_image(self.modality):
            img = F.normalize(self.clip.get_image_features(pixel_values=batch["pixel_values"]), dim=-1)
        if uses_text(self.modality):
            txt = F.normalize(self.clip.get_text_features(input_ids=batch["input_ids"],
                                                          attention_mask=batch["attention_mask"]), dim=-1)
        if self.modality == "image":
            features = img
        elif self.modality == "text":
            features = txt
        else:
            features = torch.cat([img, txt, img * txt], dim=-1)
        return self.head(features)
 
 
def build_model(args, modality, device):
    # fresh pretrained CLIP for every modality, so runs don't share any weights
    clip = CLIPModel.from_pretrained(args.model)
    config = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules=lora_target_regex(clip, args.lora_frac, args.lora_mlp, modality),
        bias="none",
    )
    # get_peft_model freezes every base weight; only LoRA A/B matrices train.
    clip = get_peft_model(clip, config)
    clip.print_trainable_parameters()
    return CLIPClassifier(clip, modality).to(device)
 
 
# ---------------------------------------------------------------- train / eval
 
def score(y_true, y_pred):
    per_class = f1_score(y_true, y_pred, labels=[0, 1], average=None, zero_division=0)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_not_informative": per_class[0],
        "f1_informative": per_class[1],
    }
 
 
def to_device(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}
 
 
@torch.no_grad()
def evaluate(model, loader, device, amp):
    model.eval()
    preds, ys = [], []
    for batch in loader:
        batch = to_device(batch, device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=amp):
            logits = model(batch)
        preds.append(logits.argmax(-1).cpu())
        ys.append(batch["labels"].cpu())
    return score(torch.cat(ys).numpy(), torch.cat(preds).numpy())
 
 
def run_modality(args, modality, frame, processor, device, amp):
    """Train one model for one modality and return its dev/test metrics."""
    print(f"\n{'=' * 20} modality: {modality} {'=' * 20}")
    seed_everything(args.seed)  # same seed for every modality, so runs are comparable
    collate = make_collate(processor, modality)
    loaders = {}
    for split in ["train", "dev", "test"]:
        part = frame[frame["split"] == split].reset_index(drop=True)
        loaders[split] = DataLoader(CrisisDataset(part, args, modality), batch_size=args.batch_size,
                                    shuffle=split == "train", num_workers=args.workers,
                                    collate_fn=collate, pin_memory=device == "cuda")
 
    model = build_model(args, modality, device)
    lora_params = [p for p in model.clip.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([
        {"params": lora_params, "lr": args.lr_lora},
        {"params": model.head.parameters(), "lr": args.lr_head},
    ], weight_decay=args.weight_decay)
    total = args.epochs * len(loaders["train"])
    scheduler = get_cosine_schedule_with_warmup(optimizer, math.ceil(args.warmup * total), total)
 
    weight = None
    if args.class_weights:
        counts = np.bincount(frame.loc[frame["split"] == "train", "y"], minlength=2)
        weight = torch.tensor(counts.sum() / (2 * counts), dtype=torch.float, device=device)
        print(f"class weights {weight.tolist()}")
 
    out_dir = Path(args.out_dir) / modality.replace("+", "_")
    out_dir.mkdir(parents=True, exist_ok=True)
    best_f1, history = -1.0, []
 
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for step, batch in enumerate(loaders["train"], 1):
            batch = to_device(batch, device)
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=amp):
                logits = model(batch)
            loss = F.cross_entropy(logits.float(), batch["labels"], weight=weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lora_params + list(model.head.parameters()), 1.0)
            optimizer.step()
            scheduler.step()
            running += loss.item()
            if step % 50 == 0:
                print(f"[{modality}] epoch {epoch} step {step}/{len(loaders['train'])} "
                      f"loss {running / step:.4f}")
 
        dev = evaluate(model, loaders["dev"], device, amp)
        history.append({"epoch": epoch, "train_loss": running / len(loaders["train"]),
                        **{f"dev_{k}": v for k, v in dev.items()}})
        print(f"[{modality}] epoch {epoch} dev " + " ".join(f"{k}={v:.3f}" for k, v in dev.items()))
        if dev["macro_f1"] > best_f1:
            best_f1 = dev["macro_f1"]
            model.clip.save_pretrained(out_dir / "lora_adapter")
            torch.save(model.head.state_dict(), out_dir / "head.pt")
            print(f"  new best dev macro F1 {best_f1:.3f}, saved")
 
    # reload best checkpoint and score dev + test
    set_peft_model_state_dict(model.clip, load_peft_weights(str(out_dir / "lora_adapter"), device=device))
    model.head.load_state_dict(torch.load(out_dir / "head.pt", map_location=device))
    rows = []
    for split in ["dev", "test"]:
        metrics = evaluate(model, loaders[split], device, amp)
        rows.append({"prompt_set": "lora_finetune", "modality": modality, "split": split, **metrics})
    result = pd.DataFrame(rows)
    print(f"[{modality}] best checkpoint")
    print(result.round(3).to_string(index=False))
 
    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
    result.to_csv(out_dir / "LoRA_test_best.csv", index=False)
 
    # free GPU memory before the next modality loads a fresh CLIP
    del model, optimizer, scheduler
    if device == "cuda":
        torch.cuda.empty_cache()
    return result
 
 
def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = device == "cuda" and torch.cuda.is_bf16_supported()
    print(f"modalities {args.modalities}")
 
    frame = load_frame(args)
    processor = CLIPProcessor.from_pretrained(args.model)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "args.json").write_text(json.dumps(vars(args), indent=2))
 
    results = []
    for modality in args.modalities:
        results.append(run_modality(args, modality, frame, processor, device, amp))
        # write after every modality so a crash later doesn't lose finished runs
        combined = pd.concat(results, ignore_index=True)
        combined.to_csv(out_root / "LoRA_last_4_layers_all_tests.csv", index=False)
        combined[combined["split"] == "test"].to_csv(out_root / "LoRA_last_4_layers_test_best.csv", index=False)
 
    print("\nsummary (test)")
    print(combined[combined["split"] == "test"].round(3).to_string(index=False))
    print(f"saved {out_root}")
 
 
if __name__ == "__main__":
    main()