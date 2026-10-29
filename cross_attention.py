import argparse
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/crisismmd_clean.csv")
    parser.add_argument("--emb", default="embeddings/crisismmd_text_clean.npz")
    parser.add_argument("--label-column", default="image_info")
    parser.add_argument("--all-rows", action="store_true", help="use disagreeing rows too")
    parser.add_argument("--fusions", nargs="+", default=["concat", "self", "cross"],
                        choices=["concat", "self", "cross"])
    parser.add_argument("--k", type=int, default=100, help="projection size, 100 in the paper")
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--no-class-weights", action="store_true")
    parser.add_argument("--out", default="results/cross_attention.csv")
    parser.add_argument("--preds-dir", default="results/preds")
    return parser.parse_args()


def load_data(csv_path, emb_path, all_rows):
    frame = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    data = np.load(emb_path)
    order = pd.Series(range(len(data["image_id"])), index=data["image_id"].astype(str))
    index = order.loc[frame["image_id"]].to_numpy()
    image_emb = data["image_emb"][index]
    text_emb = data["text_emb"][index]
    if not all_rows:
        keep = (frame["agreed"] == "True").to_numpy()
        frame, image_emb, text_emb = frame[keep].reset_index(drop=True), image_emb[keep], text_emb[keep]
    print(f"{len(frame)} rows, all_rows={all_rows}")
    return frame, image_emb, text_emb


class GatedFusion(nn.Module):
    """Cross-attention fusion in the style of Abavisani et al. (CVPR 2020).

    Each modality is projected to K dimensions with a linear layer and ReLU.
    A sigmoid mask of size K is computed for each modality and multiplied
    element-wise into its projected features. The two gated vectors are
    concatenated and passed to a small fully connected classifier.

    fusion "concat" uses no masks (plain concatenation of the projections).
    fusion "cross"  computes the image mask from the text and the text mask from the image.
    fusion "self"   computes each mask from its own modality.
    """

    def __init__(self, dim_image, dim_text, k, hidden, dropout, fusion):
        super().__init__()
        self.fusion = fusion
        self.proj_image = nn.Sequential(nn.Linear(dim_image, k), nn.ReLU())
        self.proj_text = nn.Sequential(nn.Linear(dim_text, k), nn.ReLU())
        if fusion == "cross":
            self.mask_image = nn.Linear(dim_text, k)
            self.mask_text = nn.Linear(dim_image, k)
        elif fusion == "self":
            self.mask_image = nn.Linear(dim_image, k)
            self.mask_text = nn.Linear(dim_text, k)
        self.head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(2 * k, hidden), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden, 2),
        )

    def forward(self, image, text):
        fv, fe = self.proj_image(image), self.proj_text(text)
        gates = None
        if self.fusion == "cross":
            gates = (torch.sigmoid(self.mask_image(text)), torch.sigmoid(self.mask_text(image)))
        elif self.fusion == "self":
            gates = (torch.sigmoid(self.mask_image(image)), torch.sigmoid(self.mask_text(text)))
        if gates is not None:
            fv, fe = fv * gates[0], fe * gates[1]
        return self.head(torch.cat([fv, fe], dim=1)), gates


def score(y_true, y_pred):
    per_class = f1_score(y_true, y_pred, labels=[0, 1], average=None, zero_division=0)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_not_informative": per_class[0],
        "f1_informative": per_class[1],
    }


def evaluate(model, data, split, loss_fn):
    model.eval()
    with torch.no_grad():
        logits, gates = model(data[split]["image"], data[split]["text"])
        loss = loss_fn(logits, data[split]["y"]).item()
        prob = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
    pred = (prob > 0.5).astype(int)
    return loss, pred, prob, gates


def train_one(fusion, seed, data, dims, args, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = GatedFusion(dims[0], dims[1], args.k, args.hidden, args.dropout, fusion).to(device)
    y_train = data["train"]["y"]
    if args.no_class_weights:
        weight = None
    else:
        counts = torch.bincount(y_train, minlength=2).float()
        weight = (len(y_train) / (2 * counts)).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=3)

    best = {"f1": -1.0, "state": None, "epoch": 0}
    n = len(y_train)
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(n, device=device)
        for start in range(0, n, args.batch_size):
            idx = order[start:start + args.batch_size]
            logits, _ = model(data["train"]["image"][idx], data["train"]["text"][idx])
            loss = loss_fn(logits, y_train[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        dev_loss, dev_pred, _, _ = evaluate(model, data, "dev", loss_fn)
        dev_f1 = score(data["dev"]["y"].cpu().numpy(), dev_pred)["macro_f1"]
        scheduler.step(dev_loss)
        if dev_f1 > best["f1"]:
            best = {"f1": dev_f1, "state": copy.deepcopy(model.state_dict()), "epoch": epoch}
        elif epoch - best["epoch"] >= args.patience:
            break
    model.load_state_dict(best["state"])
    print(f"  {fusion} seed {seed} stopped at epoch {epoch}, best epoch {best['epoch']}, dev macro F1 {best['f1']:.3f}")
    return model, loss_fn, best["epoch"]


def make_splits(frame, image_emb, text_emb, y, device):
    masks = {s: (frame["split"] == s).to_numpy() for s in ["train", "dev", "test"]}
    mean_i, std_i = image_emb[masks["train"]].mean(0), image_emb[masks["train"]].std(0) + 1e-6
    mean_t, std_t = text_emb[masks["train"]].mean(0), text_emb[masks["train"]].std(0) + 1e-6
    data = {}
    for split, m in masks.items():
        data[split] = {
            "image": torch.tensor((image_emb[m] - mean_i) / std_i, dtype=torch.float32, device=device),
            "text": torch.tensor((text_emb[m] - mean_t) / std_t, dtype=torch.float32, device=device),
            "y": torch.tensor(y[m], dtype=torch.long, device=device),
        }
    print({s: int(m.sum()) for s, m in masks.items()})
    return data, masks


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device {device}")
    frame, image_emb, text_emb = load_data(args.csv, args.emb, args.all_rows)
    y = (frame[args.label_column] == "informative").astype(int).to_numpy()
    data, masks = make_splits(frame, image_emb, text_emb, y, device)
    dims = (image_emb.shape[1], text_emb.shape[1])

    rows = []
    preds_dir = Path(args.preds_dir)
    preds_dir.mkdir(parents=True, exist_ok=True)
    for fusion in args.fusions:
        print(f"fusion {fusion}")
        for seed in range(args.seeds):
            model, loss_fn, best_epoch = train_one(fusion, seed, data, dims, args, device)
            for split in ["dev", "test"]:
                _, pred, prob, gates = evaluate(model, data, split, loss_fn)
                result = {"fusion": fusion, "seed": seed, "split": split, "best_epoch": best_epoch,
                          **score(data[split]["y"].cpu().numpy(), pred)}
                if gates is not None and split == "test":
                    result["mean_gate_image"] = gates[0].mean().item()
                    result["mean_gate_text"] = gates[1].mean().item()
                rows.append(result)
                if split == "test" and seed == 0:
                    out = frame.loc[masks["test"], ["image_id", "event"]].copy()
                    out["label"] = y[masks["test"]]
                    out["pred"] = pred
                    out["prob_informative"] = prob
                    out.to_csv(preds_dir / f"cross_attention_{fusion}_test.csv", index=False)

    table = pd.DataFrame(rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_path, index=False)
    metrics = ["accuracy", "f1_not_informative", "f1_informative", "macro_f1"]
    summary = table[table["split"] == "test"].groupby("fusion")[metrics].agg(["mean", "std"])
    print("test results, mean and std over seeds")
    print(summary.round(3).to_string())
    gate_cols = [c for c in table.columns if c.startswith("mean_gate")]
    if gate_cols:
        print("mean gate value on test")
        print(table[table["split"] == "test"].groupby("fusion")[gate_cols].mean().dropna().round(3).to_string())
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
