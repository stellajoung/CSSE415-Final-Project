import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

C_GRID = [0.01, 0.1, 1, 10, 100]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/crisismmd_clean.csv")
    parser.add_argument("--emb", default="embeddings/crisismmd_text_clean.npz")
    parser.add_argument("--label-column", default="image_info")
    parser.add_argument("--all-rows", action="store_true", help="use disagreeing rows too")
    parser.add_argument("--out", default="results/baselines.csv")
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


def score(y_true, y_pred):
    per_class = f1_score(y_true, y_pred, labels=[0, 1], average=None, zero_division=0)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_not_informative": per_class[0],
        "f1_informative": per_class[1],
    }


def fit_logreg(features, y, masks):
    best = None
    for c in C_GRID:
        model = LogisticRegression(C=c, class_weight="balanced", max_iter=2000)
        model.fit(features[masks["train"]], y[masks["train"]])
        dev_f1 = score(y[masks["dev"]], model.predict(features[masks["dev"]]))["macro_f1"]
        print(f"    C={c:<6} dev macro F1 {dev_f1:.3f}")
        if best is None or dev_f1 > best[0]:
            best = (dev_f1, c, model)
    return best[1], best[2]


def main():
    args = parse_args()
    frame, image_emb, text_emb = load_data(args.csv, args.emb, args.all_rows)
    y = (frame[args.label_column] == "informative").astype(int).to_numpy()
    masks = {s: (frame["split"] == s).to_numpy() for s in ["train", "dev", "test"]}
    print({s: int(m.sum()) for s, m in masks.items()}, f"informative share {y.mean():.3f}")

    rows = []
    majority = int(round(y[masks["train"]].mean()))
    for split in ["dev", "test"]:
        pred = np.full(masks[split].sum(), majority)
        rows.append({"rung": 0, "name": "majority", "C": "", "split": split,
                     **score(y[masks[split]], pred)})

    features = {
        "text only": (2, text_emb),
        "image only": (3, image_emb),
        "concat fusion": (4, np.hstack([image_emb, text_emb])),
    }
    preds_dir = Path(args.preds_dir)
    preds_dir.mkdir(parents=True, exist_ok=True)
    for name, (rung, x) in features.items():
        print(f"rung {rung} {name}")
        c, model = fit_logreg(x, y, masks)
        for split in ["dev", "test"]:
            pred = model.predict(x[masks[split]])
            rows.append({"rung": rung, "name": name, "C": c, "split": split,
                         **score(y[masks[split]], pred)})
        test = frame.loc[masks["test"], ["image_id", "event"]].copy()
        test["label"] = y[masks["test"]]
        test["pred"] = model.predict(x[masks["test"]])
        test["prob_informative"] = model.predict_proba(x[masks["test"]])[:, 1]
        test.to_csv(preds_dir / f"{name.replace(' ', '_')}_test.csv", index=False)

    table = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    print("test results")
    print(table[table["split"] == "test"].drop(columns="split").round(3).to_string(index=False))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
