import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import CLIPModel, CLIPProcessor

LABELS = ["not_informative", "informative"]

PROMPT_SETS = {
    "plain": [
        "a photo that is not informative",
        "a photo that is informative",
    ],
    "disaster": [
        "a photo that does not show useful information about a disaster",
        "a photo that shows useful information about a disaster",
    ],
    "damage": [
        "a photo unrelated to damage or help needed in a natural disaster",
        "a photo of damage, flooding, or people needing help in a natural disaster",
    ],
    "tweet": [
        "a tweet that is not useful for disaster response",
        "a tweet that is useful for disaster response",
    ],
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/crisismmd_clean.csv")
    parser.add_argument("--emb", default="embeddings/crisismmd_text_clean.npz")
    parser.add_argument("--model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--label-column", default="image_info")
    parser.add_argument("--all-rows", action="store_true", help="use disagreeing rows too")
    parser.add_argument("--out", default="results/zeroshot.csv")
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


def embed_prompts(model_name, prompts, device):
    model = CLIPModel.from_pretrained(model_name).to(device).eval()
    processor = CLIPProcessor.from_pretrained(model_name)
    inputs = processor(text=prompts, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        emb = model.get_text_features(**inputs)
    return torch.nn.functional.normalize(emb, dim=-1).cpu().numpy()


def score(y_true, y_pred):
    per_class = f1_score(y_true, y_pred, labels=[0, 1], average=None, zero_division=0)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_not_informative": per_class[0],
        "f1_informative": per_class[1],
    }


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    frame, image_emb, text_emb = load_data(args.csv, args.emb, args.all_rows)
    y = (frame[args.label_column] == "informative").astype(int).to_numpy()
    print(f"informative share {y.mean():.3f}")

    rows = []
    for name, prompts in PROMPT_SETS.items():
        prompt_emb = embed_prompts(args.model, prompts, device)
        sims = {"image": image_emb @ prompt_emb.T, "text": text_emb @ prompt_emb.T}
        sims["image+text"] = sims["image"] + sims["text"]
        for modality, sim in sims.items():
            pred = sim.argmax(axis=1)
            for split in ["dev", "test"]:
                mask = (frame["split"] == split).to_numpy()
                rows.append({"prompt_set": name, "modality": modality, "split": split,
                             **score(y[mask], pred[mask])})
    table = pd.DataFrame(rows)

    dev = table[table["split"] == "dev"].set_index(["modality", "prompt_set"])["macro_f1"]
    print("dev macro F1 by modality and prompt set")
    print(dev.unstack().round(3).to_string())
    best = dev.groupby(level="modality").idxmax()
    print("best prompt set per modality, chosen on dev")
    chosen = []
    for modality, (_, prompt_set) in best.items():
        row = table[(table["modality"] == modality) & (table["prompt_set"] == prompt_set)
                    & (table["split"] == "test")]
        chosen.append(row)
        print(row.round(3).to_string(index=False, header=False))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    pd.concat(chosen).to_csv(out.with_name("zeroshot_test_best.csv"), index=False)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
