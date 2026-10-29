import argparse
import csv
import html
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
 
import numpy as np
import pandas as pd
from PIL import Image, ImageFile
 
ImageFile.LOAD_TRUNCATED_IMAGES = True
 
URL_RE = re.compile(r"https?://\S+|www\.\S+")
RT_RE = re.compile(r"\bRT @\w+:?\s*")
MENTION_RE = re.compile(r"@\w+")
SPACE_RE = re.compile(r"\s+")
 
 
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="dataset/CrisisMMD_v2.0")
    parser.add_argument("--out", default="data/crisismmd_clean.csv")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--valid-info", nargs="+", default=["informative", "not_informative"])
    return parser.parse_args()
 
 
def load_annotations(root):
    paths = sorted(
        p for p in (Path(root) / "annotations").glob("*.tsv") if not p.name.startswith(".")
    )
    frames = []
    for path in paths:
        print(f"reading {path.name}", flush=True)
        frame = pd.read_csv(
            path, sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE
        )
        frame["event"] = frame["image_path"].str.split("/").str[1]
        print(f"{path.name} {len(frame)} rows")
        frames.append(frame)
    frame = pd.concat(frames, ignore_index=True)
    print(f"loaded {len(frame)} rows")
    return frame
 
 
def drop_bad_labels(frame, valid):
    bad = ~frame["text_info"].isin(valid) | ~frame["image_info"].isin(valid)
    print("text_info values")
    print(frame["text_info"].value_counts().to_string())
    print("image_info values")
    print(frame["image_info"].value_counts().to_string())
    if bad.mean() > 0.2:
        raise SystemExit(f"{bad.mean():.2%} of rows have labels outside {valid}, check the value counts above")
    print(f"dropped {int(bad.sum())} rows with invalid informative labels")
    return frame[~bad].reset_index(drop=True)
 
 
def drop_empty_and_duplicates(frame):
    empty = frame["tweet_text"].str.strip() == ""
    print(f"dropped {int(empty.sum())} rows with empty tweet text")
    frame = frame[~empty]
    before = len(frame)
    frame = frame.drop_duplicates(subset="image_id").reset_index(drop=True)
    print(f"dropped {before - len(frame)} duplicate image ids")
    return frame
 
 
def image_ok(root, rel_path):
    try:
        with Image.open(Path(root) / rel_path) as image:
            image.convert("RGB")
        return True
    except Exception:
        return False
 
 
def drop_bad_images(frame, root):
    workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 4))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        ok = list(pool.map(lambda p: image_ok(root, p), frame["image_path"]))
    ok = pd.Series(ok, index=frame.index)
    print(f"dropped {int((~ok).sum())} rows with missing or unreadable images")
    return frame[ok].reset_index(drop=True)
 
 
def clean_text(text):
    text = html.unescape(text)
    text = URL_RE.sub(" ", text)
    text = RT_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = text.replace("#", "")
    return SPACE_RE.sub(" ", text).strip()
 
 
def add_clean_text(frame):
    frame["text_clean"] = frame["tweet_text"].map(clean_text)
    frame["text_empty"] = frame["text_clean"] == ""
    print(f"{int(frame['text_empty'].sum())} rows have empty text after cleaning")
    print(f"{int((~frame['tweet_text'].map(str.isascii)).sum())} rows contain non-ASCII characters")
    return frame
 
 
def add_flags(frame):
    frame["agreed"] = frame["text_info"] == frame["image_info"]
    frame["agreed_human"] = frame["text_human"] == frame["image_human"]
    print(f"text and image informative labels agree on {frame['agreed'].mean():.3f} of rows")
    print(f"text and image humanitarian labels agree on {frame['agreed_human'].mean():.3f} of rows")
    return frame
 
 
def add_split(frame, seed):
    key = frame["text_clean"].str.lower().where(~frame["text_empty"], "tweet:" + frame["tweet_id"])
    groups = key.unique().tolist()
    rng = np.random.default_rng(seed)
    shuffled = [groups[i] for i in rng.permutation(len(groups))]
    cut_train, cut_dev = int(0.7 * len(shuffled)), int(0.85 * len(shuffled))
    lookup = {}
    for rank, name in enumerate(shuffled):
        lookup[name] = "train" if rank < cut_train else "dev" if rank < cut_dev else "test"
    frame["split"] = key.map(lookup)
    summary = frame.groupby("split").agg(
        rows=("image_id", "size"),
        tweets=("tweet_id", "nunique"),
        informative=("image_info", lambda s: (s == "informative").mean()),
        agreed=("agreed", "mean"),
    )
    print(summary.to_string())
    return frame
 
 
def save(frame, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_path, index=False)
    print(f"saved {out_path} {len(frame)} rows")
    print(frame["event"].value_counts().to_string())
 
 
def main():
    args = parse_args()
    frame = load_annotations(args.root)
    frame = drop_bad_labels(frame, args.valid_info)
    frame = drop_empty_and_duplicates(frame)
    frame = drop_bad_images(frame, args.root)
    frame = add_clean_text(frame)
    frame = add_flags(frame)
    frame = add_split(frame, args.seed)
    save(frame, args.out)
 
 
if __name__ == "__main__":
    main()