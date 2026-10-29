import pandas as pd

d = pd.read_csv("data/crisismmd_clean.csv", dtype=str, keep_default_na=False)
for name, f in [("all cleaned", d), ("agreed only", d[d["agreed"] == "True"])]:
    print(name)
    print("  text records (unique tweets)", f.drop_duplicates("tweet_id")["text_info"].value_counts().to_dict())
    print("  image records (rows)", f["image_info"].value_counts().to_dict())
