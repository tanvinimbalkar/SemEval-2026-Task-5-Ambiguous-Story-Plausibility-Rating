import json
import math
from scipy.stats import spearmanr

def load_jsonl(path):
    data = {}
    with open(path, "r") as f:
        for line in f:
            obj = json.loads(line)
            data[obj["id"]] = obj
    return data

def main(ref_path, pred_path, out_path):
    ref = load_jsonl(ref_path)
    pred = load_jsonl(pred_path)

    gold_scores = []
    pred_scores = []
    within_std = []

    for _id, gold in ref.items():
        if _id not in pred:
            continue

        g = gold["mean"]
        sd = gold.get("std", 1)
        p = pred[_id]["prediction"]

        gold_scores.append(g)
        pred_scores.append(p)

        if abs(p - g) <= sd:
            within_std.append(1)
        else:
            within_std.append(0)

    # spearman
    spearman = spearmanr(gold_scores, pred_scores).correlation
    if math.isnan(spearman):
        spearman = 0

    acc = sum(within_std) / len(within_std)

    results = {
        "spearman": float(spearman),
        "within_std_accuracy": float(acc)
    }

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(results)

if __name__ == "__main__":
    import sys
    main(sys.argv[1], sys.argv[2], sys.argv[3])
