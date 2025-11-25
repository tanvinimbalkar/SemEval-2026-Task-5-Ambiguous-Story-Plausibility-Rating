import json

dev = json.load(open("/kaggle/input/semeval26-task5-dataa/dev.json"))
out = open("/kaggle/working/semeval26-05-scripts/input/ref/solution.jsonl", "w")

for _id, row in dev.items():
    obj = {
        "id": str(_id),
        "mean": float(row["average"]),
        "std": float(row["stdev"])
    }
    out.write(json.dumps(obj) + "\n")

out.close()
print("Created solution.jsonl")
