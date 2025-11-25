import json

input_path = "/kaggle/working/predictions_dev.jsonl"
output_path = "/kaggle/working/predictions_dev_fixed.jsonl"

with open(input_path, "r") as fin, open(output_path, "w") as fout:
    for line in fin:
        obj = json.loads(line)
        # extract score, rename field
        score = float(obj["pred"])
        fout.write(json.dumps({"prediction": score}) + "\n")

print("Fixed file saved to:", output_path)
