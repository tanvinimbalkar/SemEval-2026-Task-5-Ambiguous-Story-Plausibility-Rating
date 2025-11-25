import json
from datasets import Dataset

LABEL_FIELD = "average"
STD_FIELD = "stdev"

TEXT_FIELDS = [
    "precontext",
    "sentence",
    "ending",
    "example_sentence",
    "judged_meaning",
    "homonym"
]

def _normalize(path):
    data = json.load(open(path, "r"))
    samples = []

    if isinstance(data, dict):
        for k, ex in data.items():
            ex = dict(ex)
            ex["id"] = str(k)
            samples.append(ex)
    else:
        for i, ex in enumerate(data):
            ex = dict(ex)
            ex.setdefault("id", str(i))
            samples.append(ex)

    return samples


def _build_text(ex):
    parts = []
    for field in TEXT_FIELDS:
        if field in ex and isinstance(ex[field], str):
            parts.append(ex[field])
    return " ".join(parts)


def load_dataset(path):
    raw = _normalize(path)
    processed = []

    for ex in raw:
        row = {
            "id": ex["id"],
            "text": _build_text(ex),
        }
        if LABEL_FIELD in ex:
            row[LABEL_FIELD] = float(ex[LABEL_FIELD])
        if STD_FIELD in ex:
            row[STD_FIELD] = float(ex[STD_FIELD])
        processed.append(row)

    return Dataset.from_list(processed)
