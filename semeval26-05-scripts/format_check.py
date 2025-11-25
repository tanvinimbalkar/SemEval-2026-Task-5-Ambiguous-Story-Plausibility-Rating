import json, sys

path = sys.argv[1]

with open(path) as f:
    for i, line in enumerate(f):
        try:
            obj = json.loads(line)
        except:
            print("Line", i, "is not valid JSON")
            exit(1)

        if "id" not in obj:
            print("Missing 'id' at line", i)
            exit(1)
        if "prediction" not in obj:
            print("Missing 'prediction' at line", i)
            exit(1)

print("Format OK")
