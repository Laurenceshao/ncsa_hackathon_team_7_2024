import json

def load_jsonl(path):
    data = []
    with open(path, 'r') as f:
        for item in list(f):
            data.append(json.loads(item))
    f.close()
    return data