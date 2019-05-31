import json
def config(path):
    return json.load(open(path,"r",encoding="utf-8"))