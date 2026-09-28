"""
Measure ArcFace cosine distributions used to calibrate similarity percentages in face_utils.

- kin:      parent-child / sibling pairs from Wikidata (one main photo each)
- stranger: random unrelated pairs drawn from the same photo pool
- same:     two different photos of the same celebrity (from data/celeb_cache)
- celeb top-1: each celebrity's display photo vs the rest of celebrity_db.json

Run after populate_celebrities.py:  python tools/calibrate_scores.py
Photos are cached under data/kin_cache/.
"""

import os
import sys
import json
import random
import urllib.parse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from face_utils import extract_face_embedding  # noqa: E402
from populate_celebrities import http_json, http_bytes, CACHE_DIR, CELEB_DIR, DB_PATH  # noqa: E402

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KIN_DIR = os.path.join(BASE_DIR, "data", "kin_cache")

KIN_QUERY = """
SELECT DISTINCT ?a ?b ?rel ?ga ?gb ?ia ?ib WHERE {
  { ?a wdt:P40 ?b. BIND("parent-child" AS ?rel) } UNION { ?a wdt:P3373 ?b. BIND("sibling" AS ?rel) FILTER(STR(?a) < STR(?b)) }
  ?a wdt:P27 ?c. VALUES ?c { wd:Q884 wd:Q17 wd:Q148 wd:Q865 wd:Q30 wd:Q145 }
  ?a wdt:P18 ?ia; wdt:P21 ?ga. ?b wdt:P18 ?ib; wdt:P21 ?gb.
  ?a wdt:P569 ?da. FILTER(YEAR(?da) >= 1930)
  ?a wdt:P106 ?occ. VALUES ?occ { wd:Q33999 wd:Q177220 wd:Q947873 wd:Q10800557 wd:Q82955 wd:Q2066131 }
} LIMIT 1500
"""


def fetch_photo(file_url: str):
    name = urllib.parse.unquote(file_url.rsplit("/", 1)[-1])
    path = os.path.join(KIN_DIR, urllib.parse.quote(name, safe="") [:150])
    if not os.path.exists(path):
        try:
            data = http_bytes(f"https://commons.wikimedia.org/wiki/Special:FilePath/{urllib.parse.quote(name)}?width=640")
        except Exception:
            return None
        with open(path, "wb") as f:
            f.write(data)
    with open(path, "rb") as f:
        return extract_face_embedding(f.read())


def q(values, label):
    v = np.array(values)
    ps = [5, 10, 25, 50, 75, 90, 95]
    print(f"{label:28s} n={len(v):4d} " + " ".join(f"p{p}={np.percentile(v, p):.3f}" for p in ps), flush=True)


def main():
    os.makedirs(KIN_DIR, exist_ok=True)
    url = "https://query.wikidata.org/sparql?" + urllib.parse.urlencode({"query": KIN_QUERY, "format": "json"})
    rows = http_json(url, timeout=120)["results"]["bindings"]
    print(f"{len(rows)} kin pairs from Wikidata", flush=True)

    emb, gender, kin = {}, {}, []
    for r in rows:
        a, b = r["a"]["value"], r["b"]["value"]
        for key, img, g in ((a, r["ia"]["value"], r["ga"]["value"]), (b, r["ib"]["value"], r["gb"]["value"])):
            if key not in emb:
                emb[key] = fetch_photo(img)
                gender[key] = g.rsplit("/", 1)[-1]
        if emb[a] is not None and emb[b] is not None and a != b:
            kin.append((a, b, r["rel"]["value"]))

    related = {frozenset((a, b)) for a, b, _ in kin}
    people = [k for k, v in emb.items() if v is not None]
    rng = random.Random(0)
    strangers = []
    while len(strangers) < 3000:
        a, b = rng.sample(people, 2)
        if frozenset((a, b)) not in related:
            strangers.append((a, b))

    def cos(a, b):
        return float(emb[a] @ emb[b])

    print("\n== ArcFace cosine, single photo vs single photo ==")
    q([cos(a, b) for a, b in strangers], "stranger (all)")
    q([cos(a, b) for a, b in strangers if gender[a] == gender[b]], "stranger (same gender)")
    q([cos(a, b) for a, b in strangers if gender[a] != gender[b]], "stranger (cross gender)")
    q([cos(a, b) for a, b, _ in kin], "kin (all)")
    q([cos(a, b) for a, b, r in kin if r == "parent-child"], "parent-child")
    q([cos(a, b) for a, b, r in kin if r == "sibling"], "sibling")
    q([cos(a, b) for a, b, _ in kin if gender[a] == gender[b]], "kin (same gender)")
    q([cos(a, b) for a, b, _ in kin if gender[a] != gender[b]], "kin (cross gender)")

    # Same person: two cached photos of one celebrity (faces that match the DB mean)
    with open(DB_PATH, encoding="utf-8") as f:
        db = json.load(f)
    same = []
    for entry in db[:200]:
        folder = os.path.join(CACHE_DIR, entry["qid"])
        mean = np.array(entry["embedding"])
        vecs = []
        for fname in sorted(os.listdir(folder)) if os.path.isdir(folder) else []:
            if fname.endswith(".img"):
                with open(os.path.join(folder, fname), "rb") as f:
                    v = extract_face_embedding(f.read())
                if v is not None and v @ mean >= 0.4:
                    vecs.append(v)
            if len(vecs) >= 3:
                break
        same += [float(vecs[i] @ vecs[j]) for i in range(len(vecs)) for j in range(i + 1, len(vecs))]
    q(same, "same person")

    # Celebrity search: one photo vs other celebrities' mean embeddings
    E = np.array([e["embedding"] for e in db], dtype=np.float32)
    top1, top1_same_gender = [], []
    for i, entry in enumerate(db):
        with open(os.path.join(CELEB_DIR, entry["filename"]), "rb") as f:
            v = extract_face_embedding(f.read())
        if v is None:
            continue
        sims = E @ v
        sims[i] = -1
        top1.append(float(sims.max()))
        same_gender = [s for j, s in enumerate(sims) if db[j]["gender"] == entry["gender"]]
        top1_same_gender.append(max(same_gender))
    print("\n== Celebrity search (photo vs DB means, self excluded) ==")
    q(top1, "top-1 any gender")
    q(top1_same_gender, "top-1 same gender")


if __name__ == "__main__":
    main()
