"""
Measure ArcFace cosine distributions used to calibrate similarity percentages in face_utils.

- kin:      parent-child / sibling pairs from Wikidata (one main photo each)
- stranger: random unrelated pairs drawn from the same photo pool
- same:     two different photos of the same celebrity (from data/celeb_cache)
- celeb top-1: each celebrity's display photo vs the rest of celebrity_db.json

Run after populate_celebrities.py:  python tools/calibrate_scores.py [kin|celeb|all]
Photos are cached under data/kin_cache/.
"""

import os
import sys
import json
import random
import urllib.parse
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from face_utils import extract_face_embedding  # noqa: E402
from populate_celebrities import http_json, http_bytes, CACHE_DIR, CELEB_DIR, DB_PATH  # noqa: E402

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KIN_DIR = os.path.join(BASE_DIR, "data", "kin_cache")

# (relation property, relation label, country QID, row limit); kept small so each query stays under the 60s WDQS limit
KIN_QUERIES = [(prop, rel, country, limit)
               for prop, rel in (("P40", "parent-child"), ("P3373", "sibling"))
               for country, limit in (("Q884", 300), ("Q148", 200), ("Q17", 200), ("Q30", 100))]
KIN_QUERY = """
SELECT ?a ?b ?ga ?gb ?ia ?ib WHERE {{
  ?a wdt:{prop} ?b.
  ?b wdt:P27 wd:{country}; wdt:P569 ?born. FILTER(YEAR(?born) >= 1930)
  ?a wdt:P18 ?ia; wdt:P21 ?ga. ?b wdt:P18 ?ib; wdt:P21 ?gb.
}} LIMIT {limit}
"""


def cache_path(file_name: str) -> str:
    return os.path.join(KIN_DIR, urllib.parse.quote(file_name, safe="")[:150])


def resolve_thumb_urls(file_names: list) -> dict:
    """Commons file name -> 640px thumbnail URL, 50 titles per API call."""
    urls = {}
    for i in range(0, len(file_names), 50):
        chunk = file_names[i:i + 50]
        url = "https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode({
            "action": "query", "titles": "|".join("File:" + n for n in chunk), "prop": "imageinfo",
            "iiprop": "url", "iiurlwidth": 640, "format": "json"})
        try:
            data = http_json(url)
        except Exception as e:
            print(f"[WARN] imageinfo batch failed: {e}", flush=True)
            continue
        normalized = {n["to"]: n["from"] for n in data.get("query", {}).get("normalized", [])}
        for page in data.get("query", {}).get("pages", {}).values():
            ii = (page.get("imageinfo") or [{}])[0]
            title = normalized.get(page["title"], page["title"])
            if ii.get("thumburl") or ii.get("url"):
                urls[title[5:]] = ii.get("thumburl") or ii.get("url")
    return urls


def embed_people(files: dict) -> dict:
    """person key -> ArcFace embedding of their main photo (None if no face)."""
    os.makedirs(KIN_DIR, exist_ok=True)
    missing = sorted({n for n in files.values() if not os.path.exists(cache_path(n))})
    urls = resolve_thumb_urls(missing)

    def download(name):
        if name in urls:
            try:
                data = http_bytes(urls[name])
                with open(cache_path(name), "wb") as f:
                    f.write(data)
            except Exception:
                pass

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(download, missing))

    emb = {}
    for key, name in files.items():
        path = cache_path(name)
        if os.path.exists(path):
            with open(path, "rb") as f:
                emb[key] = extract_face_embedding(f.read())
    return emb


def q(values, label):
    v = np.array(values)
    ps = [5, 10, 25, 50, 75, 90, 95]
    print(f"{label:28s} n={len(v):4d} " + " ".join(f"p{p}={np.percentile(v, p):.3f}" for p in ps), flush=True)


def kin_stats():
    os.makedirs(KIN_DIR, exist_ok=True)
    rows = []
    for prop, rel, country, limit in KIN_QUERIES:
        query = KIN_QUERY.format(prop=prop, country=country, limit=limit)
        url = "https://query.wikidata.org/sparql?" + urllib.parse.urlencode({"query": query, "format": "json"})
        try:
            got = http_json(url, timeout=90)["results"]["bindings"]
        except Exception as e:
            print(f"[WARN] kin query {rel}/{country} failed: {e}", flush=True)
            continue
        for r in got:
            r["rel"] = {"value": rel}
        rows += got
    print(f"{len(rows)} kin pairs from Wikidata", flush=True)

    pairs, gender, seen_pairs, files = [], {}, set(), {}
    for r in rows:
        a, b = r["a"]["value"], r["b"]["value"]
        if a == b or frozenset((a, b)) in seen_pairs:
            continue
        seen_pairs.add(frozenset((a, b)))
        pairs.append((a, b, r["rel"]["value"]))
        for key, img, g in ((a, r["ia"]["value"], r["ga"]["value"]), (b, r["ib"]["value"], r["gb"]["value"])):
            files[key] = urllib.parse.unquote(img.rsplit("/", 1)[-1])
            gender[key] = g.rsplit("/", 1)[-1]
    print(f"{len(pairs)} unique kin pairs, {len(files)} people", flush=True)

    emb = embed_people(files)
    kin = [(a, b, rel) for a, b, rel in pairs if emb.get(a) is not None and emb.get(b) is not None]
    print(f"{len(kin)} pairs with a detectable face on both sides", flush=True)

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



def celeb_stats():
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
    part = sys.argv[1] if len(sys.argv) > 1 else "all"
    if part in ("kin", "all"):
        kin_stats()
    if part in ("celeb", "all"):
        celeb_stats()
