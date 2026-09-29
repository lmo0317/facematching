"""
Evaluate lookalike search against pairs people actually call look-alikes (tools/lookalike_pairs.txt).

For each pair (A, B): query with one of A's photos while A is removed from the DB (A plays the user),
and record where B lands in the ranking. Reported per method: recall@3, recall@10, median rank, MRR.

Usage: python tools/eval_lookalike.py
Caches under data/: face_bank.npz (per-photo faces of every DB celebrity), aliases.json.
"""

import os
import sys
import json
import urllib.parse

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from face_utils import decode_image_bgr, detect_faces_yunet, embed_face  # noqa: E402
from populate_celebrities import CACHE_DIR, DB_PATH, http_json, clean_name  # noqa: E402

DATA_DIR = os.path.join(ROOT, "data")
BANK_PATH = os.path.join(DATA_DIR, "face_bank.npz")
ALIAS_PATH = os.path.join(DATA_DIR, "aliases.json")
PAIRS_PATH = os.path.join(ROOT, "tools", "lookalike_pairs.txt")
QUERIES_PER_PERSON = 3


def load_db():
    with open(DB_PATH, encoding="utf-8") as f:
        return json.load(f)


def build_face_bank(db):
    """Best-matching face of every cached photo of every DB celebrity (same rule as the builder)."""
    qids = np.array([e["qid"] for e in db])
    if os.path.exists(BANK_PATH):
        z = np.load(BANK_PATH, allow_pickle=True)
        if "qids" in z.files and np.array_equal(z["qids"], qids):
            return {k: z[k] for k in z.files}
        print("celebrity_db.json changed; rebuilding face bank", flush=True)
    owner, embs, paths, rows = [], [], [], []
    for n, e in enumerate(db):
        folder = os.path.join(CACHE_DIR, e["qid"])
        mean = np.array(e["embedding"], dtype=np.float32)
        with open(os.path.join(folder, "manifest.json"), encoding="utf-8") as f:
            photos = json.load(f)["photos"]
        for p in photos:
            path = os.path.join(folder, p["file"])
            if not os.path.exists(path):
                continue
            with open(path, "rb") as f:
                img = decode_image_bgr(f.read())
            if img is None:
                continue
            best = None
            for row in detect_faces_yunet(img, 0.7):
                if min(row[2], row[3]) < 60:
                    continue
                v = embed_face(img, row)
                if v is not None and v @ mean >= 0.4 and (best is None or v @ mean > best[0] @ mean):
                    best = (v, row)
            if best:
                owner.append(n)
                embs.append(best[0])
                paths.append(os.path.relpath(path, ROOT))
                rows.append(best[1][:15])
        if (n + 1) % 100 == 0:
            print(f"  face bank: {n + 1}/{len(db)} people, {len(embs)} faces", flush=True)
    bank = {"owner": np.array(owner), "emb": np.array(embs, dtype=np.float32),
            "path": np.array(paths), "row": np.array(rows, dtype=np.float32), "qids": qids}
    np.savez(BANK_PATH, **bank)
    return bank


def load_aliases(db):
    """qid -> set of names (DB name, English name, Wikidata ko/en labels and aliases)."""
    cached = {}
    if os.path.exists(ALIAS_PATH):
        with open(ALIAS_PATH, encoding="utf-8") as f:
            cached = {k: set(v) for k, v in json.load(f).items()}
        if all(e["qid"] in cached for e in db):
            return cached
    names = {e["qid"]: {e["name"], e.get("name_en", "")} for e in db}
    ids = list(names)
    for i in range(0, len(ids), 50):
        data = http_json("https://www.wikidata.org/w/api.php?" + urllib.parse.urlencode({
            "action": "wbgetentities", "ids": "|".join(ids[i:i + 50]), "props": "labels|aliases",
            "languages": "ko|en", "format": "json"}))
        for qid, ent in data["entities"].items():
            for lang in ("ko", "en"):
                if lang in ent.get("labels", {}):
                    names[qid].add(ent["labels"][lang]["value"])
                names[qid].update(a["value"] for a in ent.get("aliases", {}).get(lang, []))
    names = {q: {clean_name(n) for n in v if n} for q, v in names.items()}
    with open(ALIAS_PATH, "w", encoding="utf-8") as f:
        json.dump({k: sorted(v) for k, v in names.items()}, f, ensure_ascii=False)
    return names


def resolve_pairs(db, aliases):
    index = {}
    for i, e in enumerate(db):
        for n in aliases[e["qid"]]:
            index.setdefault(n, set()).add(i)
    pairs, missing = [], set()
    with open(PAIRS_PATH, encoding="utf-8") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            a, b = (s.strip() for s in line.split("|"))
            ia, ib = index.get(a, set()), index.get(b, set())
            if len(ia) == 1 and len(ib) == 1 and ia != ib:
                pairs.append((next(iter(ia)), next(iter(ib))))
            else:
                missing.update(n for n, s in ((a, ia), (b, ib)) if len(s) != 1)
    return pairs, missing


def evaluate(name, score_fn, db, bank, pairs):
    """score_fn(query_face_indices, exclude_person) -> similarity per DB person (higher = more alike)."""
    genders = np.array([e["gender"] for e in db])
    ranks = []
    for a, b in pairs + [(b, a) for a, b in pairs]:
        faces = np.where(bank["owner"] == a)[0][:QUERIES_PER_PERSON]
        for f in faces:
            sims = score_fn(np.array([f]), a).astype(np.float64)
            sims[a] = -np.inf
            if genders[a] == genders[b]:
                sims[genders != genders[a]] = -np.inf
            ranks.append(int((sims > sims[b]).sum()) + 1)
    r = np.array(ranks)
    print(f"{name:34s} queries={len(r):4d}  R@3={np.mean(r <= 3):5.1%}  R@10={np.mean(r <= 10):5.1%}  "
          f"median rank={np.median(r):5.0f}  MRR={np.mean(1 / r):.3f}", flush=True)
    return r


def main():
    db = load_db()
    bank = build_face_bank(db)
    aliases = load_aliases(db)
    pairs, missing = resolve_pairs(db, aliases)
    print(f"{len(pairs)} pairs with both people in the DB ({len(missing)} names unmatched/ambiguous)")
    n_same = int(np.mean([db[a]['gender'] == db[b]['gender'] for a, b in pairs]) * 100)
    print(f"same-gender pairs: {n_same}%  | random ranking would give median rank ~{len(db) // 4}")

    means = np.array([e["embedding"] for e in db], dtype=np.float32)
    E = bank["emb"]

    def arcface_mean(q, exclude):
        return means @ E[q].mean(axis=0)

    evaluate("ArcFace vs person mean (current)", arcface_mean, db, bank, pairs)
    return db, bank, pairs


if __name__ == "__main__":
    main()
