"""Build the BookForYou dataset from the UCSD Goodreads genre files.

    python prepare_data.py [--per-genre 11500] [--vocab-size 150]

Reads data/raw/goodreads_books_<genre>.json.gz, curates shelves into trope tags (tags.py),
keeps the top-150 tags as the vocabulary, and writes:
    data/books.jsonl   one book per line: id, title, description, tags (vocab only, by shelf count), n_ratings, in_catalog
    data/vocab.json    the vocabulary with book counts
    data/split.json    BART train/eval split (80/20, seed 42) over books that have >= 1 vocab tag
"""
import argparse
import collections
import glob
import gzip
import json
import os
import random

from sklearn.model_selection import train_test_split

import tags as T

ap = argparse.ArgumentParser()
ap.add_argument("--raw", default="data/raw")
ap.add_argument("--per-genre", type=int, default=11500, help="max books kept per genre file")
ap.add_argument("--vocab-size", type=int, default=150)
ap.add_argument("--min-shelf", type=int, default=3, help="ignore shelves used by fewer readers than this")
ap.add_argument("--min-desc", type=int, default=200, help="minimum description length in characters")
ap.add_argument("--min-tags", type=int, default=5, help="catalog needs at least this many vocab tags")
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()

random.seed(args.seed)
seen, books = set(), []
for path in sorted(glob.glob(os.path.join(args.raw, "goodreads_books_*.json.gz"))):
    kept = 0
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if kept >= args.per_genre:
                break
            if random.random() > 0.3:  # cheap random subsample before any parsing work
                continue
            b = json.loads(line)
            desc = T.clean_text(b.get("description"))
            wid = b.get("work_id") or b["book_id"]
            if wid in seen or len(desc) < args.min_desc or b.get("language_code", "") not in ("", "eng", "en-US", "en-GB", "en-CA"):
                continue
            shelves = [(s["name"], int(s["count"])) for s in b.get("popular_shelves", []) if int(s["count"]) >= args.min_shelf]
            if not shelves:
                continue
            seen.add(wid)
            kept += 1
            books.append(dict(id=b["book_id"], title=b.get("title", ""), description=desc,
                              shelves=shelves, n_ratings=int(b.get("ratings_count") or 0)))
    print(f"{os.path.basename(path)}: kept {kept}")
print("books before curation:", len(books))


def curated_counts(shelves):
    """Run the 4 curation stages per shelf, summing counts of shelves that map to the same tag."""
    c = collections.Counter()
    for name, n in shelves:
        for t in T.curate([name])[3]:
            c[t] += n
    return c


per_book = [curated_counts(b["shelves"]) for b in books]
doc_freq = collections.Counter(t for c in per_book for t in c)
vocab = [t for t, _ in doc_freq.most_common(args.vocab_size)]
vset = set(vocab)
inst_total = sum(doc_freq.values())
inst_cov = sum(doc_freq[t] for t in vocab) / inst_total
print(f"unique tags after curation: {len(doc_freq)}; top-{args.vocab_size} cover {100 * inst_cov:.1f}% of tag instances")

out, n_cat = [], 0
for b, c in zip(books, per_book):
    tags = [t for t, _ in c.most_common() if t in vset]
    if not tags:
        continue
    in_cat = len(tags) >= args.min_tags
    n_cat += in_cat
    out.append(dict(id=b["id"], title=b["title"], description=b["description"], tags=tags,
                    n_ratings=b["n_ratings"], in_catalog=in_cat))
print(f"books with >=1 vocab tag: {len(out)}; catalog (>= {args.min_tags} tags): {n_cat}")

with open("data/books.jsonl", "w", encoding="utf-8") as f:
    for r in out:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
json.dump({t: doc_freq[t] for t in vocab}, open("data/vocab.json", "w", encoding="utf-8"), indent=1)
ids = [r["id"] for r in out]
train, ev = train_test_split(ids, test_size=0.2, random_state=args.seed)
json.dump({"train": train, "eval": ev}, open("data/split.json", "w"))
print(f"BART split: {len(train)} train / {len(ev)} eval")
print("top 40 tags:", vocab[:40])
