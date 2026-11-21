"""Full evaluation of BookForYou against the baselines from the paper (leave-one-out).

    python evaluate.py [--bart models/bart-tropes] [--queries 1000]

Steps: embed the catalog with all-MiniLM-L6-v2 -> sample queries from BART's held-out split ->
BART generates T_q -> SBERT top-100 pool -> re-rank with each system -> metrics@10 on the test half.
lambda (and mu for SBERT+BM25) are chosen on the validation half, as in the paper.
Writes results/eval.md and results/eval.json.
"""
import argparse
import json
import os
import random

import numpy as np
import torch
from scipy.stats import wilcoxon
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from transformers import AutoModel, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--bart", default="models/bart-tropes")
ap.add_argument("--queries", type=int, default=1000)
ap.add_argument("--k", type=int, default=10)
ap.add_argument("--pool", type=int, default=100)
ap.add_argument("--thr", type=int, default=5, help="relevant if >= thr shared tags")
ap.add_argument("--no-cross", action="store_true", help="skip the cross-encoder baseline")
args = ap.parse_args()
K, POOL, THR = args.k, args.pool, args.thr
dev = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs("results", exist_ok=True)
os.makedirs("data/cache", exist_ok=True)

# ---------------------------------------------------------------- data
books = [json.loads(l) for l in open("data/books.jsonl", encoding="utf-8")]
vocab = list(json.load(open("data/vocab.json")))
split = json.load(open("data/split.json"))
cat = [b for b in books if b["in_catalog"]]
N = len(cat)
pos = {b["id"]: i for i, b in enumerate(cat)}
tag_ix = {t: i for i, t in enumerate(vocab)}
M = np.zeros((N, len(vocab)), dtype=np.float32)
for i, b in enumerate(cat):
    for t in b["tags"]:
        M[i, tag_ix[t]] = 1
tagsets = [set(b["tags"]) for b in cat]
pop = np.zeros(N, dtype=bool)
pop[np.argsort([-b["n_ratings"] for b in cat])[:1000]] = True
print(f"catalog: {N} books, {len(vocab)} tags")

# ---------------------------------------------------------------- SBERT embeddings
SB = "sentence-transformers/all-MiniLM-L6-v2"
cache = "data/cache/emb.npy"
if os.path.exists(cache) and np.load(cache).shape[0] == N:
    E = np.load(cache)
else:
    stok, smod = AutoTokenizer.from_pretrained(SB), AutoModel.from_pretrained(SB).to(dev).eval()
    out = []
    for i in range(0, N, 128):
        enc = stok([b["description"] for b in cat[i:i + 128]], truncation=True, max_length=256, padding=True, return_tensors="pt").to(dev)
        with torch.no_grad():
            h = smod(**enc).last_hidden_state
        m = enc["attention_mask"].unsqueeze(-1).float()
        out.append(torch.nn.functional.normalize((h * m).sum(1) / m.sum(1), dim=1).cpu().numpy())
        if (i // 128) % 50 == 0:
            print("embedding", i, "/", N, flush=True)
    E = np.concatenate(out)
    np.save(cache, E)
    del smod

# ---------------------------------------------------------------- queries (BART held-out split only)
random.seed(42)
held = [pos[i] for i in split["eval"] if i in pos]
print(f"held-out queries available in catalog: {len(held)}")
cand = random.sample(held, min(args.queries, len(held)))
SH = {}
keep = []
for q in cand:
    sh = (M @ M[q]).astype(np.int16)
    sh[q] = -1  # leave-one-out: the query is never its own match
    if (sh >= THR).any():
        SH[q] = sh
        keep.append(q)
print(f"queries with at least one relevant book: {len(keep)} / {len(cand)}")
random.shuffle(keep)
half = len(keep) // 2
val, test = keep[:half], keep[half:]
print(f"validation {len(val)} / test {len(test)}")

# ---------------------------------------------------------------- Stage 1a: BART generates T_q
qcache = "data/cache/query_tags.json"
qtags = json.load(open(qcache)) if os.path.exists(qcache) else {}
todo = [q for q in keep if str(q) not in qtags]
if todo:
    btok, bmod = AutoTokenizer.from_pretrained(args.bart), AutoModelForSeq2SeqLM.from_pretrained(args.bart).to(dev).eval()
    for i in range(0, len(todo), 32):
        chunk = todo[i:i + 32]
        enc = btok([cat[q]["description"] for q in chunk], truncation=True, max_length=256, padding=True, return_tensors="pt").to(dev)
        with torch.no_grad(), torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
            gen = bmod.generate(**enc, num_beams=4, max_new_tokens=64)
        for q, s in zip(chunk, btok.batch_decode(gen, skip_special_tokens=True)):
            tags = []
            for t in s.split(","):
                t = t.strip().lower()
                if t and t not in tags:
                    tags.append(t)
            qtags[str(q)] = tags
        print("BART", i + len(chunk), "/", len(todo), flush=True)
    json.dump(qtags, open(qcache, "w"))
    del bmod
Tq = {q: set(qtags[str(q)]) for q in keep}
all_gen = [t for q in keep for t in qtags[str(q)]]
vs = set(vocab)
print(f"avg tags/query {len(all_gen) / len(keep):.2f}, hallucination rate {100 * sum(t not in vs for t in all_gen) / len(all_gen):.2f}%")

# ---------------------------------------------------------------- lexical baselines over the full catalog
descs = [b["description"] for b in cat]
tfv = TfidfVectorizer(stop_words="english", sublinear_tf=True, max_features=100000)
X = tfv.fit_transform(descs)
cv = CountVectorizer(stop_words="english", max_features=100000)
C = cv.fit_transform(descs).tocsr().astype(np.float32)
dl = np.asarray(C.sum(1)).ravel()
dfreq = np.asarray((C > 0).sum(0)).ravel()
idf = np.log(1 + (N - dfreq + 0.5) / (dfreq + 0.5))
K1, B = 1.5, 0.75
W = C.copy()
rows = np.repeat(np.arange(N), np.diff(W.indptr))
tf = W.data
W.data = (idf[W.indices] * tf * (K1 + 1) / (tf + K1 * (1 - B + B * dl[rows] / dl.mean()))).astype(np.float32)


def bm25_scores(q):
    qv = (C[q] > 0).astype(np.float32)
    return (W @ qv.T).toarray().ravel()


# ---------------------------------------------------------------- cross-encoder (optional)
ce = None
if not args.no_cross:
    CE = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ctok, ce = AutoTokenizer.from_pretrained(CE), AutoModelForSequenceClassification.from_pretrained(CE).to(dev).eval()


def ce_scores(q, pool):
    out = []
    for i in range(0, len(pool), 50):
        ch = pool[i:i + 50]
        enc = ctok([cat[q]["description"]] * len(ch), [cat[j]["description"] for j in ch], truncation=True,
                   max_length=256, padding=True, return_tensors="pt").to(dev)
        with torch.no_grad():
            out.append(ce(**enc).logits[:, 0].float().cpu().numpy())
    return np.concatenate(out)


# ---------------------------------------------------------------- per-query candidate data
def top_idx(scores, q, n):
    scores = scores.copy()
    scores[q] = -np.inf
    p = np.argpartition(-scores, n)[:n]
    return p[np.argsort(-scores[p], kind="stable")]


D = {}
for q in keep:
    s_all = E @ E[q]
    pool = top_idx(s_all, q, POOL)                       # SBERT order
    s = s_all[pool]
    ato = np.array([len(Tq[q] & tagsets[j]) / len(Tq[q]) if Tq[q] else 0.0 for j in pool])
    d = dict(pool=pool, s=s, ato=ato)
    d["bm"] = bm25_scores(q)
    D[q] = d
print("candidate pools ready")


def order_by(score, pool):
    return pool[np.argsort(-score, kind="stable")]


def minmax(x):
    r = x.max() - x.min()
    return (x - x.min()) / r if r else np.zeros_like(x)


def sys_lists(q, lam=0.5, mu=0.8, which=None):
    d, pool = D[q], D[q]["pool"]
    sel = {}
    sel["Popularity"] = order_by(np.array([cat[j]["n_ratings"] for j in pool], dtype=float), pool)
    sel["TF-IDF"] = top_idx((X @ X[q].T).toarray().ravel(), q, K)
    sel["BM25"] = top_idx(d["bm"], q, K)
    if ce is not None:
        sel["Cross-Encoder"] = order_by(ce_scores(q, list(pool)), pool)
    sel["SBERT-Only"] = pool
    sel["SBERT+BM25 Fusion"] = order_by(mu * d["s"] + (1 - mu) * minmax(d["bm"][pool]), pool)
    sel["Trope-Only"] = order_by(d["ato"], pool)
    sel["Hybrid (lambda=%.1f)" % lam] = order_by(lam * d["s"] + (1 - lam) * d["ato"], pool)
    return {k: v[:K] for k, v in sel.items()}


def dcg(rel):
    return float(sum(r / np.log2(i + 2) for i, r in enumerate(rel)))


def metrics(top, q, thr=THR):
    sh = SH[q]
    rel = (sh[top] >= thr).astype(int)
    R = int((sh >= thr).sum())
    idcg = dcg([1] * min(R, K))
    tq = Tq[q]
    seen = set().union(*(tagsets[j] for j in top))
    return dict(ndcg=dcg(rel) / idcg if idcg else 0.0,
                mrr=next((1 / i for i, x in enumerate(rel, 1) if x), 0.0),
                p=rel.sum() / K,
                seren=sum(1 for j, x in zip(top, rel) if x and not pop[j]) / K,
                tmr=len(tq & seen) / len(tq) if tq else float("nan"))


def run(queries, **kw):
    res = {}
    for q in queries:
        for name, top in sys_lists(q, **kw).items():
            res.setdefault(name, []).append(metrics(top, q))
    return res


def mean(res, name, key):
    return float(np.nanmean([m[key] for m in res[name]]))


# ---------------------------------------------------------------- tune lambda / mu on validation
lams = [round(x * 0.1, 1) for x in range(11)]
lam_ndcg = {}
for lam in lams:
    ms = [metrics(order_by(lam * D[q]["s"] + (1 - lam) * D[q]["ato"], D[q]["pool"])[:K], q)["ndcg"] for q in val]
    lam_ndcg[lam] = float(np.mean(ms))
best_lam = max(lam_ndcg, key=lam_ndcg.get)
mu_ndcg = {}
for mu in lams:
    ms = [metrics(order_by(mu * D[q]["s"] + (1 - mu) * minmax(D[q]["bm"][D[q]["pool"]]), D[q]["pool"])[:K], q)["ndcg"] for q in val]
    mu_ndcg[mu] = float(np.mean(ms))
best_mu = max(mu_ndcg, key=mu_ndcg.get)
print(f"lambda sweep (validation NDCG@{K}):", {k: round(v, 3) for k, v in lam_ndcg.items()})
print(f"selected lambda={best_lam}, mu={best_mu}")

# ---------------------------------------------------------------- test
lists = {q: sys_lists(q, lam=best_lam, mu=best_mu) for q in test}
res = {}
for q in test:
    for name, top in lists[q].items():
        res.setdefault(name, []).append(metrics(top, q))
hyb = "Hybrid (lambda=%.1f)" % best_lam
rng = np.random.default_rng(42)


def ci(name):
    v = np.array([m["ndcg"] for m in res[name]])
    bs = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(1000)]
    return np.percentile(bs, [2.5, 97.5])


md = [f"# Evaluasi lengkap BookForYou\n",
      f"Katalog {N} buku, {len(vocab)} tag. Query diambil dari split held-out BART; validasi {len(val)} / test {len(test)} query. "
      f"Relevan jika berbagi >= {THR} tag asli. Metrik pada K = {K}, pool SBERT = {POOL}.\n",
      f"Tag per query rata-rata {len(all_gen) / len(keep):.2f}, hallucination rate "
      f"{100 * sum(t not in vs for t in all_gen) / len(all_gen):.2f}%.\n",
      "## Tabel utama (test)\n", "| Sistem | NDCG@10 [95% CI] | MRR | P@10 | Serendipity | Trope Match |", "|---|---|---|---|---|---|"]
for name in res:
    lo, hi = ci(name)
    tm = f"{mean(res, name, 'tmr'):.3f}" if name.startswith(("Trope", "Hybrid")) else "-"
    md.append(f"| {name} | {mean(res, name, 'ndcg'):.3f} [{lo:.3f}, {hi:.3f}] | {mean(res, name, 'mrr'):.3f} | "
              f"{mean(res, name, 'p'):.3f} | {mean(res, name, 'seren'):.3f} | {tm} |")

h = np.array([m["ndcg"] for m in res[hyb]])
md += ["", "## Uji signifikansi (Wilcoxon dua sisi, Hybrid vs sistem lain, NDCG@10)\n", "| Pembanding | selisih rata-rata | p |", "|---|---|---|"]
for name in res:
    if name == hyb:
        continue
    o = np.array([m["ndcg"] for m in res[name]])
    p = wilcoxon(h, o).pvalue if np.any(h != o) else 1.0
    md.append(f"| {name} | {h.mean() - o.mean():+.3f} | {p:.4f} |")

s, t, hh = (mean(res, n, "ndcg") for n in ("SBERT-Only", "Trope-Only", hyb))
md += ["", "## Kontribusi ATO vs SBERT\n",
       f"- NDCG@10: SBERT-only {s:.4f}, Trope-only {t:.4f}, Hybrid {hh:.4f}",
       f"- Kenaikan total (Hybrid - SBERT-only) = {hh - s:.4f}"]
if hh > s:
    md += [f"- Porsi ATO = (Trope-only - SBERT-only) / total = {100 * (t - s) / (hh - s):.1f}%",
           f"- Porsi SBERT (tiebreaker) = (Hybrid - Trope-only) / total = {100 * (hh - t) / (hh - s):.1f}%"]
md += ["", "## Sensitivitas lambda (validasi, NDCG@10)\n", "| lambda | " + " | ".join(str(x) for x in lams) + " |",
       "|---|" + "---|" * len(lams), "| NDCG | " + " | ".join(f"{lam_ndcg[x]:.3f}" for x in lams) + " |",
       f"\nlambda terpilih = {best_lam}; mu SBERT+BM25 terpilih = {best_mu}."]

md += ["", "## Ambang relevansi (test, NDCG@10)\n", "| Ambang | SBERT-Only | Trope-Only | Hybrid | Trope - SBERT |", "|---|---|---|---|---|"]
for thr in range(5, 11):
    row = {}
    for n in ("SBERT-Only", "Trope-Only", hyb):
        vals = [metrics(lists[q][n], q, thr)["ndcg"] for q in test if (SH[q] >= thr).any()]
        row[n] = float(np.mean(vals))
    md.append(f"| {thr} | {row['SBERT-Only']:.3f} | {row['Trope-Only']:.3f} | {row[hyb]:.3f} | {row['Trope-Only'] - row['SBERT-Only']:+.3f} |")

md += ["", "## Catatan\n",
       "- Relevansi berasal dari tag komunitas, bukan rating atau penilaian pembaca. Tag yang sama dipakai ATO untuk meranking, jadi NDCG sistem trope cenderung tinggi.",
       "- Vector search memakai perkalian matriks eksak (numpy), bukan Qdrant; hasilnya sama untuk top-100.",
       "- Popularitas = jumlah rating Goodreads; buku populer = 1.000 teratas di katalog."]
open("results/eval.md", "w", encoding="utf-8").write("\n".join(md) + "\n")
json.dump(dict(lam=best_lam, mu=best_mu, lam_sweep=lam_ndcg, n_val=len(val), n_test=len(test),
               ndcg={n: mean(res, n, "ndcg") for n in res}), open("results/eval.json", "w"), indent=1)
print("\n".join(md))
