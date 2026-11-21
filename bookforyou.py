"""BookForYou worked example (Kanchi et al., RecSys '26), scaled down to a few candidate books.

Real parts : text cleaning, 4-stage tag curation, SBERT (all-MiniLM-L6-v2, mean pooling, 384-d),
             ATO, late fusion (lambda = 0.5), leave-one-out relevance, all metrics.
Stage 1a   : --bart given    uses t_q written in the data file (offline, default)
             --bart zeroshot uses BART (bart-large-mnli) to pick tropes from the vocabulary
Data       : data/sample.json (hand-written illustration, not Goodreads data). Edit it or pass --data.

Run: python bookforyou.py [--data FILE] [--bart given|zeroshot] [--k 3] > hasil_bookforyou.md
"""
import argparse
import html
import json
import math
import os
import re
import sys

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
import torch
from transformers import AutoModel, AutoTokenizer

sys.stdout.reconfigure(encoding="utf-8")

LAMBDA = 0.5      # paper: chosen by grid search on the validation split
K = 3             # paper: 10 (from a 100-book pool); here 3 (from a 5-book pool)
MIN_SHARED = 5    # a book is relevant if it shares >= 5 original tags with the query
MAX_TOKENS = 256  # all-MiniLM-L6-v2 max_seq_length
MODEL = "sentence-transformers/all-MiniLM-L6-v2"

VOCAB, STATUS_TAGS, KEEP_PLURAL, QUERY_BOOKS, CATALOG = set(), set(), set(), {}, {}  # filled by load_data()


def load_data(path):
    global VOCAB, STATUS_TAGS, KEEP_PLURAL, QUERY_BOOKS, CATALOG
    d = json.load(open(path, encoding="utf-8"))
    VOCAB, STATUS_TAGS, KEEP_PLURAL = set(d["vocab"]), set(d["status_tags"]), set(d["keep_plural"])
    QUERY_BOOKS, CATALOG = d["query_books"], d["catalog"]
    for q in QUERY_BOOKS.values():
        q["t_q"] = set(q.get("t_q", []))


# --- Stage 1a: BART ------------------------------------------------------------------------
def bart_tags(synopsis, top_n=5):
    """Zero-shot BART (bart-large-mnli) scores every vocabulary trope against the synopsis.
    shortcut: the paper fine-tunes bart-base to *generate* tags; that checkpoint is not public,
    so this ranks vocabulary tropes instead. Swap in the fine-tuned model when you have it."""
    from transformers import pipeline
    global _zs
    if "_zs" not in globals():
        _zs = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    out = _zs(clean_synopsis(synopsis), sorted(VOCAB), hypothesis_template="This story has the trope: {}.",
              multi_label=True)
    return set(out["labels"][:top_n])  # labels come sorted by score


# --- preprocessing: synopsis -----------------------------------------------------------------
def clean_synopsis(text):
    text = html.unescape(re.sub(r"<[^>]+>", " ", text))       # strip HTML, decode &amp;
    text = re.sub(r"[^a-z0-9\s'-]", " ", text.lower())         # drop symbols, lowercase
    return re.sub(r"\s+", " ", text).strip()


# --- preprocessing: community tags (4 stages, same order as the paper) ------------------------
def singular_word(w):
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if re.search(r"(ss|us|is)$", w):
        return w
    if re.search(r"(ch|sh|x|z)es$", w):
        return w[:-2]
    return w[:-1] if w.endswith("s") else w


def stage3_singular(tags):
    out = []
    for t in tags:
        keep = re.sub(r"[^a-z]+", " ", t).strip() in KEEP_PLURAL
        out.append(t if keep else re.sub(r"[a-z]+$", lambda m: singular_word(m.group()), t))
    return out


def stage4_chars(tags):
    return [re.sub(r"[^a-z0-9]+", " ", t).strip() for t in tags]


def stage2_split(tags):
    out = []
    for t in tags:
        legit = stage4_chars(stage3_singular([t]))[0] in VOCAB  # science-fiction stays, mystery-thriller splits
        out += [t] if "-" not in t or legit else t.split("-")
    return out


def curate(raw):
    s1 = [t.lower() for t in raw if t.lower() not in STATUS_TAGS]
    s2 = stage2_split(s1)
    s3 = stage3_singular(s2)
    s4 = stage4_chars(s3)
    final = sorted({t for t in s4 if t in VOCAB})
    return s1, s2, s3, s4, final


# --- SBERT: token embeddings -> mean pooling -> L2 norm -> cosine ----------------------------
tok = AutoTokenizer.from_pretrained(MODEL)
enc_model = AutoModel.from_pretrained(MODEL).eval()


def embed(text):
    enc = tok(text, truncation=True, max_length=MAX_TOKENS, return_tensors="pt")
    with torch.no_grad():
        h = enc_model(**enc).last_hidden_state[0]           # (n_tokens, 384): one vector per token
    mean = h.mean(0)                                         # mean pooling -> sentence vector
    return torch.nn.functional.normalize(mean, dim=0), enc["input_ids"].shape[1], tok.convert_ids_to_tokens(enc["input_ids"][0])


# --- metrics ---------------------------------------------------------------------------------
def dcg(rels):
    return sum(r / math.log2(i + 2) for i, r in enumerate(rels))


def metrics(order, rel, pop, t_q, tags, k):
    r = [rel[b] for b in order]
    ideal = sorted(rel.values(), reverse=True)
    idcg = dcg(ideal[:k])
    seen = set().union(*(tags[b] for b in order[:k]))
    return dict(
        p=sum(r[:k]) / k,
        mrr=next((1 / i for i, x in enumerate(r, 1) if x), 0.0),
        dcg=dcg(r[:k]), idcg=idcg, ndcg=dcg(r[:k]) / idcg if idcg else 0.0,
        seren=sum(1 for b in order[:k] if rel[b] and not pop[b]) / k,
        tmr=len(t_q & seen) / len(t_q),
    )


def table(head, rows):
    print("| " + " | ".join(head) + " |")
    print("|" + "---|" * len(head))
    for r in rows:
        print("| " + " | ".join(str(c) for c in r) + " |")
    print()


def main():
    ap = argparse.ArgumentParser(description="BookForYou worked example")
    ap.add_argument("--data", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "sample.json"))
    ap.add_argument("--bart", choices=["given", "zeroshot"], default="given",
                    help="given: use t_q from the data file; zeroshot: extract with BART (downloads ~1.6 GB once)")
    ap.add_argument("--k", type=int, default=K)
    args = ap.parse_args()
    load_data(args.data)
    k = args.k
    if args.bart == "zeroshot":
        for d in QUERY_BOOKS.values():
            d["t_q"] = bart_tags(d["synopsis"])
    # ===== 1. method
    print("# BookForYou: contoh perhitungan lengkap\n")
    print("## 1. Proposed Method\n")
    print("- **Stage 1a (BART-base, 139M)**: sinopsis query -> himpunan tag trope T_q.")
    print("- **Stage 1b (SBERT all-MiniLM-L6-v2, 384-d)**: sinopsis query -> vektor -> kandidat terdekat (paper: top-100 dari 42.350 buku, di sini 5 buku).")
    print("- **Stage 2 (late fusion)**: Score(b) = lambda * SBERT(q,b) + (1 - lambda) * ATO(T_q,T_b), ATO = |T_q ∩ T_b| / |T_q|, lambda = 0,5.")
    print("- Catatan: " + ("Stage 1a memakai BART zero-shot (bart-large-mnli) yang memilih trope dari kamus; paper memakai bart-base hasil fine-tune yang checkpoint-nya tidak tersedia."
          if args.bart == "zeroshot" else "Stage 1a memakai T_q dari file data (checkpoint BART hasil fine-tune paper tidak tersedia; jalankan dengan --bart zeroshot untuk BART asli).")
          + " Stage 1b dan 2 dijalankan dengan model asli.\n")

    # ===== 2. input
    print("## 2. Contoh Input\n")
    print("**Query user** (menyukai 2 buku, sinopsisnya dipakai sebagai query):\n")
    for q, d in QUERY_BOOKS.items():
        print(f"- {q}: {html.unescape(re.sub('<[^>]+>', '', d['synopsis']))}")
    print("\n**Katalog kandidat:**\n")
    table(["Buku", "Judul", "Sinopsis", "Tag katalog (mentah)"],
          [(b, c["title"], c["synopsis"], ", ".join(c["tags"])) for b, c in CATALOG.items()])

    # ===== 3. preprocessing
    print("## 3. Preprocessing\n")
    query_text = " ".join(clean_synopsis(d["synopsis"]) for d in QUERY_BOOKS.values())
    q_vec, n_tok, q_pieces = embed(query_text)
    print("**3a. Sinopsis** (hapus HTML/simbol -> lowercase -> tokenisasi WordPiece -> potong maks. "
          f"{MAX_TOKENS} token)\n")
    qn = list(QUERY_BOOKS)[-1]
    print(f"- Sebelum ({qn}): `{QUERY_BOOKS[qn]['synopsis']}`")
    print(f"- Sesudah ({qn}): `{clean_synopsis(QUERY_BOOKS[qn]['synopsis'])}`")
    names = "+".join(QUERY_BOOKS)
    print(f"- Token query gabungan {names}: {n_tok} (termasuk [CLS] dan [SEP]); contoh: {q_pieces[:12]} ...\n")

    print("**3b. Tag komunitas** (4 tahap, contoh dua buku):\n")
    rows = []
    qn, cn = list(QUERY_BOOKS)[-1], max(CATALOG, key=lambda b: len(CATALOG[b]["tags"]))
    for name, raw in ((qn, QUERY_BOOKS[qn]["tags"]), (cn, CATALOG[cn]["tags"])):
        s1, s2, s3, s4, final = curate(raw)
        rows += [(name, "mentah", ", ".join(raw)), (name, "1. hapus status baca", ", ".join(s1)),
                 (name, "2. pisah kata majemuk", ", ".join(s2)), (name, "3. singularization", ", ".join(s3)),
                 (name, "4. normalisasi karakter", ", ".join(s4)), (name, "masuk kamus trope", ", ".join(final))]
    table(["Buku", "Tahap", "Hasil"], rows)

    tags = {b: set(curate(c["tags"])[4]) for b, c in CATALOG.items()}
    q_real = set().union(*(curate(d["tags"])[4] for d in QUERY_BOOKS.values()))
    t_q = set().union(*(d["t_q"] for d in QUERY_BOOKS.values()))
    assert t_q <= VOCAB, "BART tags must come from the vocabulary (hallucination check)"

    # ===== 4. calculation
    print("## 4. Perhitungan Metode\n")
    print("**Stage 1a: tag trope hasil BART (stand-in)**\n")
    for q, d in QUERY_BOOKS.items():
        print(f"- T_{q} = {sorted(d['t_q'])}")
    print(f"- T_q = gabungan semua T_Q* = {sorted(t_q)}, |T_q| = {len(t_q)}\n")

    print("**Stage 1b: SBERT** (vektor token -> rata-rata = vektor kalimat -> cosine)\n")
    print(f"Vektor query (4 dimensi pertama dari 384): {[round(x, 4) for x in q_vec[:4].tolist()]}, norma = {q_vec.norm():.4f}\n")
    sbert, rows = {}, []
    for b, c in CATALOG.items():
        v, n, _ = embed(clean_synopsis(c["synopsis"]))
        sbert[b] = float(q_vec @ v)
        rows.append((b, n, [round(x, 4) for x in v[:4].tolist()], f"{sbert[b]:.4f}"))
    table(["Buku", "Jumlah token", "Vektor (4 dim pertama)", "cos(query, buku)"], rows)
    pool = sorted(CATALOG, key=lambda b: -sbert[b])
    print(f"Urutan kandidat dari retrieval SBERT: {pool}\n")

    print("**Stage 2: ATO dan Late Fusion** (lambda = 0,5)\n")
    ato = {b: len(t_q & tags[b]) / len(t_q) for b in CATALOG}
    score = {b: LAMBDA * sbert[b] + (1 - LAMBDA) * ato[b] for b in CATALOG}
    table(["Buku", "T_q ∩ T_b", "ATO", "0,5 x SBERT", "0,5 x ATO", "Skor akhir"],
          [(b, f"{len(t_q & tags[b])}/{len(t_q)}", f"{ato[b]:.3f}", f"{LAMBDA * sbert[b]:.4f}",
            f"{(1 - LAMBDA) * ato[b]:.4f}", f"{score[b]:.4f}") for b in pool])

    # ===== 5. output
    print("## 5. Hasil Output Rekomendasi\n")
    hybrid = sorted(CATALOG, key=lambda b: -score[b])
    rows = []
    for i, b in enumerate(hybrid, 1):
        m = sorted(t_q & tags[b])
        why = f"cocok pada {len(m)} trope: {', '.join(m)}" if m else "tidak ada trope yang cocok"
        rows.append((i, f"{b} - {CATALOG[b]['title']}", f"{score[b]:.4f}", f"{ato[b]:.3f}", f"{sbert[b]:.4f}", why))
    table(["Rank", "Buku", "Skor", "ATO", "SBERT", "Penjelasan"], rows)
    ties = {}
    for b in CATALOG:
        ties.setdefault(round(ato[b], 6), []).append(b)
    for a, bs in sorted(ties.items(), reverse=True):
        if len(bs) > 1:
            order = sorted(bs, key=lambda b: -sbert[b])
            print(f"- ATO seri di {a:.3f} untuk {bs}: SBERT menjadi pemecah seri, urutan {order}.")
    print()

    # ===== 6. evaluation
    print("## 6. Evaluasi dan Contoh Perhitungannya\n")
    print(f"**Relevansi** (relevan = 1 jika berbagi >= {MIN_SHARED} tag asli dengan buku query; tag asli query = gabungan semua buku query, {len(q_real)} tag)\n")
    shared = {b: q_real & tags[b] for b in CATALOG}
    rel = {b: int(len(shared[b]) >= MIN_SHARED) for b in CATALOG}
    table(["Buku", "Jumlah tag sama", "Tag sama", "Relevan"],
          [(b, len(shared[b]), ", ".join(sorted(shared[b])), rel[b]) for b in CATALOG])

    systems = {"SBERT-only (lambda=1)": pool,
               "Trope-only (lambda=0)": sorted(pool, key=lambda b: -ato[b]),  # stable: ties keep SBERT retrieval order
               "Hybrid (lambda=0,5)": hybrid}
    pop = {b: CATALOG[b]["popular"] for b in CATALOG}
    res = {n: metrics(o, rel, pop, t_q, tags, k) for n, o in systems.items()}

    print(f"**Perhitungan manual untuk Hybrid, K = {k}**\n")
    r = [rel[b] for b in hybrid]
    table(["Rank i", "Buku", "rel_i", "rel_i / log2(i+1)"],
          [(i, hybrid[i - 1], r[i - 1], f"{r[i - 1] / math.log2(i + 1):.4f}") for i in range(1, k + 1)])
    m = res["Hybrid (lambda=0,5)"]
    ideal = sorted(rel.values(), reverse=True)[:k]
    print(f"- Precision@{k} = {sum(r[:k])}/{k} = {m['p']:.3f}")
    print(f"- MRR = 1/(rank relevan pertama) = {m['mrr']:.3f}")
    print(f"- DCG@{k} = {m['dcg']:.4f}; IDCG@{k} (urutan ideal {ideal}) = {m['idcg']:.4f}; NDCG@{k} = DCG/IDCG = {m['ndcg']:.4f}")
    rec_ser = [b for b in hybrid[:k] if rel[b] and not pop[b]]
    print(f"- Serendipity@{k} = (relevan dan bukan buku populer) / {k} = {len(rec_ser)}/{k} = {m['seren']:.3f}  (buku populer: {sorted(b for b in pop if pop[b])})")
    seen = set().union(*(tags[b] for b in hybrid[:k]))
    print(f"- Trope Match Rate = |T_q ∩ tag semua rekomendasi| / |T_q| = {len(t_q & seen)}/{len(t_q)} = {m['tmr']:.3f}\n")

    print(f"**Perbandingan tiga sistem (K = {k})**\n")
    table(["Sistem", "Urutan", f"P@{k}", "MRR", f"NDCG@{k}", f"Serendipity@{k}", "Trope Match"],
          [(n, " ".join(systems[n]), f"{v['p']:.3f}", f"{v['mrr']:.3f}", f"{v['ndcg']:.4f}",
            f"{v['seren']:.3f}", f"{v['tmr']:.3f}") for n, v in res.items()])

    s, t, h = (res[n]["ndcg"] for n in systems)
    print("**Kontribusi ATO vs SBERT** (cara paper: bagi kenaikan NDCG Hybrid terhadap SBERT-only)\n")
    print(f"- Kenaikan total = NDCG Hybrid - NDCG SBERT-only = {h:.4f} - {s:.4f} = {h - s:.4f}")
    if h > s:
        print(f"- Porsi ATO = (Trope-only - SBERT-only) / total = ({t:.4f} - {s:.4f}) / {h - s:.4f} = {100 * (t - s) / (h - s):.1f}%")
        print(f"- Porsi SBERT (tiebreaker) = (Hybrid - Trope-only) / total = ({h:.4f} - {t:.4f}) / {h - s:.4f} = {100 * (h - t) / (h - s):.1f}%")
    print("- Karena kandidat dari retrieval sudah urut menurut SBERT, Trope-only sudah memakai SBERT untuk memecah skor ATO yang seri. "
          "Porsi SBERT hanya muncul bila selisih cosine mengalahkan selisih ATO (minimal 1/|T_q|) sehingga urutan berubah.")
    ps, pt, ph = 0.546, 0.783, 0.789
    print(f"- Rumus yang sama pada angka paper (NDCG@10 test set, 489 query, dibulatkan 3 desimal): "
          f"ATO = ({pt} - {ps}) / ({ph} - {ps}) = {100 * (pt - ps) / (ph - ps):.1f}%, "
          f"SBERT = ({ph} - {pt}) / ({ph} - {ps}) = {100 * (ph - pt) / (ph - ps):.1f}%. "
          "Paper melaporkan 97,8% dan 2,2% dari skor yang belum dibulatkan.\n")

    print("**Sensitivitas ambang relevansi** (paper menguji 5 sampai 10 tag)\n")
    rows = []
    for thr in range(3, 8):
        rl = {b: int(len(shared[b]) >= thr) for b in CATALOG}
        if not any(rl.values()):
            rows.append((thr, "-", "-", "-")); continue
        rows.append((thr, *(f"{metrics(o, rl, pop, t_q, tags, k)['ndcg']:.4f}" for o in systems.values())))
    table(["Ambang"] + [f"NDCG@{k} {n.split()[0]}" for n in systems], rows)

    print("**Analisis kritis**\n")
    print("- Relevansi diukur dari kesamaan tag komunitas, dan tag yang sama itulah yang dipakai ATO untuk meranking. Sistem trope-based jadi diuji dengan kriteria yang sejalan dengan sinyalnya sendiri, sehingga NDCG-nya cenderung tinggi.")
    print("- Belum ada rating atau penilaian pembaca. Skor tinggi belum membuktikan pembaca menyukai rekomendasinya, dan kegunaan penjelasan trope belum diuji ke pengguna.")
    print(f"- Contoh ini hanya {len(CATALOG)} buku kandidat dan 1 query, jadi angkanya untuk memperlihatkan cara menghitung, bukan untuk membuktikan performa.")

    # self-checks
    assert curate(["to-read", "mystery-thriller", "Found-Families", "enemies-to-lovers", "Dragons"])[4] == \
        ["dragon", "enemies to lovers", "found family", "mystery", "thriller"]
    assert abs(q_vec.norm() - 1) < 1e-5
    assert all(abs(score[b] - (LAMBDA * sbert[b] + (1 - LAMBDA) * ato[b])) < 1e-12 for b in CATALOG)
    assert all(0 <= v["ndcg"] <= 1 for v in res.values())


if __name__ == "__main__":
    main()
