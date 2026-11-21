"""Stage 1a: fine-tune BART-base to generate trope tags from a book description.

    python finetune_bart.py [--epochs 3] [--out models/bart-tropes]

Init checkpoint: KamilAin/bart-base-booksum (as in the paper). Target = the book's vocabulary tags
(most-shelved first, up to 15) joined by ", ". The best epoch by F1@15 on the eval split is kept.
"""
import argparse
import json
import math
import os
import random
import time

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--init", default="KamilAin/bart-base-booksum")
ap.add_argument("--out", default="models/bart-tropes")
ap.add_argument("--epochs", type=int, default=3)
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--accum", type=int, default=4, help="effective batch = batch * accum (paper: 32)")
ap.add_argument("--lr", type=float, default=5e-5)
ap.add_argument("--max-in", type=int, default=256)
ap.add_argument("--max-tags", type=int, default=15)
ap.add_argument("--eval-n", type=int, default=800, help="eval pairs used for checkpoint selection")
ap.add_argument("--limit", type=int, default=0, help="debug: only use this many train pairs")
args = ap.parse_args()

random.seed(42)
torch.manual_seed(42)
dev = "cuda" if torch.cuda.is_available() else "cpu"
books = {}
for line in open("data/books.jsonl", encoding="utf-8"):
    r = json.loads(line)
    books[r["id"]] = r
split = json.load(open("data/split.json"))
train = [books[i] for i in split["train"]]
ev = [books[i] for i in split["eval"]][: args.eval_n]
if args.limit:
    train = train[: args.limit]
vocab = set(json.load(open("data/vocab.json")))

tok = AutoTokenizer.from_pretrained(args.init)
model = AutoModelForSeq2SeqLM.from_pretrained(args.init).to(dev)


def target(r):
    return ", ".join(r["tags"][: args.max_tags])


def batches(rows, bs, shuffle):
    idx = list(range(len(rows)))
    if shuffle:
        random.shuffle(idx)
    for i in range(0, len(idx), bs):
        yield [rows[j] for j in idx[i:i + bs]]


def generate(descs, num_beams=4):
    enc = tok(descs, truncation=True, max_length=args.max_in, padding=True, return_tensors="pt").to(dev)
    with torch.no_grad(), torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
        out = model.generate(**enc, num_beams=num_beams, max_new_tokens=64)
    res = []
    for s in tok.batch_decode(out, skip_special_tokens=True):
        tags = []
        for t in s.split(","):
            t = t.strip().lower()
            if t and t not in tags:
                tags.append(t)
        res.append(tags)
    return res


def evaluate():
    model.eval()
    p5, f1, hall, gen_n, n_tags = [], [], 0, 0, 0
    for b in batches(ev, 32, False):
        for r, g in zip(b, generate([r["description"] for r in b])):
            gold = set(r["tags"])
            top = g[:15]
            hit = len(set(top) & gold)
            p = hit / len(top) if top else 0.0
            rec = hit / len(gold)
            f1.append(2 * p * rec / (p + rec) if p + rec else 0.0)
            p5.append(sum(t in gold for t in g[:5]) / 5)
            hall += sum(t not in vocab for t in g)
            gen_n += len(g)
    model.train()
    return dict(P5=sum(p5) / len(p5), F1_15=sum(f1) / len(f1), hallucination=hall / max(gen_n, 1), avg_tags=gen_n / len(ev))


opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
steps_per_epoch = math.ceil(len(train) / (args.batch * args.accum))
total = steps_per_epoch * args.epochs
warm = max(1, int(0.03 * total))
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))

best, step, t0 = -1.0, 0, time.time()
model.train()
for ep in range(1, args.epochs + 1):
    run, k = 0.0, 0
    for i, b in enumerate(batches(train, args.batch, True), 1):
        x = tok([r["description"] for r in b], truncation=True, max_length=args.max_in, padding=True, return_tensors="pt").to(dev)
        y = tok(text_target=[target(r) for r in b], truncation=True, max_length=64, padding=True, return_tensors="pt").input_ids.to(dev)
        y[y == tok.pad_token_id] = -100
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
            loss = model(**x, labels=y).loss / args.accum
        loss.backward()
        run, k = run + loss.item() * args.accum, k + 1
        if i % args.accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad(); step += 1
            if step % 50 == 0:
                print(f"ep {ep} step {step}/{total} loss {run / k:.4f} lr {sched.get_last_lr()[0]:.2e} {time.time() - t0:.0f}s", flush=True)
                run, k = 0.0, 0
    m = evaluate()
    print(f"== epoch {ep}: {m}", flush=True)
    if m["F1_15"] > best:
        best = m["F1_15"]
        model.save_pretrained(args.out); tok.save_pretrained(args.out)
        json.dump(dict(epoch=ep, **m), open(os.path.join(args.out, "metrics.json"), "w"))
print("best F1@15:", best)
