# BookForYou

Implementasi BookForYou (Kanchi, Hazarika, Thirumoorthy. RecSys '26, https://doi.org/10.1145/3773078.3831814):
rekomendasi buku berbasis konten. BART membuat tag trope dari sinopsis, SBERT mencari kandidat, lalu
skor digabung lewat late fusion: `skor = lambda * SBERT + (1 - lambda) * ATO`, dengan
`ATO = |T_q ∩ T_b| / |T_q|`.

Repo ini punya dua bagian.

## 1. Contoh hitung 5 buku (untuk laporan)

```bash
pip install -r requirements.txt
python bookforyou.py > hasil_bookforyou.md
```

Mencetak 6 bagian: metode, input, preprocessing, perhitungan, hasil rekomendasi, evaluasi. Data ada di
`data/sample.json` (tulisan tangan, bisa diubah). Opsi: `--bart zeroshot` (BART zero-shot, unduh sekitar 1,6 GB),
`--data FILE`, `--k N`, `--lam X` (bobot SBERT, ATO mendapat 1 - X). Hasil contoh: `hasil_bookforyou.md` dan `hasil_bart_zeroshot.md`.

## 2. Pipeline penuh (data Goodreads, fine-tune BART, evaluasi 8 sistem)

```bash
# 1. unduh 4 file genre dari UCSD Goodreads ke data/raw/ (total sekitar 1 GB)
base=https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre
for g in fantasy_paranormal romance mystery_thriller_crime young_adult; do
  curl -L -o data/raw/goodreads_books_$g.json.gz $base/goodreads_books_$g.json.gz
done

python prepare_data.py          # kurasi tag 4 tahap, kamus 150 trope, katalog, split 80/20 seed 42
python finetune_bart.py         # Stage 1a: fine-tune bart-base (butuh GPU, ~6 GB cukup)
python evaluate.py              # Stage 1b + 2 dan evaluasi leave-one-out, tulis results/eval.md
                                # lambda dipilih otomatis di validasi; paksa dengan --lam 0.5
```

| File | Isi |
|---|---|
| `tags.py` | pembersihan tag (4 tahap + penggabungan sinonim) dan teks |
| `prepare_data.py` | bangun `data/books.jsonl`, `vocab.json`, `split.json` |
| `finetune_bart.py` | fine-tune `KamilAin/bart-base-booksum` untuk menghasilkan tag |
| `evaluate.py` | SBERT, pool top-100, 8 sistem, metrik, Wilcoxon, bootstrap CI, sapuan lambda |
| `bookforyou.py` | contoh hitung 5 buku |

## Hasil (500 query test, K = 10, kandidat top-100)

Detail lengkap, uji signifikansi, sapuan lambda, dan ambang relevansi ada di `results/eval.md`.

| Sistem | NDCG@10 | MRR | P@10 | Serendipity |
|---|---|---|---|---|
| Popularity | 0,581 | 0,743 | 0,562 | 0,274 |
| TF-IDF | 0,463 | 0,640 | 0,449 | 0,422 |
| BM25 | 0,537 | 0,721 | 0,518 | 0,484 |
| Cross-Encoder | 0,523 | 0,695 | 0,511 | 0,477 |
| SBERT-Only | 0,530 | 0,701 | 0,517 | 0,486 |
| SBERT+BM25 | 0,577 | 0,748 | 0,561 | 0,523 |
| Trope-Only | 0,788 | 0,862 | 0,780 | 0,713 |
| Hybrid (lambda 0,6) | 0,788 | 0,862 | 0,780 | 0,713 |

- Pola utama paper terulang: re-ranking dengan ATO menaikkan NDCG@10 dari 0,530 ke 0,788 (paper 0,546 ke 0,789), p < 0,001.
- Berbeda dari paper: Hybrid tidak lebih baik dari Trope-Only (selisih -0,001, p = 0,77). Nilai lambda dari 0,0 sampai 0,6 hampir sama di validasi (0,762), jadi SBERT tidak menambah apa-apa di sini dan porsi ATO menjadi 100%. Paper melaporkan tambahan kecil 0,006 (p = 0,017) dan porsi 97,8% / 2,2%.
- Lambda terpilih 0,6 oleh sapuan validasi, bukan 0,5 seperti paper, karena nilainya nyaris seri.
- Popularity (0,581) lebih tinggi dari SBERT-Only, tidak seperti paper (0,384). Buku populer punya lebih banyak tag, sehingga lebih mudah berbagi >= 5 tag dengan query.
- Relevansi memakai tag komunitas yang sama dengan sinyal ATO, jadi skor trope-based cenderung tinggi. Belum ada rating atau penilaian pembaca.
- Fine-tune BART: 3 epoch, epoch terbaik ke-2 (P@5 0,499, F1@15 0,526, hallucination 0,02%).

## Beda dengan paper

- Data: 46.000 buku dari 4 genre UCSD Goodreads (paper: 45.293 buku dari daftar genre dan "best of").
  Saringan sinopsis dan ambang rak ditentukan sendiri, jadi kamus 150 trope tidak identik dengan paper.
- Kurasi tag: ditambah penggabungan sinonim (misalnya `ya` dan `young adult`) dan daftar kata non-konten
  (`giveaway`, `arc`, dan sejenisnya), karena rak Goodreads mentah penuh label status.
- Fine-tune BART hanya beberapa epoch (paper 10 epoch, RTX A2000).
- Vector search memakai perkalian matriks eksak, bukan Qdrant. Hasil top-100 sama.
- Popularitas = jumlah rating Goodreads.
- Tidak ada uji kosakata di luar data training BART (vocabulary independence).
