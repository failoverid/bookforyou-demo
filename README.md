# BookForYou: contoh perhitungan

Implementasi kecil dari BookForYou (Kanchi dkk., RecSys '26): rekomendasi buku berbasis konten dengan
tag trope dari BART, kandidat dari SBERT, lalu digabung lewat late fusion (SBERT + ATO).
Skrip ini menghitung satu contoh dari awal sampai akhir (preprocessing, skor, ranking, 5 metrik) dan
mencetaknya sebagai markdown.

## Menjalankan

```bash
pip install -r requirements.txt
python bookforyou.py > hasil_bookforyou.md
```

Model all-MiniLM-L6-v2 (sekitar 90 MB) diunduh otomatis pada run pertama.

| Opsi | Fungsi |
|---|---|
| `--bart given` (default) | T_q dibaca dari `data/sample.json`, jalan offline setelah model SBERT terunduh |
| `--bart zeroshot` | BART (`facebook/bart-large-mnli`, sekitar 1,6 GB) memilih 5 trope per sinopsis dari kamus |
| `--data FILE` | pakai data sendiri, formatnya sama dengan `data/sample.json` |
| `--k N` | panjang daftar rekomendasi untuk metrik (default 3) |

## Data

`data/sample.json` berisi kamus trope, 2 buku query, dan 5 buku kandidat (A-E). Semuanya ditulis tangan
untuk ilustrasi, bukan data Goodreads. Ubah sinopsis atau tag di file itu untuk mencoba kasus lain.

## Yang sama dan berbeda dari paper

- Sama: SBERT all-MiniLM-L6-v2 (mean pooling, 384 dimensi), 4 tahap pembersihan tag, ATO = |T_q ∩ T_b| / |T_q|,
  skor = 0,5 x SBERT + 0,5 x ATO, relevan jika berbagi >= 5 tag asli, metrik P@K, MRR, NDCG@K, Serendipity, Trope Match Rate.
- Berbeda: paper memakai bart-base yang di-fine-tune untuk menghasilkan tag. Checkpoint itu tidak tersedia,
  jadi `--bart zeroshot` memilih trope dari kamus dengan BART zero-shot, dan hasilnya lebih kasar.
- Berbeda: kamus berisi 24 trope (paper 150), katalog 5 buku (paper 42.350), K = 3 (paper 10).
  Penanda "populer" untuk Serendipity diisi manual (paper: 1.000 buku terpopuler).

Referensi: Kanchi, Hazarika, Thirumoorthy. BookForYou: Leveraging BART-Generated Narrative Tropes for
Content-Based Book Recommendations. RecSys '26. https://doi.org/10.1145/3773078.3831814
