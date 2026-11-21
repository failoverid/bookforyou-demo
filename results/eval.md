# Evaluasi lengkap BookForYou

Katalog 22656 buku, 150 tag. Query diambil dari split held-out BART; validasi 500 / test 500 query. Relevan jika berbagi >= 5 tag asli. Metrik pada K = 10, pool SBERT = 100.

Tag per query rata-rata 6.84, hallucination rate 0.09%.

## Tabel utama (test)

| Sistem | NDCG@10 [95% CI] | MRR | P@10 | Serendipity | Trope Match |
|---|---|---|---|---|---|
| Popularity | 0.581 [0.552, 0.612] | 0.743 | 0.562 | 0.274 | - |
| TF-IDF | 0.463 [0.434, 0.490] | 0.640 | 0.449 | 0.422 | - |
| BM25 | 0.537 [0.510, 0.565] | 0.721 | 0.518 | 0.484 | - |
| Cross-Encoder | 0.523 [0.496, 0.551] | 0.695 | 0.511 | 0.477 | - |
| SBERT-Only | 0.530 [0.499, 0.557] | 0.701 | 0.517 | 0.486 | - |
| SBERT+BM25 Fusion | 0.577 [0.550, 0.604] | 0.748 | 0.561 | 0.523 | - |
| Trope-Only | 0.788 [0.762, 0.815] | 0.862 | 0.780 | 0.713 | 0.996 |
| Hybrid (lambda=0.6) | 0.788 [0.760, 0.815] | 0.862 | 0.780 | 0.713 | 0.996 |

## Uji signifikansi (Wilcoxon dua sisi, Hybrid vs sistem lain, NDCG@10)

| Pembanding | selisih rata-rata | p |
|---|---|---|
| Popularity | +0.207 | 0.0000 |
| TF-IDF | +0.324 | 0.0000 |
| BM25 | +0.251 | 0.0000 |
| Cross-Encoder | +0.264 | 0.0000 |
| SBERT-Only | +0.257 | 0.0000 |
| SBERT+BM25 Fusion | +0.211 | 0.0000 |
| Trope-Only | -0.001 | 0.7655 |

## Kontribusi ATO vs SBERT

- NDCG@10: SBERT-only 0.5303, Trope-only 0.7885, Hybrid 0.7875
- Kenaikan total (Hybrid - SBERT-only) = 0.2572
- Porsi ATO = (Trope-only - SBERT-only) / total = 100.4%
- Porsi SBERT (tiebreaker) = (Hybrid - Trope-only) / total = -0.4%

## Sensitivitas lambda (validasi, NDCG@10)

| lambda | 0.0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 1.0 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| NDCG | 0.762 | 0.762 | 0.762 | 0.762 | 0.762 | 0.762 | 0.762 | 0.759 | 0.749 | 0.690 | 0.516 |

lambda terpilih = 0.6; mu SBERT+BM25 terpilih = 0.5.

## Ambang relevansi (test, NDCG@10)

| Ambang | SBERT-Only | Trope-Only | Hybrid | Trope - SBERT |
|---|---|---|---|---|
| 5 | 0.530 | 0.788 | 0.788 | +0.258 |
| 6 | 0.457 | 0.730 | 0.728 | +0.273 |
| 7 | 0.388 | 0.675 | 0.673 | +0.287 |
| 8 | 0.330 | 0.608 | 0.608 | +0.278 |
| 9 | 0.293 | 0.552 | 0.552 | +0.259 |
| 10 | 0.245 | 0.494 | 0.495 | +0.249 |

## Catatan

- Relevansi berasal dari tag komunitas, bukan rating atau penilaian pembaca. Tag yang sama dipakai ATO untuk meranking, jadi NDCG sistem trope cenderung tinggi.
- Vector search memakai perkalian matriks eksak (numpy), bukan Qdrant; hasilnya sama untuk top-100.
- Popularitas = jumlah rating Goodreads; buku populer = 1.000 teratas di katalog.
