# BookForYou: contoh perhitungan lengkap

## 1. Proposed Method

- **Stage 1a (BART-base, 139M)**: sinopsis query -> himpunan tag trope T_q.
- **Stage 1b (SBERT all-MiniLM-L6-v2, 384-d)**: sinopsis query -> vektor -> kandidat terdekat (paper: top-100 dari 42.350 buku, di sini 5 buku).
- **Stage 2 (late fusion)**: Score(b) = lambda * SBERT(q,b) + (1 - lambda) * ATO(T_q,T_b), ATO = |T_q ∩ T_b| / |T_q|, lambda = 0,5.
- Catatan: Stage 1a memakai BART zero-shot (bart-large-mnli) yang memilih trope dari kamus; paper memakai bart-base hasil fine-tune yang checkpoint-nya tidak tersedia. Stage 1b dan 2 dijalankan dengan model asli.

## 2. Contoh Input

**Query user** (menyukai 2 buku, sinopsisnya dipakai sebagai query):

- Q1: Fourteen-year-old orphan Elara learns she is the chosen one and is sent to a hidden magic school. Her rival, a proud prince, slowly becomes her ally as a slow burn romance grows, while a dragon threatens the royal court.
- Q2: Two sworn enemies from rival houses are forced to share a throne. Amid betrayal & intrigue, a found family of exiles helps them survive, and hatred turns to love.

**Katalog kandidat:**

| Buku | Judul | Sinopsis | Tag katalog (mentah) |
|---|---|---|---|
| A | The Quiet Detective of Vael | A retired detective investigates a murder inside a royal court, where every courtier hides a secret and a betrayal waits behind each door. | to-read, royal-court, betrayal, mystery, detective, murder, kindle |
| B | Dragon Rider Academy | A young mage enrolls at an academy for dragon riders, where a rival prince challenges her at every turn. The rivalry softens into a slow burn romance as an ancient dragon wakes beneath the school. | to-read, Dragons, fantasy, young-adult, war, witches |
| C | The Hollow Crown | Two rival heirs are bound by a treaty marriage and must rule together. Betrayal stalks the royal court, and a found family of exiles gives them a reason to trust each other. | favorites, royal-court, enemies-to-lovers, found-families, betrayal, fantasy, romance, mystery-thriller |
| D | Chosen of the Storm | An orphan farm girl is named the chosen one by a prophecy and trains at a school of storm magic. A dragon-hunting prince opposes her, but a slow burn alliance forms. | chosen-one, magic-school, slow-burn, dragons, enemies-to-lovers, fantasy, young-adult |
| E | Summer of the Island Witch | A teenage witch spends the summer on a quiet island, befriends a young dragon, and finds a gentle romance that takes its time to bloom. | witches, dragons, slow-burn, fantasy, young-adult, romance, to-read |

## 3. Preprocessing

**3a. Sinopsis** (hapus HTML/simbol -> lowercase -> tokenisasi WordPiece -> potong maks. 256 token)

- Sebelum (Q2): `Two sworn enemies from rival houses are forced to share a throne. Amid betrayal &amp; intrigue, a found family of exiles helps them survive, and hatred turns to love.`
- Sesudah (Q2): `two sworn enemies from rival houses are forced to share a throne amid betrayal intrigue a found family of exiles helps them survive and hatred turns to love`
- Token query gabungan Q1+Q2: 75 (termasuk [CLS] dan [SEP]); contoh: ['[CLS]', 'fourteen', '-', 'year', '-', 'old', 'orphan', 'el', '##ara', 'learns', 'she', 'is'] ...

**3b. Tag komunitas** (4 tahap, contoh dua buku):

| Buku | Tahap | Hasil |
|---|---|---|
| Q2 | mentah | kindle, fantasy, romance, enemies-to-lovers, found-families, royal-court, betrayal, slow-burn, mystery-thriller |
| Q2 | 1. hapus status baca | fantasy, romance, enemies-to-lovers, found-families, royal-court, betrayal, slow-burn, mystery-thriller |
| Q2 | 2. pisah kata majemuk | fantasy, romance, enemies-to-lovers, found-families, royal-court, betrayal, slow-burn, mystery, thriller |
| Q2 | 3. singularization | fantasy, romance, enemies-to-lovers, found-family, royal-court, betrayal, slow-burn, mystery, thriller |
| Q2 | 4. normalisasi karakter | fantasy, romance, enemies to lovers, found family, royal court, betrayal, slow burn, mystery, thriller |
| Q2 | masuk kamus trope | betrayal, enemies to lovers, fantasy, found family, mystery, romance, royal court, slow burn, thriller |
| C | mentah | favorites, royal-court, enemies-to-lovers, found-families, betrayal, fantasy, romance, mystery-thriller |
| C | 1. hapus status baca | royal-court, enemies-to-lovers, found-families, betrayal, fantasy, romance, mystery-thriller |
| C | 2. pisah kata majemuk | royal-court, enemies-to-lovers, found-families, betrayal, fantasy, romance, mystery, thriller |
| C | 3. singularization | royal-court, enemies-to-lovers, found-family, betrayal, fantasy, romance, mystery, thriller |
| C | 4. normalisasi karakter | royal court, enemies to lovers, found family, betrayal, fantasy, romance, mystery, thriller |
| C | masuk kamus trope | betrayal, enemies to lovers, fantasy, found family, mystery, romance, royal court, thriller |

## 4. Perhitungan Metode

**Stage 1a: tag trope hasil BART (stand-in)**

- T_Q1 = ['chosen one', 'coming of age', 'enemies to lovers', 'romance', 'slow burn']
- T_Q2 = ['betrayal', 'enemies to lovers', 'found family', 'mystery', 'romance']
- T_q = gabungan semua T_Q* = ['betrayal', 'chosen one', 'coming of age', 'enemies to lovers', 'found family', 'mystery', 'romance', 'slow burn'], |T_q| = 8

**Stage 1b: SBERT** (vektor token -> rata-rata = vektor kalimat -> cosine)

Vektor query (4 dimensi pertama dari 384): [-0.0669, 0.0653, 0.0216, 0.0191], norma = 1.0000

| Buku | Jumlah token | Vektor (4 dim pertama) | cos(query, buku) |
|---|---|---|---|
| A | 26 | [-0.0168, 0.0595, -0.0212, -0.018] | 0.2435 |
| B | 39 | [-0.0793, 0.0746, 0.0218, 0.0479] | 0.6312 |
| C | 34 | [-0.0588, 0.0198, 0.0211, -0.0233] | 0.4427 |
| D | 35 | [-0.0974, 0.0823, 0.026, 0.0755] | 0.7009 |
| E | 30 | [0.0182, 0.0181, 0.0151, 0.0121] | 0.5427 |

Urutan kandidat dari retrieval SBERT: ['D', 'B', 'E', 'C', 'A']

**Stage 2: ATO dan Late Fusion** (lambda = 0,5)

| Buku | T_q ∩ T_b | ATO | 0,5 x SBERT | 0,5 x ATO | Skor akhir |
|---|---|---|---|---|---|
| D | 3/8 | 0.375 | 0.3504 | 0.1875 | 0.5379 |
| B | 0/8 | 0.000 | 0.3156 | 0.0000 | 0.3156 |
| E | 2/8 | 0.250 | 0.2713 | 0.1250 | 0.3963 |
| C | 5/8 | 0.625 | 0.2213 | 0.3125 | 0.5338 |
| A | 2/8 | 0.250 | 0.1218 | 0.1250 | 0.2468 |

## 5. Hasil Output Rekomendasi

| Rank | Buku | Skor | ATO | SBERT | Penjelasan |
|---|---|---|---|---|---|
| 1 | D - Chosen of the Storm | 0.5379 | 0.375 | 0.7009 | cocok pada 3 trope: chosen one, enemies to lovers, slow burn |
| 2 | C - The Hollow Crown | 0.5338 | 0.625 | 0.4427 | cocok pada 5 trope: betrayal, enemies to lovers, found family, mystery, romance |
| 3 | E - Summer of the Island Witch | 0.3963 | 0.250 | 0.5427 | cocok pada 2 trope: romance, slow burn |
| 4 | B - Dragon Rider Academy | 0.3156 | 0.000 | 0.6312 | tidak ada trope yang cocok |
| 5 | A - The Quiet Detective of Vael | 0.2468 | 0.250 | 0.2435 | cocok pada 2 trope: betrayal, mystery |

- ATO seri di 0.250 untuk ['A', 'E']: SBERT menjadi pemecah seri, urutan ['E', 'A'].

## 6. Evaluasi dan Contoh Perhitungannya

**Relevansi** (relevan = 1 jika berbagi >= 5 tag asli dengan buku query; tag asli query = gabungan semua buku query, 13 tag)

| Buku | Jumlah tag sama | Tag sama | Relevan |
|---|---|---|---|
| A | 3 | betrayal, mystery, royal court | 0 |
| B | 3 | dragon, fantasy, young adult | 0 |
| C | 8 | betrayal, enemies to lovers, fantasy, found family, mystery, romance, royal court, thriller | 1 |
| D | 7 | chosen one, dragon, enemies to lovers, fantasy, magic school, slow burn, young adult | 1 |
| E | 5 | dragon, fantasy, romance, slow burn, young adult | 1 |

**Perhitungan manual untuk Hybrid, K = 3**

| Rank i | Buku | rel_i | rel_i / log2(i+1) |
|---|---|---|---|
| 1 | D | 1 | 1.0000 |
| 2 | C | 1 | 0.6309 |
| 3 | E | 1 | 0.5000 |

- Precision@3 = 3/3 = 1.000
- MRR = 1/(rank relevan pertama) = 1.000
- DCG@3 = 2.1309; IDCG@3 (urutan ideal [1, 1, 1]) = 2.1309; NDCG@3 = DCG/IDCG = 1.0000
- Serendipity@3 = (relevan dan bukan buku populer) / 3 = 1/3 = 0.333  (buku populer: ['D', 'E'])
- Trope Match Rate = |T_q ∩ tag semua rekomendasi| / |T_q| = 7/8 = 0.875

**Perbandingan tiga sistem (K = 3)**

| Sistem | Urutan | P@3 | MRR | NDCG@3 | Serendipity@3 | Trope Match |
|---|---|---|---|---|---|---|
| SBERT-only (lambda=1) | D B E C A | 0.667 | 1.000 | 0.7039 | 0.000 | 0.500 |
| Trope-only (lambda=0) | C D E A B | 1.000 | 1.000 | 1.0000 | 0.333 | 0.875 |
| Hybrid (lambda=0,5) | D C E B A | 1.000 | 1.000 | 1.0000 | 0.333 | 0.875 |

**Kontribusi ATO vs SBERT** (cara paper: bagi kenaikan NDCG Hybrid terhadap SBERT-only)

- Kenaikan total = NDCG Hybrid - NDCG SBERT-only = 1.0000 - 0.7039 = 0.2961
- Porsi ATO = (Trope-only - SBERT-only) / total = (1.0000 - 0.7039) / 0.2961 = 100.0%
- Porsi SBERT (tiebreaker) = (Hybrid - Trope-only) / total = (1.0000 - 1.0000) / 0.2961 = 0.0%
- Karena kandidat dari retrieval sudah urut menurut SBERT, Trope-only sudah memakai SBERT untuk memecah skor ATO yang seri. Porsi SBERT hanya muncul bila selisih cosine mengalahkan selisih ATO (minimal 1/|T_q|) sehingga urutan berubah.
- Rumus yang sama pada angka paper (NDCG@10 test set, 489 query, dibulatkan 3 desimal): ATO = (0.783 - 0.546) / (0.789 - 0.546) = 97.5%, SBERT = (0.789 - 0.783) / (0.789 - 0.546) = 2.5%. Paper melaporkan 97,8% dan 2,2% dari skor yang belum dibulatkan.

**Sensitivitas ambang relevansi** (paper menguji 5 sampai 10 tag)

| Ambang | NDCG@3 SBERT-only | NDCG@3 Trope-only | NDCG@3 Hybrid |
|---|---|---|---|
| 3 | 1.0000 | 1.0000 | 1.0000 |
| 4 | 0.7039 | 1.0000 | 1.0000 |
| 5 | 0.7039 | 1.0000 | 1.0000 |
| 6 | 0.6131 | 1.0000 | 1.0000 |
| 7 | 0.6131 | 1.0000 | 1.0000 |

**Analisis kritis**

- Relevansi diukur dari kesamaan tag komunitas, dan tag yang sama itulah yang dipakai ATO untuk meranking. Sistem trope-based jadi diuji dengan kriteria yang sejalan dengan sinyalnya sendiri, sehingga NDCG-nya cenderung tinggi.
- Belum ada rating atau penilaian pembaca. Skor tinggi belum membuktikan pembaca menyukai rekomendasinya, dan kegunaan penjelasan trope belum diuji ke pengguna.
- Contoh ini hanya 5 buku kandidat dan 1 query, jadi angkanya untuk memperlihatkan cara menghitung, bukan untuk membuktikan performa.
