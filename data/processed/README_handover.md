# Handover: Korpus PubMed Bersih

Berkas ini menjelaskan korpus yang dihasilkan notebook `notebooks/pubmed_screening_assistant.ipynb` (Cell 1 sampai 7). Cakupan pekerjaan berhenti di data bersih: pengambilan, penyaringan, cleaning, dan preprocessing biomedis. Feature extraction, clustering, topic modeling, similarity search, dan summarization dikerjakan terpisah.

Dibuat: 2026-11-20. Sumber: NCBI E-utilities (esearch + efetch), bukan scraping HTML.

## Berkas

| Berkas | Isi |
|---|---|
| `corpus_handover.csv` | Korpus final, 500 baris, 13 kolom, UTF-8 |
| `corpus_clean.csv` | Sumber ekspor, masih menyimpan kolom bantu |
| `corpus_filtered.csv` | Hasil penyaringan dan penyeimbangan sebelum cleaning |

Muat korpus:

```python
import pandas as pd
df = pd.read_csv("data/processed/corpus_handover.csv", dtype=str, keep_default_na=False)
df["pub_year"] = pd.to_numeric(df["pub_year"], errors="coerce").astype("Int32")
```

## Kamus Kolom

| Kolom | Tipe | Keterangan |
|---|---|---|
| `pmid` | string | PubMed ID, unik |
| `domain` | string | `immunotherapy_cancer` atau `cardiovascular_disease` |
| `title` | string | Judul asli dari XML |
| `abstract` | string | Abstrak asli, label bagian masih ada |
| `title_clean` | string | Judul setelah cleaning |
| `text_clean` | string | `title_clean` + abstrak bersih, tanda baca dipertahankan untuk pemisahan kalimat |
| `text_lemma` | string | Hasil scispaCy: lowercase, stopword dibuang, lemmatisasi |
| `abstract_words` | int32 | Jumlah kata abstrak asli |
| `clean_words` | int32 | Jumlah kata `text_clean` |
| `lemma_words` | int32 | Jumlah token `text_lemma` |
| `journal` | string | Nama jurnal |
| `pub_year` | Int32 | Tahun terbit, 2020 sampai 2026, tanpa nilai kosong |
| `pub_types` | string | Tipe publikasi PubMed, dipisah titik koma |

Pakai `text_clean` untuk Word2Vec/GloVe dan TextRank (butuh batas kalimat). Pakai `text_lemma` untuk TF-IDF dan LDA (bag-of-words).

## Jejak Penyusutan Data

| Tahap | Immunotherapy | Cardiovascular | Total |
|---|---|---|---|
| Hit PubMed dengan filter lengkap | 80.499 | 90.378 | 160.877 |
| PMID diambil (retmax 350 per domain) | 350 | 350 | 700 |
| Record berhasil di-parse dari XML | 350 | 348 | 698 |
| Abstrak kosong | 1 | 18 | 19 |
| Abstrak di bawah 50 kata | 16 | 67 | 83 |
| Duplikat PMID | 0 | 0 | 0 |
| Kandidat valid | 334 | 281 | 615 |
| Dipotong kuota 250 | 84 | 31 | 115 |
| **Korpus final** | **250** | **250** | **500** |

Dua PMID cardiovascular tidak dikembalikan EFetch. Tidak bisa dibedakan apakah record ditarik atau berubah status.

## Filter PubMed yang Dipakai

| Filter | Nilai |
|---|---|
| Domain 1 | `immunotherapy[Title/Abstract] AND cancer[Title/Abstract]` |
| Domain 2 | `"cardiovascular disease"[Title/Abstract]` |
| Abstrak tersedia | `hasabstract[text]` |
| Bahasa | `english[Language]` |
| Tipe | `"journal article"[Publication Type]` |
| Tanggal | `2020/01/01` sampai `2026/12/31` |
| Urutan | relevance |

## Aturan Cleaning (`text_clean`)

1. Decode entity HTML/XML dua kali, lalu normalisasi Unicode NFKC.
2. En dash atau em dash di antara angka menjadi `-` (rentang tahun). Em dash dan en dash lainnya menjadi spasi, supaya tidak lahir token palsu seperti `loss-has`.
3. Hyphen non-breaking (U+2011) dan varian dash lain menjadi `-`, sehingga `anti‑tumor` menjadi `anti-tumor`.
4. Tag HTML/XML, URL, dan sitasi berkurung angka (`[12]`, `[3-5]`) dibuang.
5. Label bagian abstrak terstruktur dibuang memakai daftar kepala label medis (`BACKGROUND:`, `METHODS:`, `PURPOSE OF REVIEW:`, `AREAS COVERED:`, `KEY MESSAGES:`, dan sejenisnya). Prefiks nama jurnal seperti `JACC:` di judul sengaja dipertahankan.
6. Karakter kontrol dan zero-width dibuang, spasi berlebih dirapatkan.
7. Tanda baca akhir kalimat dipertahankan.

## Aturan Preprocessing (`text_lemma`)

Model: `en_core_sci_sm` 0.5.4 (scispaCy), pipeline `tok2vec, tagger, attribute_ruler, lemmatizer`, parser dan NER dimatikan.

1. Token tanda baca dan spasi dibuang.
2. Token yang hanya berisi angka dibuang (termasuk rentang seperti `10-14`).
3. Stopword bahasa Inggris dari spaCy dibuang.
4. Stopword domain tambahan dibuang: study, studies, patient, patients, result, results, method, methods, conclusion, conclusions, background, objective, objectives, aim, aims, group, groups, data, analysis, analyses, significant, significantly, compared, comparison, outcome, outcomes, review, reviews, trial, trials, cohort, article, paper, report, reported, reports, included, including.
5. Token dilemakan dengan lemmatizer spaCy.

Istilah klinis berhubung tetap utuh: `anti-pd-1`, `non-small`, `meta-analysis`.

## Batasan yang Perlu Disebut di Laporan

- **Bias pemotongan kuota**: 115 record valid dibuang hanya karena urutan relevance, bukan karena kualitas (84 immunotherapy, 31 cardiovascular). Komposisi sub-tema bisa bergeser akibat ini.
- **Filter kata kunci** bekerja pada Title/Abstract, bukan MeSH, sehingga recall bergantung pada pilihan kata penulis.
- **`journal article`** di PubMed juga mencakup review dan meta-analisis, jadi tipe ini tidak homogen.
- **Lemmatisasi** memakai model yang dilatih dengan spaCy 3.7.4 dan dimuat di spaCy 3.8.18. Model memberi peringatan kompatibilitas; hasil uji pada 500 record konsisten, tetapi ini bukan konfigurasi resmi yang diuji vendor.
- **Instalasi tidak standar**: paket `scispacy` gagal dipasang di Python 3.14, sehingga model dipasang dengan `--no-deps` dan `config.cfg`-nya diperbaiki. Kalau lingkungan dibangun ulang, langkah ini harus diulang.
- **Stopword domain dikurasi manual**, bukan daftar baku. Daftarnya ada di atas dan di Cell 6, sehingga bisa diubah dan dijalankan ulang.
- **Tidak ada label relevansi manusia**, jadi korpus ini tidak bisa dipakai mengukur akurasi pencarian tanpa anotasi tambahan.

## Reproduksi

Jalankan notebook dari Cell 1 sampai 7 berurutan. Cell 1 sampai 3 memanggil API NCBI (jeda 0,35 detik per request); Cell 4 sampai 7 bekerja dari berkas di disk. Semua sel punya asersi yang akan gagal kalau jumlah record atau skema berubah.
