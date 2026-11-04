# Klasifikasi Gambar Dokumen — EfficientNet-B0

Proyek Supervised Machine Learning untuk klasifikasi 16 kelas dokumen hasil scan (subset RVL-CDIP) menggunakan PyTorch dan fine-tuning EfficientNet-B0 pretrained ImageNet.

## Struktur repository

```text
kaggle-document-classification/
├── README.md
├── requirements.txt
├── .gitignore
├── data/.gitkeep
├── checkpoint/.gitkeep
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_modeling.ipynb
│   └── 03_submission.ipynb
├── scripts/
│   ├── __init__.py
│   ├── config.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── submissions/.gitkeep
└── reports/figures/.gitkeep
```

## Dataset

Dataset kompetisi berisi 32.003 train, 7.994 test, dan 16 kelas. File metadata yang diperlukan adalah `train.csv`, `test.csv`, `sample_submission.csv`. Data gambar TIFF tidak disertakan di GitHub.

Unduh dengan Kaggle API setelah menerima aturan kompetisi:
```bash
pip install kaggle
kaggle competitions download -c <competition-slug> -p data
```
Ekstrak arsip ke `data/` sehingga CSV dan folder gambar train/test tersedia. Unduhan manual: https://www.kaggle.com/t/ec93bc56cc9842959bb6728723d23d29

## Setup

```bash
git clone <URL_REPOSITORY>
cd kaggle-document-classification
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```
Di Kaggle, sebaiknya gunakan PyTorch/Torchvision bawaan image Kaggle dan jangan memasang ulang versi CUDA secara sembarang.

## Pretrained weights

Sediakan `efficientnet_b0_rwightman-7f5810bc.pth` dari bobot Torchvision ImageNet. Tambahkan file itu sebagai Kaggle Input untuk penggunaan Internet OFF. Atur environment variable `DOC_WEIGHTS_PATH` ke path file tersebut jika tidak ditemukan otomatis.

## Workflow

1. EDA: distribusi kelas, contoh dokumen, ukuran, aspect ratio, intensitas, dan audit TIFF.
2. Preprocessing: Pillow dengan fallback tifffile; RGB; resize proporsional 224×224 dengan padding putih; normalisasi ImageNet.
3. Augmentasi train: rotasi ±5°, affine ringan, brightness/contrast, Gaussian blur. Tidak menggunakan flip horizontal/vertikal.
4. Split: stratified 90:10, seed 42.
5. Modeling: EfficientNet-B0 pretrained, classifier 16 kelas, CrossEntropyLoss label smoothing 0,05, AdamW, ReduceLROnPlateau, AMP, gradient clipping, early stopping.
6. Evaluasi: accuracy, macro F1, classification report, confusion matrix.
7. Submission: tiga file CSV, satu per eksperimen.

## Eksperimen

| Hyperparameter | Baseline | Tuning 1 | Tuning 2 |
|---|---:|---:|---:|
| Dropout | 0,30 | 0,40 | 0,35 |
| Backbone LR | 0,0001 | 0,00005 | 0,000075 |
| Head LR | 0,0008 | 0,0003 | 0,0005 |
| Weight decay | 0,0001 | 0,0005 | 0,00025 |
| Epoch maksimal | 10 | 12 | 12 |
| Batch size | 16 | 16 | 16 |

Setiap eksperimen dimulai dari pretrained ImageNet yang sama. Checkpoint terbaik disimpan berdasarkan validation accuracy.

## CLI

Jalankan dari root repository:
```bash
python -m scripts.train --experiment baseline
python -m scripts.train --experiment tuning1
python -m scripts.train --experiment tuning2
python -m scripts.train --experiment all
python -m scripts.evaluate --experiment all
python -m scripts.predict --experiment all
python -m scripts.predict --experiment all --tta
```

Opsi data dan resource:
```bash
python -m scripts.train --help
python -m scripts.train --data-dir data --batch-size 16 --num-workers 2
```

## Import scripts di Kaggle (Internet OFF)

Cara paling aman adalah mengunggah repository sebagai Kaggle Dataset, lalu menambahkannya melalui **Add Input** bersama dataset kompetisi dan dataset pretrained weights.

Cell pertama di Kaggle:
```python
from pathlib import Path
import sys

matches = list(Path("/kaggle/input").rglob("scripts/config.py"))
if not matches:
    raise FileNotFoundError("Tambahkan dataset repository melalui Add Input.")

REPO_ROOT = matches[0].parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.config import CLASS_NAMES, EXPERIMENTS
from scripts.dataset import read_tiff_with_fallback
from scripts.model import build_model
print("Import berhasil:", REPO_ROOT)
```

Folder `/kaggle/input` bersifat read-only. Sebelum menjalankan modul, arahkan output ke `/kaggle/working`:
```python
import os
os.environ["DOC_DATA_DIR"] = "/kaggle/input/<folder-dataset-kompetisi>"
os.environ["DOC_CHECKPOINT_DIR"] = "/kaggle/working/checkpoint"
os.environ["DOC_SUBMISSION_DIR"] = "/kaggle/working/submissions"
os.environ["DOC_REPORT_DIR"] = "/kaggle/working/reports"
os.environ["DOC_WEIGHTS_PATH"] = "/kaggle/input/<folder-bobot>/<nama-file-bobot>.pth"
```
Environment variable harus diset **sebelum** import `scripts.config`, karena konfigurasi dibaca saat modul diimpor. Jika sudah telanjur import, restart kernel atau reload modul.

Kemudian:
```python
%cd $REPO_ROOT
!python -m scripts.train --experiment all
!python -m scripts.evaluate --experiment all
!python -m scripts.predict --experiment all
```

Jika Internet ON, alternatifnya clone repository:
```python
!git clone https://github.com/<username>/kaggle-document-classification.git
import sys
sys.path.insert(0, "/kaggle/working/kaggle-document-classification")
from scripts.dataset import DocumentDataset
```

## Hasil

| Eksperimen | Validation Accuracy | Macro F1 | Public LB | Private LB |
|---|---:|---:|---:|---:|
| Baseline | 0,85442 | 0,85348 | 0,84608 | — |
| Tuning 1 | 0,85098 | 0,85006 | 0,83647 | — |
| Tuning 2 | Isi setelah training | Isi setelah training | Isi setelah submit | Isi setelah kompetisi |

## Interpretasi

Baseline saat ini menjadi acuan terbaik: validation accuracy 0,85442 dan public score 0,84608. Tuning 1 menghasilkan validation accuracy 0,85098 dan public score 0,83647. Tuning 2 belum dinilai.

Analisis confusion matrix harus menyebut pasangan kelas yang paling sering tertukar berdasarkan angka aktual. Periksa terutama form/questionnaire, memo/letter, dan scientific_publication/scientific_report. Jelaskan kelas mudah/sulit berdasarkan precision, recall, dan F1 per kelas.

Preprocessing mempertahankan rasio aspek dengan padding putih. Augmentasi ringan ditujukan untuk variasi kemiringan dan kualitas scan. Karena eksperimen tidak mengisolasi setiap transformasi, pengaruh masing-masing augmentasi tidak boleh diklaim sebagai sebab tunggal.

Keterbatasan: resolusi 224 dapat menghilangkan detail teks kecil; validasi memakai satu split; public leaderboard belum tentu mewakili private test. Pengembangan berikutnya dapat menguji resolusi lebih tinggi, EfficientNet-B2/B3, augmentasi perspektif ringan, TTA, atau ensemble jika waktu dan GPU memungkinkan.

## Lisensi dan kredit

Dataset mengikuti ketentuan sumber RVL-CDIP dan aturan kompetisi Kaggle. Jangan mendistribusikan ulang gambar tanpa memeriksa lisensinya. Kode repository tidak menyertakan dataset. EfficientNet-B0 berasal dari Torchvision.
