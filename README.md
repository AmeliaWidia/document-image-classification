# Klasifikasi Gambar Dokumen dengan EfficientNet-B0

Proyek ini menggunakan PyTorch untuk mengklasifikasikan dokumen hasil scan ke dalam 16 kelas. Model yang digunakan adalah **EfficientNet-B0** dengan bobot pretrained ImageNet.

Repository ini memuat tiga eksperimen training EfficientNet-B0:
1. Baseline
2. Hyperparameter Tuning 1
3. Hyperparameter Tuning 2

Ketiga eksperimen menggunakan EfficientNet-B0. Repository ini tidak mencakup model ConvNeXt.

## Struktur Repository

```text
kaggle-document-classification/
â”œâ”€â”€ README.md
â”œâ”€â”€ requirements.txt
â”œâ”€â”€ .gitignore
â”œâ”€â”€ data/
â”‚   â””â”€â”€ .gitkeep
â”œâ”€â”€ checkpoint/
â”‚   â””â”€â”€ efficientnet_b0/
â”‚       â”œâ”€â”€ best_document_efficientnet_b0_baseline.pth
â”‚       â”œâ”€â”€ best_document_efficientnet_b0_tuning.pth
â”‚       â””â”€â”€ best_document_efficientnet_b0_tuning2.pth
â”œâ”€â”€ notebooks/
â”‚   â”œâ”€â”€ 01_eda.ipynb
â”‚   â”œâ”€â”€ 02_modeling.ipynb
â”‚   â””â”€â”€ 03_submission.ipynb
â”œâ”€â”€ scripts/
â”‚   â”œâ”€â”€ __init__.py
â”‚   â”œâ”€â”€ config.py
â”‚   â”œâ”€â”€ dataset.py
â”‚   â”œâ”€â”€ model.py
â”‚   â”œâ”€â”€ train.py
â”‚   â”œâ”€â”€ evaluate.py
â”‚   â””â”€â”€ predict.py
â”œâ”€â”€ submissions/
â”‚   â””â”€â”€ .gitkeep
â””â”€â”€ reports/
    â”œâ”€â”€ EXPERIMENTS.md
    â””â”€â”€ figures/
        â””â”€â”€ .gitkeep
```

> Folder `checkpoint/efficientnet_b0/` berisi checkpoint terbaik dari ketiga eksperimen. 

## Dataset

Dataset kompetisi terdiri dari:
- 32.003 gambar training
- 7.994 gambar testing
- 16 kelas dokumen

File metadata yang diperlukan:
- `train.csv`
- `test.csv`
- `sample_submission.csv`

Gambar disimpan dalam format TIFF dan diakses melalui Kaggle Input. 

## Setup

Clone repository:

```bash
git clone <URL_REPOSITORY>
cd kaggle-document-classification
```

Buat virtual environment dan instal dependensi:

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

Di Kaggle, gunakan PyTorch dan Torchvision yang sudah tersedia selama versinya sesuai. Periksa kompatibilitas GPU sebelum memasang atau memperbarui PyTorch dan CUDA.

## Pretrained Weights

Model EfficientNet-B0 menggunakan bobot pretrained ImageNet.

File bobot yang digunakan:

```text
efficientnet_b0_rwightman-7f5810bc.pth
```

Jika notebook dijalankan dengan Internet OFF, tambahkan dataset yang berisi pretrained weights melalui **Add Input** di Kaggle. Jika bobot tidak terdeteksi otomatis, isi `DOC_WEIGHTS_PATH` dengan lokasi file `.pth` tersebut.

## Workflow

1. **EDA:** memeriksa distribusi kelas, contoh gambar, ukuran gambar, aspect ratio, intensitas, dan kondisi file TIFF.
2. **Preprocessing:** membaca TIFF menggunakan Pillow dengan fallback `tifffile`, mengubah gambar menjadi RGB, melakukan resize proporsional ke ukuran 224 Ã— 224 dengan padding putih, lalu normalisasi ImageNet.
3. **Augmentasi:** rotasi ringan, affine, perubahan brightness/contrast, dan Gaussian blur. Tidak menggunakan horizontal flip atau vertical flip.
4. **Data split:** stratified split 90:10 dengan seed 42.
5. **Training:** fine-tuning EfficientNet-B0 dengan classifier 16 kelas, CrossEntropyLoss, AdamW, mixed precision, gradient clipping, dan early stopping sesuai implementasi eksperimen.
6. **Evaluasi:** mengukur validation accuracy, macro F1, classification report, dan confusion matrix.
7. **Submission:** menghasilkan file CSV untuk masing-masing eksperimen.

## Tiga Eksperimen Training

Ketiga eksperimen menggunakan arsitektur EfficientNet-B0 dan pretrained ImageNet yang sama. Perbedaan eksperimen terletak pada konfigurasi hyperparameter.

| Hyperparameter | Baseline | Tuning 1 | Tuning 2 |
|---|---:|---:|---:|
| Dropout | 0,30 | 0,40 | 0,35 |
| Backbone learning rate | 0,0001 | 0,00005 | 0,000075 |
| Head learning rate | 0,0008 | 0,0003 | 0,0005 |
| Weight decay | 0,0001 | 0,0005 | 0,00025 |
| Epoch maksimal | 10 | 12 | 12 |
| Batch size | 16 | 16 | 16 |

Checkpoint terbaik setiap eksperimen disimpan dengan nama berbeda. Dengan begitu, model dapat dievaluasi atau digunakan untuk prediksi tanpa perlu menjalankan training lagi.

### Hasil Validation

| Eksperimen | Validation Accuracy | Macro F1 | Best Epoch |
|---|---:|---:|---:|
| Baseline | 0,855358 | 0,854976 | 8 |
| Hyperparameter Tuning 1 | 0,845986 | 0,845834 | 12 |
| Hyperparameter Tuning 2 | 0,860669* | 0,860052 | 12 |

## Menjalankan Training

Training Baseline:

```bash
python -m scripts.train --experiment baseline
```

Training Hyperparameter Tuning 1:

```bash
python -m scripts.train --experiment tuning1
```

Training Hyperparameter Tuning 2:

```bash
python -m scripts.train --experiment tuning2
```

Menjalankan ketiga eksperimen secara berurutan:

```bash
python -m scripts.train --experiment all
```

Periksa opsi training yang tersedia:

```bash
python -m scripts.train --help
```

Contoh pengaturan direktori data dan DataLoader:

```bash
python -m scripts.train --data-dir data --batch-size 16 --num-workers 2
```

## Evaluasi

Untuk mengevaluasi ketiga eksperimen:

```bash
python -m scripts.evaluate --experiment all
```

Script evaluasi menampilkan validation accuracy, macro F1, classification report, dan confusion matrix.

## Membuat Submission

Membuat submission untuk ketiga eksperimen:

```bash
python -m scripts.predict --experiment all
```

Jika ingin menggunakan Test-Time Augmentation (TTA):

```bash
python -m scripts.predict --experiment all --tta
```

File CSV tersimpan di folder `submissions/`. Sebelum diunggah ke Kaggle, periksa nama kolom, jumlah baris, urutan ID, dan rentang label agar sesuai dengan `sample_submission.csv`.

## Menjalankan Repository di Kaggle

Untuk menjalankan project di Kaggle, unggah repository sebagai Kaggle Dataset. Setelah itu, tambahkan repository dan sumber data yang dibutuhkan melalui **Add Input**:
1. Dataset kompetisi document image classification.
2. Dataset pretrained EfficientNet-B0 apabila menggunakan Internet OFF.

Folder `/kaggle/input` hanya dapat dibaca. Simpan checkpoint baru, hasil evaluasi, dan file submission di `/kaggle/working`.

Contoh cell untuk menemukan repository dan dataset:

```python
from pathlib import Path
import os
import sys

INPUT_ROOT = Path("/kaggle/input")

repo_candidates = list(INPUT_ROOT.rglob("scripts/config.py"))
if not repo_candidates:
    raise FileNotFoundError(
        "Repository belum ditambahkan melalui Kaggle Add Input."
    )

REPO_ROOT = repo_candidates[0].parent.parent

data_candidates = [
    path for path in INPUT_ROOT.rglob("train.csv")
    if (path.parent / "test.csv").exists()
    and (path.parent / "sample_submission.csv").exists()
]

if not data_candidates:
    raise FileNotFoundError(
        "Dataset kompetisi tidak ditemukan di Kaggle Input."
    )

DATA_ROOT = data_candidates[0].parent

os.environ["DOC_DATA_DIR"] = str(DATA_ROOT)
os.environ["DOC_CHECKPOINT_DIR"] = "/kaggle/working/checkpoint"
os.environ["DOC_SUBMISSION_DIR"] = "/kaggle/working/submissions"
os.environ["DOC_REPORT_DIR"] = "/kaggle/working/reports"

sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

print("Repository:", REPO_ROOT)
print("Dataset:", DATA_ROOT)
```

Jika checkpoint tiga eksperimen sudah tersedia sebagai Kaggle Input, salin ke folder kerja:

```python
from pathlib import Path
import shutil

checkpoint_names = [
    "best_document_efficientnet_b0_baseline.pth",
    "best_document_efficientnet_b0_tuning.pth",
    "best_document_efficientnet_b0_tuning2.pth",
]

source_root = Path("/kaggle/input")
target_dir = Path(
    "/kaggle/working/checkpoint/efficientnet_b0"
)
target_dir.mkdir(parents=True, exist_ok=True)

for filename in checkpoint_names:
    matches = list(source_root.rglob(filename))

    if not matches:
        raise FileNotFoundError(
            f"Checkpoint tidak ditemukan: {filename}"
        )

    shutil.copy2(matches[0], target_dir / filename)
    print("Disalin:", filename)
```

Setelah repository dan dataset ditambahkan, buka `02_modeling.ipynb` untuk menjalankan training atau evaluasi. Gunakan `03_submission.ipynb` untuk membuat file submission.

## Hasil dan Interpretasi

Dari ketiga eksperimen, Hyperparameter Tuning 2 memperoleh validation accuracy dan macro F1 tertinggi. Baseline berada di urutan kedua, sedangkan Hyperparameter Tuning 1 memperoleh skor terendah.

Perbandingan performa didasarkan pada metrik validation dan skor leaderboard jika sudah tersedia. Karena beberapa hyperparameter diubah sekaligus, hasil ini belum dapat menunjukkan pengaruh masing-masing hyperparameter secara terpisah.

Gunakan confusion matrix untuk melihat kelas yang sering tertukar. Precision, recall, dan F1-score per kelas juga dapat digunakan untuk melihat kelas yang sudah dikenali dengan baik maupun yang masih sulit diprediksi.

## Catatan Reproducibility dan Keterbatasan

- Seed eksperimen: 42.
- Jumlah kelas: 16.
- Ukuran input: 224 Ã— 224.
- Validation split: stratified 90:10.
- Dataset gambar TIFF tidak disimpan di GitHub.
- Hasil training dapat berbeda bergantung pada versi PyTorch dan Torchvision, jenis GPU, serta kondisi data.
- Evaluasi menggunakan satu validation split, sehingga hasilnya belum tentu menggambarkan performa pada seluruh test set.
- Resolusi 224 Ã— 224 berpotensi menghilangkan detail teks berukuran kecil.
