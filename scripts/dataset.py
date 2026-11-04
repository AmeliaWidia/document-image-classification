"""Fungsi pemuatan metadata, decoder TIFF, preprocessing, dan DataLoader."""
from pathlib import Path
import random
import numpy as np
import pandas as pd
import tifffile
import torch
from PIL import Image, ImageFile, ImageOps
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from scripts.config import (
    CLASS_NAMES, IMAGE_SIZE, SEED, VALID_SIZE, BATCH_SIZE, NUM_WORKERS,
)

ImageFile.LOAD_TRUNCATED_IMAGES = True


def set_seed(seed: int = SEED) -> None:
    """Atur seed Python, NumPy, dan PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def read_tiff_with_fallback(path: str | Path) -> Image.Image:
    """Baca TIFF dengan Pillow dan fallback tifffile; hasil akhir RGB."""
    try:
        with Image.open(path) as image:
            image.seek(0)
            image.load()
            if image.width <= 0 or image.height <= 0:
                raise ValueError("Ukuran gambar kosong.")
            return image.convert("RGB")
    except Exception as pil_error:
        try:
            array = np.asarray(tifffile.imread(path))
            if array.size == 0 or array.ndim < 2:
                raise ValueError(f"TIFF kosong: shape={array.shape}")
            if array.ndim == 3 and array.shape[-1] not in (1, 3, 4):
                array = np.moveaxis(array, 0, -1) if array.shape[0] in (1, 3, 4) else array[0]
            if array.ndim == 2:
                array = np.repeat(array[..., None], 3, axis=-1)
            elif array.ndim == 3 and array.shape[-1] == 1:
                array = np.repeat(array, 3, axis=-1)
            elif array.ndim == 3 and array.shape[-1] == 4:
                array = array[..., :3]
            elif array.ndim != 3 or array.shape[-1] != 3:
                raise ValueError(f"Dimensi TIFF tidak didukung: {array.shape}")
            if array.dtype != np.uint8:
                if np.issubdtype(array.dtype, np.integer):
                    max_value = np.iinfo(array.dtype).max
                    array = (array.astype(np.float32) / max_value * 255).clip(0, 255).astype(np.uint8)
                else:
                    array = np.nan_to_num(array.astype(np.float32))
                    low, high = float(array.min()), float(array.max())
                    array = ((array - low) / (high - low) * 255).clip(0, 255).astype(np.uint8) if high > low else np.zeros_like(array, dtype=np.uint8)
            return Image.fromarray(array).convert("RGB")
        except Exception as tiff_error:
            raise RuntimeError(
                f"TIFF gagal dibaca: {path}; Pillow={pil_error}; tifffile={tiff_error}"
            ) from tiff_error


def validate_tiff_content(path: str | Path) -> tuple[bool, str]:
    """Periksa apakah TIFF menghasilkan piksel valid."""
    try:
        with Image.open(path) as image:
            image.seek(0)
            image.load()
            if image.width > 0 and image.height > 0:
                return True, "Pillow"
    except Exception:
        pass
    try:
        array = np.asarray(tifffile.imread(path))
        return (True, "tifffile") if array.size > 0 and array.ndim >= 2 else (False, f"array invalid {array.shape}")
    except Exception as error:
        return False, str(error)


def build_tiff_index(root: Path) -> dict[str, list[Path]]:
    """Buat indeks nama file TIFF untuk pencarian fallback."""
    result: dict[str, list[Path]] = {}
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".tif", ".tiff"}:
            result.setdefault(path.name, []).append(path)
    return result


def resolve_image_path(filename: str, split: str, root: Path, index: dict, class_name: str | None = None) -> Path:
    """Cari path gambar dari path CSV, struktur folder, atau indeks nama file."""
    raw = Path(str(filename))
    candidates = [root / raw, root / split / raw, root / split / split / raw.name]
    if class_name:
        candidates += [root / split / class_name / raw.name, root / split / split / class_name / raw.name]
    candidates.append(root / split / raw.name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    matches = index.get(raw.name, [])
    class_matches = [p for p in matches if class_name in p.parts] if class_name else []
    if len(class_matches) == 1:
        return class_matches[0]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise RuntimeError(f"Nama file ambigu/duplikat: {filename}")
    raise FileNotFoundError(f"File tidak ditemukan: {filename}")


class ResizeWithPadding:
    """Resize proporsional ke kanvas persegi dengan padding putih."""

    def __init__(self, image_size: int = IMAGE_SIZE, fill: int = 255) -> None:
        """Simpan ukuran kanvas dan warna padding."""
        self.image_size = image_size
        self.fill = fill

    def __call__(self, image: Image.Image) -> Image.Image:
        """Pertahankan rasio aspek gambar dan letakkan di tengah kanvas."""
        image = image.convert("RGB")
        resampling = getattr(Image, "Resampling", Image).LANCZOS
        resized = ImageOps.contain(image, (self.image_size, self.image_size), method=resampling)
        canvas = Image.new("RGB", (self.image_size, self.image_size), (self.fill,) * 3)
        canvas.paste(resized, ((self.image_size - resized.width) // 2, (self.image_size - resized.height) // 2))
        return canvas


def build_transform(train: bool, image_size: int = IMAGE_SIZE):
    """Buat preprocessing validasi/test atau augmentasi ringan untuk train."""
    ops = [ResizeWithPadding(image_size)]
    if train:
        ops += [
            transforms.RandomApply([transforms.RandomRotation(5, fill=(255, 255, 255))], p=0.5),
            transforms.RandomApply([transforms.RandomAffine(degrees=0, translate=(0.02, 0.02), scale=(0.97, 1.03), fill=(255, 255, 255))], p=0.30),
            transforms.ColorJitter(brightness=0.12, contrast=0.15),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.7))], p=0.10),
        ]
    ops += [
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
    return transforms.Compose(ops)


class DocumentDataset(Dataset):
    """Dataset TIFF dokumen yang mengembalikan tensor dan label/ID."""

    def __init__(self, dataframe: pd.DataFrame, paths: list[Path], transform, labeled: bool = True) -> None:
        """Simpan metadata, daftar path, transformasi, dan mode label."""
        self.df = dataframe.reset_index(drop=True).copy()
        self.paths = list(paths)
        self.transform = transform
        self.labeled = labeled
        if len(self.df) != len(self.paths):
            raise ValueError("Jumlah metadata dan path gambar tidak sama.")

    def __len__(self) -> int:
        """Kembalikan jumlah sampel dataset."""
        return len(self.df)

    def __getitem__(self, index: int):
        """Baca gambar dan kembalikan tensor beserta label atau ID."""
        row = self.df.iloc[index]
        image = read_tiff_with_fallback(self.paths[index])
        if self.transform:
            image = self.transform(image)
        return (image, int(row["label"])) if self.labeled else (image, str(row["id"]))


def load_metadata(data_dir: Path):
    """Muat CSV kompetisi dan validasi nama kolom serta mapping label."""
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    sample = pd.read_csv(data_dir / "sample_submission.csv")
    if list(train.columns) != ["id", "filename", "class_name", "label"]:
        raise ValueError("Struktur train.csv tidak sesuai.")
    if list(test.columns) != ["id", "filename"] or list(sample.columns) != ["id", "label"]:
        raise ValueError("Struktur test.csv/sample_submission.csv tidak sesuai.")
    if not train["label"].between(0, 15).all():
        raise ValueError("Label train harus 0–15.")
    mapping_frame = train[["class_name", "label"]].drop_duplicates()
    mapping = dict(zip(mapping_frame["class_name"], mapping_frame["label"]))
    expected = {name: i for i, name in enumerate(CLASS_NAMES)}
    if mapping != expected:
        raise ValueError(f"Mapping kelas tidak cocok: {mapping}")
    return train, test, sample


def build_datasets_and_loaders(train_df, test_df, data_root: Path, batch_size: int = BATCH_SIZE, num_workers: int = NUM_WORKERS):
    """Audit TIFF, buat split stratified, dataset, dan DataLoader."""
    index = build_tiff_index(data_root)
    train_paths = [resolve_image_path(r.filename, "train", data_root, index, r.class_name) for r in train_df.itertuples(index=False)]
    test_paths = [resolve_image_path(r.filename, "test", data_root, index) for r in test_df.itertuples(index=False)]
    flags = [validate_tiff_content(p)[0] for p in train_paths]
    train_df = train_df.loc[flags].reset_index(drop=True)
    train_paths = [p for p, ok in zip(train_paths, flags) if ok]
    test_flags = [validate_tiff_content(p)[0] for p in test_paths]
    if not all(test_flags):
        raise RuntimeError("Ada TIFF test tidak valid; submission dihentikan.")
    train_idx, valid_idx = train_test_split(
        np.arange(len(train_df)), test_size=VALID_SIZE, random_state=SEED,
        stratify=train_df["label"].values
    )
    train_part, valid_part = train_df.iloc[train_idx].reset_index(drop=True), train_df.iloc[valid_idx].reset_index(drop=True)
    train_paths_part = [train_paths[i] for i in train_idx]
    valid_paths_part = [train_paths[i] for i in valid_idx]
    train_ds = DocumentDataset(train_part, train_paths_part, build_transform(True), True)
    valid_ds = DocumentDataset(valid_part, valid_paths_part, build_transform(False), True)
    test_ds = DocumentDataset(test_df.reset_index(drop=True), test_paths, build_transform(False), False)
    kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=torch.cuda.is_available(), persistent_workers=False)
    if num_workers > 0:
        kwargs["prefetch_factor"] = 2
    train_loader = DataLoader(train_ds, shuffle=True, **kwargs)
    valid_loader = DataLoader(valid_ds, shuffle=False, **kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **kwargs)
    return train_ds, valid_ds, test_ds, train_loader, valid_loader, test_loader
