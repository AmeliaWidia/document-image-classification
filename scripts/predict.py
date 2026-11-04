"""Prediksi test set dan pembuatan submission baseline, tuning 1, tuning 2."""
import argparse
from pathlib import Path
import pandas as pd
import torch
from tqdm.auto import tqdm
from torchvision.transforms import functional as TF
from scripts.config import DATA_DIR, CHECKPOINT_DIR, SUBMISSION_DIR, EXPERIMENTS, BATCH_SIZE, NUM_WORKERS
from scripts.dataset import load_metadata, build_datasets_and_loaders
from scripts.model import load_checkpoint_model


def parse_args():
    """Parse argumen CLI prediksi."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=["baseline", "tuning1", "tuning2", "all"], default="all")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--tta", action="store_true")
    return parser.parse_args()


def predict_test(model, loader, device, use_tta=False):
    """Prediksi test; TTA opsional merata-ratakan asli dan rotasi +/-2 derajat."""
    ids, predictions = [], []
    model.eval()
    with torch.no_grad():
        for images, batch_ids in tqdm(loader, desc="Prediksi"):
            images = images.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                logits = model(images)
                if use_tta:
                    plus = torch.stack([TF.rotate(x.cpu(), 2, fill=0) for x in images]).to(device)
                    minus = torch.stack([TF.rotate(x.cpu(), -2, fill=0) for x in images]).to(device)
                    logits = (logits + model(plus) + model(minus)) / 3
            ids.extend([str(x) for x in batch_ids])
            predictions.extend(logits.argmax(1).cpu().tolist())
    return ids, predictions


def create_submission(model, loader, sample_df, output_path: Path, device, use_tta=False):
    """Buat CSV submission dan validasi jumlah baris, ID, dan rentang label."""
    ids, labels = predict_test(model, loader, device, use_tta)
    pred = pd.DataFrame({"id": ids, "label": labels})
    pred["id"] = pred["id"].str.replace(r"\.tiff?$", "", regex=True)
    sample = sample_df[["id"]].copy()
    sample["id"] = sample["id"].astype(str)
    submission = sample.merge(pred, on="id", how="left", validate="one_to_one")
    if submission["label"].isna().any():
        raise ValueError("Ada ID test yang tidak memiliki prediksi.")
    submission["label"] = submission["label"].astype(int)
    assert len(submission) == len(sample)
    assert submission["label"].between(0, 15).all()
    assert submission["id"].tolist() == sample["id"].tolist()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    print("Submission:", output_path, submission.shape)
    return submission


def main():
    """Muat data dan checkpoint lalu buat satu atau tiga submission."""
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_df, test_df, sample_df = load_metadata(args.data_dir)
    loaders = build_datasets_and_loaders(train_df, test_df, args.data_dir, args.batch_size, args.num_workers)
    test_loader = loaders[5]
    names = ["baseline", "tuning1", "tuning2"] if args.experiment == "all" else [args.experiment]
    output_names = {"baseline": "submission_baseline.csv", "tuning1": "submission_tuning.csv", "tuning2": "submission_tuning2.csv"}
    for name in names:
        cfg = EXPERIMENTS[name]
        model = load_checkpoint_model(CHECKPOINT_DIR / cfg["checkpoint"], cfg["dropout"], device)
        create_submission(model, test_loader, sample_df, SUBMISSION_DIR / output_names[name], device, args.tta)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
