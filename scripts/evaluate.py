"""Evaluasi validation: accuracy, classification report, dan confusion matrix."""
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from tqdm.auto import tqdm
from scripts.config import DATA_DIR, CHECKPOINT_DIR, FIGURE_DIR, EXPERIMENTS, CLASS_NAMES, BATCH_SIZE, NUM_WORKERS
from scripts.dataset import load_metadata, build_datasets_and_loaders, set_seed
from scripts.model import load_checkpoint_model


def parse_args():
    """Parse argumen CLI evaluasi."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=["baseline", "tuning1", "tuning2", "all"], default="all")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    return parser.parse_args()


def predict_validation(model, loader, device):
    """Jalankan inferensi validation dan kembalikan target serta prediksi."""
    targets, predictions = [], []
    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validasi"):
            images = images.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                logits = model(images)
            predictions.extend(logits.argmax(1).cpu().tolist())
            targets.extend(labels.tolist())
    return targets, predictions


def main():
    """Evaluasi eksperimen dan simpan confusion matrix serta tabel perbandingan."""
    args = parse_args()
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_df, test_df, _ = load_metadata(args.data_dir)
    loaders = build_datasets_and_loaders(train_df, test_df, args.data_dir, args.batch_size, args.num_workers)
    valid_loader = loaders[4]
    names = ["baseline", "tuning1", "tuning2"] if args.experiment == "all" else [args.experiment]
    rows = []
    for name in names:
        cfg = EXPERIMENTS[name]
        ckpt_path = CHECKPOINT_DIR / cfg["checkpoint"]
        model = load_checkpoint_model(ckpt_path, cfg["dropout"], device)
        y_true, y_pred = predict_validation(model, valid_loader, device)
        report = classification_report(y_true, y_pred, labels=list(range(16)), target_names=CLASS_NAMES, output_dict=True, zero_division=0)
        rows.append({"Eksperimen": cfg["display_name"], "Validation Accuracy": accuracy_score(y_true, y_pred),
                     "Macro F1": report["macro avg"]["f1-score"],
                     "Best Epoch": torch.load(ckpt_path, map_location="cpu", weights_only=False)["best_epoch"]})
        cm = confusion_matrix(y_true, y_pred, labels=list(range(16)))
        fig, ax = plt.subplots(figsize=(13, 11))
        ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(ax=ax, cmap="Blues", xticks_rotation=75, colorbar=False)
        ax.set_title(f"{cfg['display_name']} — accuracy {accuracy_score(y_true, y_pred):.4f}")
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / f"confusion_matrix_{name}.png", dpi=160)
        plt.close(fig)
        pd.DataFrame(report).T.to_csv(FIGURE_DIR / f"classification_report_{name}.csv")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    result = pd.DataFrame(rows)
    print(result.to_string(index=False))
    result.to_csv(FIGURE_DIR / "experiment_comparison.csv", index=False)


if __name__ == "__main__":
    main()
