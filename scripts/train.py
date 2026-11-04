"""CLI training untuk eksperimen baseline, tuning 1, dan tuning 2."""
import argparse
import json
import time
from pathlib import Path
import pandas as pd
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm.auto import tqdm
from scripts.config import (
    CHECKPOINT_DIR, DATA_DIR, EXPERIMENTS, CLASS_NAMES, SEED,
    LABEL_SMOOTHING, GRAD_CLIP_NORM, EARLY_STOPPING_PATIENCE,
    BATCH_SIZE, NUM_WORKERS,
)
from scripts.dataset import set_seed, load_metadata, build_datasets_and_loaders
from scripts.model import build_model


def parse_args():
    """Parse argumen command line training."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=["baseline", "tuning1", "tuning2", "all"], default="all")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    return parser.parse_args()


def run_one_epoch(model, loader, criterion, optimizer, scaler, device, amp, training):
    """Jalankan satu epoch dan hitung rata-rata loss serta accuracy."""
    model.train(training)
    loss_sum, correct, total = 0.0, 0, 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, labels in tqdm(loader, leave=False):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp):
                logits = model(images)
                loss = criterion(logits, labels)
            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                scaler.step(optimizer)
                scaler.update()
            size = labels.size(0)
            loss_sum += loss.detach().item() * size
            correct += (logits.detach().argmax(1) == labels).sum().item()
            total += size
            del images, labels, logits, loss
    return loss_sum / max(total, 1), correct / max(total, 1)


def train_experiment(name, train_loader, valid_loader, device, weights_path, epochs_override=None):
    """Latih satu konfigurasi, gunakan early stopping, dan simpan checkpoint terbaik."""
    cfg = EXPERIMENTS[name]
    model = build_model(cfg["dropout"], pretrained=True, weights_path=weights_path).to(device)
    head_ids = {id(p) for p in model.classifier.parameters()}
    backbone_params = [p for p in model.parameters() if id(p) not in head_ids]
    head_params = list(model.classifier.parameters())
    optimizer = AdamW(
        [{"params": backbone_params, "lr": cfg["backbone_lr"]},
         {"params": head_params, "lr": cfg["head_lr"]}],
        weight_decay=cfg["weight_decay"]
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=1)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    max_epochs = epochs_override or cfg["epochs"]
    history = {"experiment": name, "train_loss": [], "valid_loss": [], "train_acc": [], "valid_acc": []}
    best_acc, best_epoch, wait = -1.0, 0, 0
    checkpoint_path = CHECKPOINT_DIR / cfg["checkpoint"]
    start = time.time()

    for epoch in range(max_epochs):
        tick = time.time()
        tr_loss, tr_acc = run_one_epoch(model, train_loader, criterion, optimizer, scaler, device, amp, True)
        va_loss, va_acc = run_one_epoch(model, valid_loader, criterion, optimizer, scaler, device, amp, False)
        scheduler.step(va_acc)
        for key, value in [("train_loss", tr_loss), ("valid_loss", va_loss), ("train_acc", tr_acc), ("valid_acc", va_acc)]:
            history[key].append(value)
        print(f"{cfg['display_name']} | epoch {epoch+1}/{max_epochs} | train_acc={tr_acc:.4f} | valid_acc={va_acc:.4f} | {(time.time()-tick)/60:.1f} menit")
        if va_acc > best_acc:
            best_acc, best_epoch, wait = va_acc, epoch + 1, 0
            torch.save({
                "model_state_dict": model.state_dict(), "class_names": CLASS_NAMES,
                "image_size": 224, "best_valid_accuracy": best_acc,
                "best_epoch": best_epoch, "experiment": name,
                "dropout": cfg["dropout"], "seed": SEED, "history": history,
            }, checkpoint_path)
            print("Checkpoint terbaik:", checkpoint_path)
        else:
            wait += 1
            if wait >= EARLY_STOPPING_PATIENCE:
                print("Early stopping aktif.")
                break
    (CHECKPOINT_DIR / cfg["history"]).write_text(json.dumps(history, indent=2))
    result = {"experiment": cfg["display_name"], "validation_accuracy": best_acc,
              "best_epoch": best_epoch, "checkpoint": str(checkpoint_path),
              "minutes": round((time.time()-start)/60, 2)}
    del model, optimizer, scheduler, scaler
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def main():
    """Jalankan eksperimen terpilih dan simpan tabel hasil training."""
    args = parse_args()
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        raise RuntimeError("GPU tidak terdeteksi. Aktifkan GPU Kaggle.")
    train_df, test_df, _ = load_metadata(args.data_dir)
    prepared = build_datasets_and_loaders(
        train_df, test_df, args.data_dir, args.batch_size, args.num_workers
    )
    train_loader, valid_loader = prepared[3], prepared[4]
    names = ["baseline", "tuning1", "tuning2"] if args.experiment == "all" else [args.experiment]
    results = []
    for name in names:
        set_seed(SEED)
        results.append(train_experiment(name, train_loader, valid_loader, device, args.weights, args.epochs))
    (CHECKPOINT_DIR / "experiment_results.json").write_text(json.dumps(results, indent=2))
    print(pd.DataFrame(results).to_string(index=False))


if __name__ == "__main__":
    main()
