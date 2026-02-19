import argparse
import time
import torch
import torch.nn as nn
from tqdm import tqdm

from src.data import build_dataloaders
from src.models import build_resnet18, freeze_backbone, unfreeze_all
from src.eval import predict_probs, compute_metrics
from src.utils import set_seed, ensure_dir, save_json, device


def train_one_epoch(model, loader, optimizer, criterion, dev):
    model.train()
    total_loss = 0.0
    for x, y in tqdm(loader, desc="train", leave=False):
        x, y = x.to(dev), y.to(dev)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_loss(model, loader, criterion, dev):
    model.eval()
    total_loss = 0.0
    for x, y in tqdm(loader, desc="val", leave=False):
        x, y = x.to(dev), y.to(dev)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/chest_xray")
    parser.add_argument("--out_dir", type=str, default="outputs")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs_head", type=int, default=2, help="train only final layer")
    parser.add_argument("--epochs_finetune", type=int, default=3, help="fine-tune entire network")
    parser.add_argument("--lr_head", type=float, default=1e-3)
    parser.add_argument("--lr_finetune", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    dev = device()
    ensure_dir(args.out_dir)

    train_loader, val_loader, test_loader, idx_to_class = build_dataloaders(
        args.data_dir, img_size=args.img_size, batch_size=args.batch_size
    )
    save_json({"idx_to_class": idx_to_class}, f"{args.out_dir}/classes.json")

    model = build_resnet18(num_classes=2, pretrained=True).to(dev)

    criterion = nn.CrossEntropyLoss()

    best_f1 = -1.0
    best_path = f"{args.out_dir}/best.pt"

    # ---- Stage 1: train head only ----
    freeze_backbone(model)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr_head)

    for epoch in range(1, args.epochs_head + 1):
        t0 = time.time()
        tr_loss = train_one_epoch(model, train_loader, optimizer, criterion, dev)
        va_loss = eval_loss(model, val_loader, criterion, dev)
        probs, y_true = predict_probs(model, val_loader, dev)
        metrics = compute_metrics(probs, y_true, threshold=0.5)

        elapsed = time.time() - t0
        print(f"[Head] epoch {epoch}/{args.epochs_head} "
              f"train_loss={tr_loss:.4f} val_loss={va_loss:.4f} "
              f"f1={metrics['f1']:.4f} auc={metrics['roc_auc']} time={elapsed:.1f}s")

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            torch.save({"model": model.state_dict(), "metrics": metrics}, best_path)

    # ---- Stage 2: fine-tune all ----
    unfreeze_all(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_finetune)

    for epoch in range(1, args.epochs_finetune + 1):
        t0 = time.time()
        tr_loss = train_one_epoch(model, train_loader, optimizer, criterion, dev)
        va_loss = eval_loss(model, val_loader, criterion, dev)
        probs, y_true = predict_probs(model, val_loader, dev)
        metrics = compute_metrics(probs, y_true, threshold=0.5)

        elapsed = time.time() - t0
        print(f"[FT] epoch {epoch}/{args.epochs_finetune} "
              f"train_loss={tr_loss:.4f} val_loss={va_loss:.4f} "
              f"f1={metrics['f1']:.4f} auc={metrics['roc_auc']} time={elapsed:.1f}s")

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            torch.save({"model": model.state_dict(), "metrics": metrics}, best_path)

    # ---- Test evaluation using best checkpoint ----
    ckpt = torch.load(best_path, map_location=dev)
    model.load_state_dict(ckpt["model"])

    probs, y_true = predict_probs(model, test_loader, dev)
    test_metrics = compute_metrics(probs, y_true, threshold=0.5)

    save_json({"val_best": ckpt["metrics"], "test": test_metrics}, f"{args.out_dir}/metrics.json")
    print("Saved:", best_path)
    print("Test metrics:", test_metrics)


if __name__ == "__main__":
    main()
