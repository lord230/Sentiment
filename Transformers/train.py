# Created By LORD 

from dataset import create_loaders
from Models import blocks
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import random

CSV_PATH    = "dataset/Domain.csv"
BATCH_SIZE  = 32
MAX_LEN     = 20
EPOCHS      = 5
LR          = 2e-4
SEED        = 42
CHECK_DIR   = "CheckPoints"
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

@torch.no_grad()
def evaluate(model, loader: DataLoader, criterion, device, num_classes: int):
    model.eval()
    total_loss = 0.0
    n_items = 0

    tp = torch.zeros(num_classes, dtype=torch.long, device=device)
    fp = torch.zeros(num_classes, dtype=torch.long, device=device)
    fn = torch.zeros(num_classes, dtype=torch.long, device=device)

    correct = 0

    for X, y in loader:
        X = X.to(device)
        y = y.to(device)

        logits = model(X)
        loss = criterion(logits, y)

        total_loss += float(loss.item()) * y.size(0)
        n_items += y.size(0)

        pred = logits.argmax(dim=1)
        correct += int((pred == y).sum().item())

        for c in range(num_classes):
            pc = (pred == c)
            yc = (y == c)
            tp[c] += (pc & yc).sum()
            fp[c] += (pc & (~yc)).sum()
            fn[c] += ((~pc) & yc).sum()

    acc = correct / max(1, n_items)

    eps = 1e-12
    precision = tp.float() / torch.clamp(tp + fp, min=1)
    recall    = tp.float() / torch.clamp(tp + fn, min=1)
    f1_per_c  = 2 * precision * recall / torch.clamp(precision + recall + eps, min=eps)
    macro_f1  = float(f1_per_c.mean().item())

    avg_loss = total_loss / max(1, n_items)
    return avg_loss, acc, macro_f1

def main():
    set_seed(SEED)

    train_loader, test_loader, stoi, label_map = create_loaders(
        CSV_PATH, batch_size=BATCH_SIZE, max_len=MAX_LEN
    )

    num_classes = len(label_map)
    vocab_size  = len(stoi)

    model = blocks.SimpleTransformerClassifier(
        vocab_size=vocab_size,
        num_classes=num_classes,
        max_len=MAX_LEN,
        embed_dim=128,
        num_heads=4,
        depth=2,
        ff_dim=256,
        dropout=0.1,
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    os.makedirs(CHECK_DIR, exist_ok=True)
    best_f1 = 0.0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        seen = 0
        tp = torch.zeros(num_classes, dtype=torch.long, device=DEVICE)
        fp = torch.zeros(num_classes, dtype=torch.long, device=DEVICE)
        fn = torch.zeros(num_classes, dtype=torch.long, device=DEVICE)

        for X, y in train_loader:
            X = X.to(DEVICE)
            y = y.to(DEVICE)

            logits = model(X)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            batch_size = y.size(0)
            running_loss += float(loss.item()) * batch_size
            seen += batch_size

            pred = logits.argmax(dim=1)
            running_correct += int((pred == y).sum().item())


            for c in range(num_classes):
                pc = (pred == c)
                yc = (y == c)
                tp[c] += (pc & yc).sum()
                fp[c] += (pc & (~yc)).sum()
                fn[c] += ((~pc) & yc).sum()

        train_loss = running_loss / max(1, seen)
        train_acc  = running_correct / max(1, seen)

        eps = 1e-12
        precision = tp.float() / torch.clamp(tp + fp, min=1)
        recall    = tp.float() / torch.clamp(tp + fn, min=1)
        f1_per_c  = 2 * precision * recall / torch.clamp(precision + recall + eps, min=eps)
        train_f1  = float(f1_per_c.mean().item())

        val_loss, val_acc, val_f1 = evaluate(model, test_loader, criterion, DEVICE, num_classes)

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} acc={train_acc:.4f} f1={train_f1:.4f}  ||  "
            f"val_loss={val_loss:.4f} acc={val_acc:.4f} f1={val_f1:.4f}"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "stoi": stoi,
                    "label_map": label_map,
                    "config": {
                        "max_len": MAX_LEN,
                        "vocab_size": vocab_size,
                        "num_classes": num_classes,
                    },
                },
                os.path.join(CHECK_DIR, "best.pt"),
            )
            print(f"Saved best checkpoint (macro F1={best_f1:.4f}) -> {os.path.join(CHECK_DIR, 'best.pt')}")

if __name__ == "__main__":
    main()
