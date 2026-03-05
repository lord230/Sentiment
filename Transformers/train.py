import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import csv
import torch
import matplotlib.pyplot as plt
from torch.optim import AdamW
from tqdm import tqdm
import config
from dataset import create_dataloaders
from Models.blocks import Robust_SentimentModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def train_one_epoch(model, dataloader, optimizer, scaler,accumulation_steps=4):
    model.train()

    running_loss = 0.0
    correct_sent = 0
    correct_sarc = 0
    total = 0

    progress = tqdm(dataloader, desc="Training", leave=False)

    optimizer.zero_grad()
    for i, batch in enumerate(progress):
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.amp.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
            output = model(batch)
            loss = output["loss"] / accumulation_steps

            sentiment_logits = output["sentiment_logits"]
            sarcasm_logits = output["sarcasm_logits"]

        scaler.scale(loss).backward()
        
        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(dataloader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        running_loss += loss.item() * accumulation_steps


        sent_preds = torch.argmax(sentiment_logits, dim=-1)
        sent_labels = batch["sentiment_labels"]
        correct_sent += (sent_preds == sent_labels).sum().item()


        sarc_preds = (torch.sigmoid(sarcasm_logits) > 0.5).long()
        sarc_labels = batch["sarcasm_labels"]
        correct_sarc += (sarc_preds == sarc_labels).sum().item()

        total += sent_labels.size(0)

        progress.set_postfix({
            "loss": f"{running_loss / total:.4f}",
            "sent_acc": f"{correct_sent / total:.4f}",
            "sarc_acc": f"{correct_sarc / total:.4f}"
        })

    return (
        running_loss / len(dataloader),
        correct_sent / total,
        correct_sarc / total
    )

def evaluate(model, dataloader):
    model.eval()

    running_loss = 0.0
    correct_sent = 0
    correct_sarc = 0
    total = 0

    progress = tqdm(dataloader, desc="Evaluating", leave=False)

    with torch.no_grad():
        for batch in progress:
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.amp.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
                output = model(batch)

                sentiment_logits = output["sentiment_logits"]
                sarcasm_logits = output["sarcasm_logits"]
                loss = output["loss"]

            running_loss += loss.item()

            # Sentiment
            sent_preds = torch.argmax(sentiment_logits, dim=-1)
            sent_labels = batch["sentiment_labels"]
            correct_sent += (sent_preds == sent_labels).sum().item()

            # Sarcasm
            sarc_preds = (torch.sigmoid(sarcasm_logits) > 0.5).long()
            sarc_labels = batch["sarcasm_labels"]
            correct_sarc += (sarc_preds == sarc_labels).sum().item()

            total += sent_labels.size(0)

            progress.set_postfix({
                "loss": f"{running_loss / total:.4f}",
                "sent_acc": f"{correct_sent / total:.4f}",
                "sarc_acc": f"{correct_sarc / total:.4f}"
            })

    return (
        running_loss / len(dataloader),
        correct_sent / total,
        correct_sarc / total
    )

import os

def main():
    DATASET_PATH = config.CSV_PATH_NEW
    BATCH_SIZE = 4  
    ACCUMULATION_STEPS = 4 
    EPOCHS = 10                
    LR = 1e-5                   
    CHECKPOINT_PATH = "CheckPoints/best_sentiment_model.pt"

    os.makedirs("CheckPoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    train_loader, test_loader = create_dataloaders(
        dataset_path=DATASET_PATH,
        batch_size=BATCH_SIZE,
        max_samples=getattr(config, "MAX_SAMPLES", None)
    )

    # Optimization: if checkpoint exists, we bypass downloading/loading the base weights
    # into RAM, avoiding the "paging file too small" error.
    pretrained = not os.path.exists(CHECKPOINT_PATH)
    model = Robust_SentimentModel(pretrained=pretrained).to(device)

    optimizer = AdamW(model.parameters(), lr=LR)
    scaler = torch.amp.GradScaler("cuda" if torch.cuda.is_available() else "cpu")

    start_epoch = 0
    best_acc = 0.0
    history = {
        "epoch": [],
        "train_loss": [], "train_sent_acc": [], "train_sarc_acc": [],
        "val_loss": [], "val_sent_acc": [], "val_sarc_acc": []
    }

    if os.path.exists(CHECKPOINT_PATH):
        try:
            checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
            if "model_state" in checkpoint:
                model.load_state_dict(checkpoint["model_state"])
                optimizer.load_state_dict(checkpoint["optimizer_state"])
                if "scaler_state" in checkpoint:
                    scaler.load_state_dict(checkpoint["scaler_state"])
                start_epoch = checkpoint["epoch"] + 1
                best_acc = checkpoint.get("best_acc", 0.0)
                history = checkpoint.get("history", history)
                print(f"Loaded checkpoint from {CHECKPOINT_PATH} (resuming from epoch {start_epoch + 1})")
            else:
                model.load_state_dict(checkpoint)
                print(f"Loaded old checkpoint format from {CHECKPOINT_PATH}")
        except Exception as e:
            print(f"Could not load checkpoint: {e}. Training from scratch.")
            print("Re-initializing model with pretrained weights...")
            model = Robust_SentimentModel(pretrained=True).to(device)
            optimizer = AdamW(model.parameters(), lr=LR)
    else:
        print(" No checkpoint found, training from scratch")

    for epoch in range(start_epoch, EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")

        train_loss, train_sent_acc, train_sarc_acc = train_one_epoch(
            model, train_loader, optimizer, scaler, accumulation_steps=ACCUMULATION_STEPS
        )

        val_loss, val_sent_acc, val_sarc_acc = evaluate(
            model, test_loader
        )
        
        torch.cuda.empty_cache()

        print(
            f"Train | Loss: {train_loss:.4f} | "
            f"Sent Acc: {train_sent_acc:.4f} | "
            f"Sarc Acc: {train_sarc_acc:.4f}"
        )
        print(
            f"Val   | Loss: {val_loss:.4f} | "
            f"Sent Acc: {val_sent_acc:.4f} | "
            f"Sarc Acc: {val_sarc_acc:.4f}"
        )

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_loss)
        history["train_sent_acc"].append(train_sent_acc)
        history["train_sarc_acc"].append(train_sarc_acc)
        history["val_loss"].append(val_loss)
        history["val_sent_acc"].append(val_sent_acc)
        history["val_sarc_acc"].append(val_sarc_acc)

        if val_sent_acc > best_acc:
            best_acc = val_sent_acc
            checkpoint = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scaler_state": scaler.state_dict(),
                "best_acc": best_acc,
                "history": history
            }
            torch.save(
                checkpoint,
                CHECKPOINT_PATH
            )
            print("Best model updated and saved to", CHECKPOINT_PATH)
            
        with open("results/training_metrics.csv", mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(history.keys())
            for i in range(len(history["epoch"])):
                writer.writerow([history[k][i] for k in history.keys()])

    print(f"\nTraining finished. Best Val Acc: {best_acc:.4f}")

    # Plotting
    plt.figure(figsize=(12, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(history["epoch"], history["train_loss"], label='Train Loss', marker='o')
    plt.plot(history["epoch"], history["val_loss"], label='Val Loss', marker='x')
    plt.title('Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history["epoch"], history["train_sent_acc"], label='Train Sent Acc', marker='o')
    plt.plot(history["epoch"], history["val_sent_acc"], label='Val Sent Acc', marker='x')
    plt.plot(history["epoch"], history["train_sarc_acc"], label='Train Sarc Acc', linestyle='--', marker='o')
    plt.plot(history["epoch"], history["val_sarc_acc"], label='Val Sarc Acc', linestyle='--', marker='x')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("results/training_performance.png")
    print("Training graphs saved to results/training_performance.png")

if __name__ == "__main__":
    main()
