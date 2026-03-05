import torch
from transformers import AutoTokenizer
from Models.blocks import Robust_SentimentModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_path, model_name="xlm-roberta-base"):
    model = Robust_SentimentModel(model_name=model_name)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def predict(model, tokenizer, text):
    encoding = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    batch = {
        "input_ids": encoding["input_ids"].to(device),
        "attention_mask": encoding["attention_mask"].to(device),
        "sentiment_labels": torch.tensor([1]).to(device),
        "sarcasm_labels": torch.tensor([0]).to(device),
        "aspect_labels": torch.full(
            (1, encoding["input_ids"].size(1)), -100
        ).to(device)
    }

    with torch.no_grad():
        output = model(batch)
        logits = output["logits"]

    pred = torch.argmax(logits, dim=-1).item()
    sentiment = ["Negative", "Neutral", "Positive"][pred]

    return sentiment
