import torch
from transformers import AutoTokenizer
from Models.blocks import Robust_SentimentModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------------------------
# Load Model
# --------------------------------------------------
def load_model(model_path, model_name="xlm-roberta-base"):
    model = Robust_SentimentModel(model_name=model_name)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


# --------------------------------------------------
# Core Prediction + Explainability
# --------------------------------------------------
def predict(model, tokenizer, text, top_k=10):
    encoding = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    # ------------------------
    # 1️⃣ SENTIMENT PREDICTION
    # ------------------------
    batch = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "sentiment_labels": torch.tensor([1]).to(device),
        "sarcasm_labels": torch.tensor([0]).to(device),
        "aspect_labels": torch.full(
            (1, input_ids.size(1)), -100
        ).to(device)
    }

    with torch.no_grad():
        output = model(batch)
        logits = output["logits"]

    pred = torch.argmax(logits, dim=-1).item()
    sentiment = ["Negative", "Neutral", "Positive"][pred]

    # ------------------------
    # 2️⃣ GRADIENT × INPUT
    # ------------------------
    embeddings = model.encoder.embeddings.word_embeddings(input_ids)
    embeddings = embeddings.detach()
    embeddings.requires_grad_(True)

    encoder_out = model.encoder(
        inputs_embeds=embeddings,
        attention_mask=attention_mask
    )

    cls_embedding = encoder_out.last_hidden_state[:, 0, :]
    sent_logits = model.sentiment_head(cls_embedding)

    score = sent_logits[0, pred]
    model.zero_grad()
    score.backward()

    grads = embeddings.grad
    importance = (grads * embeddings).abs().sum(dim=-1).squeeze(0)

    # ------------------------
    # 3️⃣ ASPECT / POLARITY TAG
    # ------------------------
    with torch.no_grad():
        aspect_logits = model.aspect_head(
            encoder_out.last_hidden_state
        )
        aspect_preds = torch.argmax(aspect_logits, dim=-1).squeeze(0)

    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0))

    explanation = []
    for tok, imp, tag in zip(tokens, importance.tolist(), aspect_preds.tolist()):
        if tok in ["<s>", "</s>", "<pad>"]:
            continue

        if tag in [1, 2]:
            pol = "POS"
        elif tag in [3, 4]:
            pol = "NEG"
        else:
            continue

        explanation.append((tok, pol, imp))

    # Normalize importance
    if explanation:
        max_val = max(v for _, _, v in explanation) + 1e-9
        explanation = [
            (t, p, round(v / max_val, 4)) for t, p, v in explanation
        ]

    explanation = sorted(explanation, key=lambda x: x[2], reverse=True)[:top_k]

    return {
        "sentiment": sentiment,
        "explanation": explanation
    }
