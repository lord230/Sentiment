# Created By LORD 
import torch
from transformers import AutoTokenizer
from Models.blocks import Robust_SentimentModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_path, model_name="xlm-roberta-base"):
    # Pass pretrained=False since we are overwriting all weights with our checkpoint anyway.
    # This prevents loading the 1.1GB base model into memory, avoiding paging file errors.
    model = Robust_SentimentModel(model_name=model_name, pretrained=False)
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
    else:
        model.load_state_dict(checkpoint)
        
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def predict_sentiment(model, tokenizer, text):
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
        # labels not required for inference
    }

    with torch.no_grad():
        output = model(batch)

    sentiment_logits = output["sentiment_logits"]
    sarcasm_logits = output["sarcasm_logits"]

    sent_pred = torch.argmax(sentiment_logits, dim=-1).item()
    sarc_pred = (torch.sigmoid(sarcasm_logits) > 0.5).item()

    return {
        "sentiment": ["Negative", "Neutral", "Positive"][sent_pred],
        "sarcasm": "Sarcastic" if sarc_pred else "Not Sarcastic"
    }

def gradient_x_input(model, tokenizer, text):
    model.eval()

    encoding = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    # ---- Get embeddings ----
    embeddings = model.encoder.embeddings.word_embeddings(input_ids)
    embeddings = embeddings.detach()
    embeddings.requires_grad_(True)

    # ---- Forward encoder ----
    encoder_outputs = model.encoder(
        inputs_embeds=embeddings,
        attention_mask=attention_mask
    )

    token_embeddings = encoder_outputs.last_hidden_state
    cls_embedding = token_embeddings[:, 0, :]

    # ---- ABSA projection + pooling ----
    absa_token_features = model.aspect_projection(token_embeddings)
    absa_pooled = absa_token_features.mean(dim=1)

    # ---- CONCAT (same as training) ----
    sentiment_input = torch.cat(
        [cls_embedding, absa_pooled],
        dim=1
    )

    logits = model.sentiment_head(sentiment_input)

    pred_class = torch.argmax(logits, dim=-1)
    score = logits[0, pred_class]

    model.zero_grad()
    score.backward()

    grads = embeddings.grad
    importance = (grads * embeddings).abs().sum(dim=-1).squeeze(0)

    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0))

    results = []
    for tok, val in zip(tokens, importance.tolist()):
        if tok not in ["<s>", "</s>", "<pad>"]:
            results.append((tok, val))

    max_val = max(v for _, v in results) + 1e-9
    results = [(t, v / max_val) for t, v in results]

    return results

def extract_absa(model, tokenizer, text):
    model.eval()

    encoding = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        encoder_outputs = model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        token_embeddings = encoder_outputs.last_hidden_state
        aspect_logits = model.aspect_head(
            model.aspect_projection(token_embeddings)
        )

    aspect_preds = torch.argmax(aspect_logits, dim=-1).squeeze(0)
    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0))

    aspect_map = {}
    for tok, tag in zip(tokens, aspect_preds.tolist()):
        if tok in ["<s>", "</s>", "<pad>"]:
            continue
        if tag in [1, 2]:
            aspect_map[tok] = "POS"
        elif tag in [3, 4]:
            aspect_map[tok] = "NEG"

    return aspect_map

def explain_prediction(model, tokenizer, text, top_k=8):
    preds = predict_sentiment(model, tokenizer, text)
    grad_info = gradient_x_input(model, tokenizer, text)
    aspect_info = extract_absa(model, tokenizer, text)

    explanation = []
    for token, score in grad_info:
        if token in aspect_info:
            explanation.append((token, aspect_info[token], score))

    explanation = sorted(explanation, key=lambda x: x[2], reverse=True)

    print("\nINPUT TEXT:")
    print(text)

    print("\nPREDICTION:")
    print(f"Sentiment : {preds['sentiment']}")
    print(f"Sarcasm   : {preds['sarcasm']}")

    print("\nKEY CONTRIBUTING WORDS:")
    for tok, pol, score in explanation[:top_k]:
        print(f"{tok:<15} | {pol:<3} | importance={score:.4f}")


if __name__ == "__main__":
    model, tokenizer = load_model("CheckPoints/best_sentiment_model.pt")

    text = "Oh, fantastic! That's just what I needed today"
    explain_prediction(model, tokenizer, text)
