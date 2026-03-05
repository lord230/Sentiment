from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import torch

from model import (
    load_model,
    predict_sentiment,
    gradient_x_input,
    extract_absa
)


app = Flask(__name__, static_folder="frontend", static_url_path="")
CORS(app)

MODEL_PATH = "CheckPoints/best_sentiment_model.pt"

print("Loading model...")
model, tokenizer = load_model(MODEL_PATH)
print("Model loaded successfully!")

@app.route("/")
def home():
    """
    Serves the UI instead of 404
    """
    return send_from_directory(app.static_folder, "index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "Text required"}), 400

    preds = predict_sentiment(model, tokenizer, text)
    importance = gradient_x_input(model, tokenizer, text)
    aspects = extract_absa(model, tokenizer, text)

    explanation = []
    for token, score in importance:
        aspect = aspects.get(token, "NEUTRAL")
        explanation.append([token, aspect, round(score, 4)])

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return jsonify({
        "text": text,
        "sentiment": preds["sentiment"],
        "word_count": len(importance),
        "char_count": len(text),
        "explanation": explanation
    })

if __name__ == "__main__":
    app.run(
        debug=True,
        use_reloader=False, 
        host="0.0.0.0",
        port=5000
    )
