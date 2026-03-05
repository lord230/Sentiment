from flask import Flask, request, jsonify
from flask_cors import CORS
from model_utils import load_model, predict

app = Flask(__name__)
CORS(app)

model, tokenizer = load_model("CheckPoints/best_sentiment_model.pt")

@app.route("/predict", methods=["POST"])
def predict_api():
    data = request.json
    text = data.get("text", "")

    if not text.strip():
        return jsonify({"error": "Empty input"}), 400

    sentiment = predict(model, tokenizer, text)

    return jsonify({
        "text": text,
        "sentiment": sentiment
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
