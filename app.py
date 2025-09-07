# app.py
# Flask API for BERT text classification (TensorFlow/Keras backend)
# Requires (in your venv):
#   pip install flask flask-cors "transformers==4.44.2" "tensorflow==2.17.0" "tf-keras==2.17.0"

import os
import logging
from typing import Dict, Any

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import tensorflow as tf
from transformers import TFBertForSequenceClassification, BertTokenizerFast

# ---------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------
app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("bert_app")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model", "bert_text_classifier")

if not os.path.isdir(MODEL_DIR):
    raise FileNotFoundError(
        f"Model directory not found at {MODEL_DIR}. "
        "Run your init/training script to create it."
    )

# ---------------------------------------------------------------------
# Load model & tokenizer
# ---------------------------------------------------------------------
logger.info("Loading tokenizer from %s", MODEL_DIR)
tokenizer = BertTokenizerFast.from_pretrained(MODEL_DIR)

logger.info("Loading model from %s", MODEL_DIR)
model = TFBertForSequenceClassification.from_pretrained(MODEL_DIR)

# ---- Force readable labels here (EDIT if your task differs) ----
ID2LABEL: Dict[int, str] = {0: "negative", 1: "positive"}
LABEL2ID: Dict[str, int] = {v: k for k, v in ID2LABEL.items()}

# Inject into config so downstream uses friendly names
model.config.id2label = {int(k): v for k, v in ID2LABEL.items()}
model.config.label2id = {str(k): v for v, k in ID2LABEL.items()}  # keep as strings for safety

logger.info("Label mapping forced to: %s", ID2LABEL)

# ---------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------
def classify_text(text: str, max_length: int = 128) -> Dict[str, Any]:
    enc = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="tf",
    )
    outputs = model(enc, training=False)
    logits = outputs.logits  # [1, num_labels]

    pred_id = int(tf.math.argmax(logits, axis=-1).numpy()[0])
    probs = tf.nn.softmax(logits, axis=-1).numpy()[0]
    confidence = float(probs[pred_id])

    pred_label = ID2LABEL.get(pred_id, f"LABEL_{pred_id}")
    return {
        "classification": pred_label,
        "label_id": pred_id,
        "confidence": round(confidence, 6),
        "num_labels": int(logits.shape[-1]),
    }

# ---------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------
@app.get("/ping")
def ping():
    return jsonify({"status": "ok"}), 200

@app.post("/classify")
def classify():
    if not request.is_json:
        return jsonify({"error": "Request must be application/json"}), 400

    data = request.get_json(silent=True) or {}
    text = data.get("text", "")
    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "Field 'text' is required and must be a non-empty string"}), 400

    try:
        result = classify_text(text)
        return jsonify(result), 200
    except Exception as e:
        logger.exception("Classification error")
        return jsonify({"error": str(e)}), 500

@app.get("/")
def root():
    index_path = os.path.join(BASE_DIR, "index.html")
    if os.path.exists(index_path):
        return send_from_directory(BASE_DIR, "index.html")
    return jsonify({"message": "BERT classifier API. POST JSON to /classify with {'text': '...'}"}), 200

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)

