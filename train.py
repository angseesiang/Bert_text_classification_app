# init_model_sst2.py
# Downloads a BERT already fine-tuned on SST-2, converts to TF if needed,
# and saves it to model/bert_text_classifier so your existing app.py can load it.

import os
from pathlib import Path
from transformers import TFBertForSequenceClassification, AutoTokenizer

SAVE_DIR = Path("model/bert_text_classifier")
REPO_ID = "textattack/bert-base-uncased-SST-2"  # BERT fine-tuned for binary sentiment

def main():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[sst2] Loading fine-tuned model from: {REPO_ID}")
    # from_pt=True allows converting PyTorch weights to TF
    model = TFBertForSequenceClassification.from_pretrained(REPO_ID, from_pt=True)
    tok = AutoTokenizer.from_pretrained(REPO_ID, use_fast=True)

    # Ensure readable labels in config (these repos already set them, but enforce anyway)
    model.config.id2label = {0: "negative", 1: "positive"}
    model.config.label2id = {"negative": 0, "positive": 1}

    print(f"[sst2] Saving to: {SAVE_DIR}")
    model.save_pretrained(SAVE_DIR)
    tok.save_pretrained(SAVE_DIR)
    print("[sst2] Done. Now run: python app.py")

if __name__ == "__main__":
    main()

