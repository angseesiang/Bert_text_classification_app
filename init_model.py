# init_model.py
# Creates a local TF BERT classifier checkpoint at model/bert_text_classifier/
# - Tries native TF weights first
# - Falls back to converting PyTorch weights (requires torch + safetensors)

import os
from pathlib import Path

from transformers import (
    TFBertForSequenceClassification,
    BertTokenizerFast,
)

SAVE_DIR = Path("model/bert_text_classifier")
MODEL_NAME = "bert-base-uncased"
NUM_LABELS = 2  # change if your task has different classes

def already_exists(p: Path) -> bool:
    return (p / "config.json").exists() and (
        (p / "tf_model.h5").exists() or (p / "saved_model.pb").exists()
    )

def main():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    if already_exists(SAVE_DIR):
        print(f"[init_model] Found existing model at {SAVE_DIR} — nothing to do.")
        return

    print(f"[init_model] Preparing tokenizer: {MODEL_NAME}")
    tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)

    # Try native TF weights first
    try:
        print(f"[init_model] Trying native TF weights for {MODEL_NAME} ...")
        model = TFBertForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=NUM_LABELS,
        )
    except Exception as e_tf:
        print("[init_model] Native TF load failed. Details:")
        print(e_tf)
        print("[init_model] Falling back to PyTorch → TF conversion (requires torch + safetensors) ...")

        # Now force-load from PyTorch weights and convert to TF
        model = TFBertForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=NUM_LABELS,
            from_pt=True,
        )

    print(f"[init_model] Saving model + tokenizer to: {SAVE_DIR}")
    model.save_pretrained(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)

    expected = [
        "config.json",
        "tf_model.h5",  # or saved_model.pb
        "tokenizer_config.json",
        "vocab.txt",
        "special_tokens_map.json",
    ]
    missing = [f for f in expected if not (SAVE_DIR / f).exists()]
    if missing:
        print("[init_model] Note: some files not found after save (may be OK depending on format):", missing)
    else:
        print("[init_model] All expected files present.")
    print("[init_model] Done. You can now run: python app.py")

if __name__ == "__main__":
    main()

