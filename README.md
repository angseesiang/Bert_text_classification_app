# 🧠 BERT Text Classification App

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](#)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.17-orange)](#)
[![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-green)](#)

This repository contains a **Flask-based BERT text classification application** using TensorFlow/Keras and Hugging Face Transformers.  
The model is fine-tuned for **binary sentiment classification** (positive / negative).  

---

## 📖 Contents

- `app.py` – Flask REST API for serving predictions
- `index.html` – Simple frontend for text input and classification
- `init_model.py` – Script to initialize a BERT model locally
- `train.py` – Training script for custom datasets
- `test_app.py` – Endpoint test script
- `requirements.txt` – Python dependencies
- `url.txt` – Repository link

---

## 🚀 How to Use

### 1) Clone this repository

```bash
git clone https://github.com/angseesiang/Bert_text_classification_app.git
cd Bert_text_classification_app
```

### 2) Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate   # On Linux / macOS
venv\Scripts\activate      # On Windows
```

### 3) Install dependencies

```bash
pip install -r requirements.txt
pip install --index-url https://download.pytorch.org/whl/cpu torch
```

### 4) Initialize / Train the model

Run one of the following:

```bash
python init_model.py        # Initialize base BERT model
python train.py             # Train on your dataset
```

### 5) Start the Flask application

```bash
python app.py
```

### 6) Access the application

Open your browser and go to:  

👉 [http://127.0.0.1:5000](http://127.0.0.1:5000)  

From the interface, you can enter sample text such as:  

- **`I love you`** → expected classification: *positive*  
- **`I hate you`** → expected classification: *negative*  

### 7) Run tests

```bash
python test_app.py
```

If everything is configured correctly, the script should output:

```
Test passed!
```

---

## 🛠️ Requirements

- Python 3.9+
- TensorFlow 2.17
- Hugging Face Transformers
- Flask & Flask-CORS
- Torch (for conversion if needed)

See `requirements.txt` for the full list.

---

## 📌 Notes

- The app provides sentiment classification (`positive` / `negative`) via REST API and a simple web UI.  
- Model files are saved under `model/bert_text_classifier`.  
- You can adapt this framework to other text classification tasks by changing the dataset and labels.

---

## 📜 License

This project is for **educational purposes only**.
