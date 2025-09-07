python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txts
pip install --index-url https://download.pytorch.org/whl/cpu torch
python train.py
python app.py

