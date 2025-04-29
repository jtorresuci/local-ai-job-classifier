# Job Posting Classifier

A lightweight local tool to automatically split job descriptions into:
- Overview
- Responsibilities
- Requirements
- Other

Built with Hugging Face transformers and Logistic Regression. Runs fully offline after setup.

## 🚀 Quick Start

1. Set up a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Train the classifier:
```bash
python train_classifier.py
```

4. Classify:
- Single paragraph:
```bash
python classify_paragraph.py
```
- Entire CSV (requires `description` column):
```bash
python bulk_classify_csv.py
```

Results saved to `classified_jobs.csv`.

## 📦 Structure

- `data/` - Training examples
- `models/` - Trained classifier
- `*.py` - Scripts to train, classify, and bulk process
