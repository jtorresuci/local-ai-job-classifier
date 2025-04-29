# Job Posting Paragraph Classifier

## Setup
```bash
pip install sentence-transformers scikit-learn
```

## Train the Model
```bash
python train_classifier.py
```

## Classify a Paragraph
```bash
python classify_paragraph.py
```

---

## Notes
- Uses `all-MiniLM-L6-v2` from Hugging Face locally.
- You can add more labeled examples to `data/training_data.json` to improve accuracy!
