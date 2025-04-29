import json
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
import os

# Load training data
with open('data/training_data.json', 'r') as f:
    data = json.load(f)

texts = [item['text'] for item in data]
labels = [item['label'] for item in data]

# Load local model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Encode the texts
embeddings = model.encode(texts)

# Train classifier
classifier = LogisticRegression(max_iter=1000)
classifier.fit(embeddings, labels)

# Ensure models directory exists
os.makedirs('models', exist_ok=True)

# Save classifier
with open('models/classifier.pkl', 'wb') as f:
    pickle.dump(classifier, f)

print("✅ Training complete! Classifier saved to models/classifier.pkl")
