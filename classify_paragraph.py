import pickle
from sentence_transformers import SentenceTransformer

# Load model
embedder = SentenceTransformer('all-MiniLM-L6-v2')
with open('models/classifier.pkl', 'rb') as f:
    classifier = pickle.load(f)

# Classify function
def classify_paragraph(text):
    embedding = embedder.encode([text])
    prediction = classifier.predict(embedding)
    return prediction[0]

if __name__ == "__main__":
    paragraph = input("Enter a paragraph to classify:\n")
    result = classify_paragraph(paragraph)
    print(f"\n🔎 Predicted Section: {result}")
