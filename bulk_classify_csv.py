import pandas as pd
import pickle
import os
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

# Read your CSV
input_csv = 'data_to_classify.csv' 
df = pd.read_csv(input_csv)

# Prepare new columns
overview_list = []
responsibilities_list = []
requirements_list = []
other_list = []

for description in df['description'].fillna(''):
    # Split into paragraphs (simple split on newlines)
    paragraphs = [p.strip() for p in description.split('\n') if p.strip()]
    
    # Buckets for this row
    overview = []
    responsibilities = []
    requirements = []
    other = []

    for paragraph in paragraphs:
        category = classify_paragraph(paragraph)
        if category == 'Overview':
            overview.append(paragraph)
        elif category == 'Responsibilities':
            responsibilities.append(paragraph)
        elif category == 'Requirements':
            requirements.append(paragraph)
        else:
            other.append(paragraph)

    # Combine back into single text blobs
    overview_list.append('\n'.join(overview))
    responsibilities_list.append('\n'.join(responsibilities))
    requirements_list.append('\n'.join(requirements))
    other_list.append('\n'.join(other))

# Add new columns to the DataFrame
df['overview'] = overview_list
df['responsibilities'] = responsibilities_list
df['requirements'] = requirements_list
df['other'] = other_list

# Save to a new CSV
output_csv = 'output/classified_jobs.csv'
df.to_csv(output_csv, index=False)

print(f"✅ Saved results to {output_csv}")
