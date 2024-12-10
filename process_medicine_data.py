import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle

print("Done-1")
# Load the medicine dataset
df = pd.read_csv('https://raw.githubusercontent.com/YashBaravaliya/MedGuide-AI/main/med.csv')
print("Done-2")

# Combine relevant columns into a single text field
df['combined_text'] = df['Name'] + ' ' + df['Uses'] + ' ' + df['Composition'] + ' ' + df['Side_effects'] + ' ' + df['Manufacturer'] + ' ' + df['img_url']
print("Done-3")

# Load a pre-trained sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Done-4")

# Create embeddings
embeddings = model.encode(df['combined_text'].tolist())
print("Done-5")

# Normalize the vectors
faiss.normalize_L2(embeddings)
print("Done-6")

# Create a FAISS index
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)
print("Done-7")

# Save the FAISS index
faiss.write_index(index, 'medicine_faiss_index.bin')
print("Done-8")

# Save the model and DataFrame
with open('sentence_transformer_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("Done-9")

df.to_pickle('medicine_df.pkl')
print("Done-10")

print("Vector database created and saved successfully.")
print("Done-11")