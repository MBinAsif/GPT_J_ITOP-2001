import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load structured JSON
with open("../data/income_tax_ordinance_2001.json", "r", encoding="utf-8") as f:
    tax_data = json.load(f)

# Initialize embedding model
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

sections = []
metadata = []
for section in tax_data:
    section_number = section["section"]
    section_title = section["title"]
    section_text = " ".join(section["content"])

    sections.append(section_text)
    metadata.append({"section": section_number, "title": section_title, "text": section_text})

# Convert text to embeddings
embeddings = embedder.encode(sections, convert_to_numpy=True)

# Store in FAISS index
dimension = embeddings.shape[1]
faiss_index = faiss.IndexFlatL2(dimension)
faiss_index.add(embeddings)

# Save index and metadata
faiss.write_index(faiss_index, "../data/tax_law_faiss.index")
np.save("../data/metadata.npy", np.array(metadata, dtype=object))

print("✅ FAISS index and metadata saved!")
