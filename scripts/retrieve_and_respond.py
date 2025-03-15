import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load FAISS index and metadata
faiss_index = faiss.read_index("../data/tax_law_faiss.index")
metadata = np.load("../data/metadata.npy", allow_pickle=True)

# Load embedding model
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Load GPT-J model
model_name = "../models/fine_tuned_gptj"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

def retrieve_sections(query, top_k=3):
    query_embedding = embedder.encode([query], convert_to_numpy=True)
    distances, indices = faiss_index.search(query_embedding, top_k)

    results = []
    for idx in indices[0]:
        if idx < len(metadata):
            results.append(metadata[idx])

    return results

def generate_answer(query):
    retrieved_sections = retrieve_sections(query)

    context = "\n".join([f"📌 Section {sec['section']}: {sec['title']}\n{sec['text']}" for sec in retrieved_sections])
    prompt = f"Query: {query}\n\nRelevant Sections:\n{context}\n\nAnswer:"

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=1024)
    inputs = {key: val.to("cuda") for key, val in inputs.items()}

    output = model.generate(**inputs, max_new_tokens=100)
    response = tokenizer.decode(output[0], skip_special_tokens=True)

    return response

# Example Query
query = "What is the tax rate for salaried individuals?"
answer = generate_answer(query)
print(f"💬 AI Response: {answer}")
