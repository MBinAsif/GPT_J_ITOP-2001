# retrieve_and_respond.py (Fixed)
import faiss
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Use absolute paths for FAISS index and metadata
faiss_index_path = r"D:\Exp\Chatbot\data\tax_law_faiss.index"
metadata_path = r"D:\Exp\Chatbot\data\metadata.npy"

if not os.path.exists(faiss_index_path):
    raise FileNotFoundError(f"❌ FAISS index not found at {faiss_index_path}")

if not os.path.exists(metadata_path):
    raise FileNotFoundError(f"❌ Metadata file not found at {metadata_path}")

# Load FAISS index and metadata
faiss_index = faiss.read_index(faiss_index_path)
metadata = np.load(metadata_path, allow_pickle=True)

# Load embedding model
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Load GPT-J model with error handling
model_name = r"D:\Exp\Chatbot\models\fine_tuned_gptj"
try:
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Detect available device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load base model with minimal options
    model = AutoModelForCausalLM.from_pretrained(
        "EleutherAI/gpt-j-6B",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True
    )
    
    # Simple resize without mean resizing
    model.resize_token_embeddings(50401)

    # Move to device 
    model = model.to(device)
    
    # Now load the adapter
    from peft import PeftModel
    adapter_model = PeftModel.from_pretrained(model, model_name)
    adapter_model = adapter_model.to(device)
    
    # Replace model with adapter_model
    model = adapter_model
    
    print("✅ Model successfully loaded!")
    
except Exception as e:
    print(f"Detailed error: {str(e)}")
    raise RuntimeError(f"❌ Error loading GPT-J model: {e}")

def retrieve_sections(query, top_k=3):
    query_embedding = embedder.encode([query], convert_to_numpy=True)
    distances, indices = faiss_index.search(query_embedding, top_k)

    results = []
    if indices.size > 0:
        for idx in indices[0]:
            if 0 <= idx < len(metadata):
                results.append(metadata[idx])
    
    return results

async def generate_answer(query):
    retrieved_sections = retrieve_sections(query)

    if not retrieved_sections:
        return "No relevant sections found for the query."

    context = "\n".join([f"📌 Section {sec['section']}: {sec['title']}\n{sec['text']}" for sec in retrieved_sections])

    max_context_length = 1024 - len(tokenizer.encode(f"Query: {query}\n\nAnswer:"))
    context_tokens = tokenizer.encode(context)[:max_context_length]
    context = tokenizer.decode(context_tokens)

    prompt = (
        f"Query: {query}\n\n"
        f"Relevant Sections:\n{context}\n\n"
        "Answer:\n"
        "Please start with a concise summary (crux) of the answer in 2-3 bullet points, "
        "followed by a detailed explanation."
    )
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=1024)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = {key: val.to(device) for key, val in inputs.items() if key in ["input_ids", "attention_mask"]}

    output = model.generate(
        **inputs,
        max_new_tokens=150,  # Reduce from 150 to 100
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )

    response = tokenizer.decode(output[0], skip_special_tokens=True)

    return response

# Example Query (Comment out to avoid running at import time)
# query = "What is the tax rate for salaried individuals?"
# answer = generate_answer(query)
# print(f"💬 AI Response: {answer}")
