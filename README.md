# Tax Law Chatbot (Pakistan Income Tax Ordinance 2001)

This project fine-tunes GPT-J on the **Income Tax Ordinance 2001 of Pakistan** using **LoRA (Low-Rank Adaptation)** for efficient fine-tuning. It also implements **Retrieval-Augmented Generation (RAG)** with FAISS for metadata-based retrieval and dynamic tax-related query answering.

## 🚀 Features

- **Fine-Tunes GPT-J with LoRA**: Optimized training for resource efficiency.
- **Retrieval-Augmented Generation (RAG)**: Uses FAISS to fetch the most relevant sections before answering.
- **Metadata-Based Search**: Supports retrieval by section number, keywords, and other metadata.
- **Fast API Backend**: Deploys as an API for querying tax-related information dynamically.

## 📁 Project Structure

```
📂 TaxLawChatbot
│── 📂 data
│   ├── tax_law.json          # Processed dataset (converted from PDF)
│   ├── tax_law_faiss.index   # FAISS index for retrieval
│── 📂 models
│   ├── fine_tuned_gptj       # Fine-tuned GPT-J model with LoRA
│── 📂 scripts
│   ├── extract_text.py       # Extracts text from the Income Tax Ordinance PDF
│   ├── preprocess.py         # Tokenization, metadata tagging, and JSON conversion
│   ├── create_faiss_index.py # Builds FAISS index from JSON dataset
│   ├── fine_tune.py          # Fine-tunes GPT-J with LoRA
│   ├── retrieve_and_respond.py # Retrieves sections and generates responses using RAG
│── 📂 api
│   ├── app.py                # FastAPI backend for querying
│── requirements.txt          # Required Python packages
│── README.md                 # Project Documentation
```

## 🛠️ Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/yourusername/TaxLawChatbot.git
cd TaxLawChatbot
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Download and Prepare Data

1. Place the **Income Tax Ordinance 2001** PDF inside the `data/` folder.
2. Run the script to extract and preprocess text:

```bash
python scripts/extract_text.py
python scripts/preprocess.py
```

3. Generate the FAISS index:

```bash
python scripts/create_faiss_index.py
```

### 4️⃣ Fine-Tune GPT-J

```bash
python scripts/fine_tune.py
```

### 5️⃣ Start API Server

```bash
uvicorn api.app:app --reload
```

## 🔍 Usage

Once the API is running, you can query tax-related questions:

```bash
curl -X POST "http://127.0.0.1:8000/query" -H "Content-Type: application/json" -d '{"question": "What is the tax rate for salaried individuals in Pakistan?"}'
```

Example response:

```json
{
    "answer": "As per section 149 of the Income Tax Ordinance 2001, the tax rate for salaried individuals varies based on income slabs. For example, individuals earning up to PKR 600,000 annually are exempt from tax..."
}
```

## 📌 Notes

- Ensure that `tax_law_faiss.index` is present before running the API.
- Use `retrieve_and_respond.py` for debugging retrieval issues.

## 🤝 Contribution

Contributions are welcome! Feel free to submit a PR or open an issue.

## 📜 License

This project is licensed under the MIT License.

---

🚀 **Built with AI & Tax Law Expertise!** 🚀
