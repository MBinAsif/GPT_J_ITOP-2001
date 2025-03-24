# app.py (Updated)
from fastapi import FastAPI
import sys
import os
import logging

# Configure logging
logging.basicConfig(
    filename="api_responses.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Add parent directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from scripts.retrieve_and_respond import generate_answer
except ImportError as e:
    import logging
    logging.error(f"Import error: {e}")
    generate_answer = lambda x: "Error: Could not import function"

app = FastAPI()

@app.get("/query")
async def get_answer(query: str):
    response = await generate_answer(query)
    logging.info(f"Query: {query}\nResponse: {response}")
    return {"response": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
