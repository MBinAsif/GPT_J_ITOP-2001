from fastapi import FastAPI
import sys
import os

# Add parent directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.retrieve_and_respond import generate_answer

app = FastAPI()

@app.get("/query")
def get_answer(query: str):
    return {"response": generate_answer(query)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
