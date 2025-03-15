from fastapi import FastAPI
from scripts.retrieve_and_respond import generate_answer

app = FastAPI()

@app.get("/query")
def get_answer(query: str):
    return {"response": generate_answer(query)}
