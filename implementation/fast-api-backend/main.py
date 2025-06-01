from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Item(BaseModel):
    name: str
    price: float
    description: str = None

@app.get("/")
def read_root():
    return {"message": "Welcome to the API!"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "query": q}


@app.post("/items/")
def create_item(item: Item):
    return {"item": item}
