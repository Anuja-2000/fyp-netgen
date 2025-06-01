from fastapi import FastAPI, Request
from pydantic import BaseModel
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

app = FastAPI()


model = joblib.load('svmModel.pkl')
class Item(BaseModel):
    name: str
    price: float
    description: str = ""


@app.get("/")
def read_root():
    return {"message": "Welcome to the API!"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: str = ""):
    return {"item_id": item_id, "query": q}


@app.post("/items/")
def create_item(item: Item):
    return {"item": item}

@app.post("/predict/")
async def predict(input_data: Request):
    print("Received input data:", input_data)

    sample_input_data = pd.DataFrame(await input_data.json(), index=[0])
    print("Sample input data:", sample_input_data)
    # Load the saved label encoders
    loaded_label_encoders = joblib.load('label_encoder.pkl')
    # Apply the loaded label encoders to the new data
    for column, encoder in loaded_label_encoders.items():
        if column in sample_input_data.columns:
            # Use .transform() to encode the new data
            sample_input_data[column] = encoder.fit_transform(sample_input_data[column])
        else:
            print(f"Warning: Column '{column}' not found in new data. Skipping encoding for this column.")

    scaler = joblib.load('scaler.pkl')
    # Standardize numerical features in sample data
    sample_input_scaled = scaler.fit_transform(sample_input_data) # Use the scaler fitted on the training data

    # Generate prediction using the trained SVM model
    predicted_category_encoded = model.predict(sample_input_scaled)

    # Decode the predicted category back to original words
    predicted_category = loaded_label_encoders['Preferred Destination Category'].inverse_transform(predicted_category_encoded)

    return {
        "predicted_category": predicted_category[0]
    }
