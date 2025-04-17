from fastapi import FastAPI, Query
import torch
import numpy as np
import pandas as pd

import sys
import os
# Add the code directory to sys.path
root_dir = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.abspath(root_dir))

from app.model import LSTMPredictor
from app.utils import load_and_process_data, normalize_data, create_sequences, prepare_data

# Define the global directory/file path
DATA_PATH = os.path.abspath(os.path.join(root_dir, 'app/data', 'time_series_19-covid-Confirmed_archived_0325.csv'))
PLOTS_DIR = os.path.abspath(os.path.join(root_dir, 'app/plots'))
MODELS_DIR = os.path.abspath(os.path.join(root_dir, 'app/models'))

# Initialize app
app = FastAPI()

# Load and preprocess data
daily_cases = load_and_process_data(DATA_PATH)
normalized_data, scaler = normalize_data(daily_cases)

# Ensure plot dir exists
os.makedirs(PLOTS_DIR, exist_ok=True)

# ----------------------------- Helper functions -----------------------------

def load_model(model_path: str, input_size: int, hidden_size: int, seq_length: int, n_layers: int, dropout: float):
    model = LSTMPredictor(
        input_size=input_size,
        hidden_size=hidden_size,
        seq_len=seq_length,
        n_layers=n_layers,
        dropout=dropout
    )
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu"), weights_only=True))
    return model.eval()


def run_future_predictions(model, seq_length, n_days):
    X_train, _ = create_sequences(normalized_data, seq_length)
    X_train = torch.tensor(X_train).float()

    #test_seq = X_train[:1] # Use the first historical sequence as initial test seq
    test_seq = X_train[-1:] # Use the most recent historical sequence as initial test seq
    preds = []

    with torch.no_grad():
        for _ in range(n_days):
            y_pred = model(test_seq)
            pred = torch.flatten(y_pred).item()
            preds.append(pred)

            new_seq = test_seq.numpy().flatten()
            new_seq = np.append(new_seq, [pred])
            new_seq = new_seq[1:]
            test_seq = torch.tensor(new_seq).view(1, seq_length, 1).float()

    predicted_cases = scaler.inverse_transform(np.expand_dims(preds, axis=0)).flatten().astype(int)
    predicted_index = pd.date_range(
        start=daily_cases.index[-1] + pd.Timedelta(days=1),
        periods=n_days,
    )
    predicted_series = pd.Series(data=predicted_cases, index=predicted_index)
    
    return predicted_series


# ----------------------------- FastAPI Routes -----------------------------

@app.get("/")
def health_check():
    return {"status": "API running!"}

model = None

@app.get("/predict")
def predict_future(
    n_days: int = Query(..., gt=0, description="Number of future days to predict"),
    seq_length: int = 5,
    hidden_size: int = 512,
    n_layers: int = 2,
    dropout: float = 0.5,
    num_epochs: int = 300
):
    try:
        # Predict with all available data
        model_path = os.path.join(MODELS_DIR, f"finalized_lstm_epochs_{num_epochs}.pt")

        global model
        if model is None:
            model = load_model(model_path, input_size=1, hidden_size=hidden_size, seq_length=seq_length, n_layers=n_layers, dropout=dropout)

        predicted_series = run_future_predictions(model, seq_length, n_days)

        return {
            "message": f"Predicted {n_days} future days.",
            "predicted_series": predicted_series.to_dict(),
        }

    except Exception as e:
        return {"error": str(e)}


