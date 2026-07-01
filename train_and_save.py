import torch
import torch.nn as nn
import joblib
import os
from src.data_processing import load_and_preprocess_data
from src.models import FeedForwardANN, ANFISThenANN

def train_model(model, train_loader, device, epochs=150, lr=0.005):
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for epoch in range(epochs):
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)
            loss = criterion(preds, yb)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model

if __name__ == "__main__":
    print("1. Loading dataset and initializing pipeline...")
    try:
        train_loader, _, _, device, scaler = load_and_preprocess_data("diabetes.csv")
    except FileNotFoundError:
        print("ERROR: diabetes.csv not found. Please place it in this directory.")
        exit(1)

    print("2. Exporting MinMaxScaler to scaler.pkl...")
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/scaler.pkl")

    print("3. Training Feedforward ANN...")
    fnn = FeedForwardANN(input_size=8).to(device)
    fnn = train_model(fnn, train_loader, device, lr=0.005)
    torch.save(fnn.state_dict(), "models/fnn_model.pth")

    print("4. Training Hybrid ANN-ANFIS...")
    anfis = ANFISThenANN(in_features=8, n_mfs=6, hidden_size=32).to(device)
    anfis = train_model(anfis, train_loader, device, lr=0.003)
    torch.save(anfis.state_dict(), "models/anfis_model.pth")

    print("\nSUCCESS! Pipeline models compiled successfully.")
    print("You can now run: streamlit run app.py")
