import time
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from modwt import modwt, modwtmra
from sklearn.preprocessing import MinMaxScaler

# Ensure GPU is used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔥 Using device: {device}")

# Fetch stock data (RELIANCE.NS)
stock_data = yf.download("RELIANCE.NS", start="2025-01-01", end="2025-03-11", interval="1d")
close_prices = stock_data['Close'].to_numpy()

# MODWT decomposition
wavelet_level = int(np.floor(np.log2(len(close_prices))))
coeffs = modwt(close_prices, 'haar', wavelet_level)
mra_components = modwtmra(coeffs, 'haar')
print("Hi")
# Data preprocessing
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length - 1):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# Define PyTorch Transformer model
class WaveletTransformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dropout=0.1):
        super(WaveletTransformer, self).__init__()
        
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.pos_encoder, num_layers=num_layers)
        self.decoder = nn.Linear(d_model, 1)
        
    def forward(self, src):
        # src shape: [batch_size, seq_len, input_dim]
        src = src.permute(1, 0, 2)  # [seq_len, batch_size, input_dim]
        
        # Linear embedding
        src = self.embedding(src)  # [seq_len, batch_size, d_model]
        
        # Transformer encoder
        output = self.transformer_encoder(src)  # [seq_len, batch_size, d_model]
        
        # Get the last output for prediction
        output = output[-1, :, :]  # [batch_size, d_model]
        
        # Decode to prediction
        output = self.decoder(output)  # [batch_size, 1]
        
        return output
print("HI")
# Hyperparameters
input_chunk_length = 30
batch_size = 32
d_model = 16
nhead = 8
num_layers = 2
dropout = 0.1
learning_rate = 0.001
epochs = 50

# Prepare data and models
scalers = []
train_datasets = []
test_datasets = []
models = []
optimizers = []

# Track training time
start_time = time.time()

# Process each wavelet component
for i, component in enumerate(mra_components):
    
    print(f"Processing component {i+1}/{len(mra_components)}")
    
    # Scale data
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_data = scaler.fit_transform(component.reshape(-1, 1)).flatten()
    scalers.append(scaler)
    
    # Create sequences
    X, y = create_sequences(scaled_data, input_chunk_length)
    
    # Split into train/test
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train).unsqueeze(2).to(device)  # [batch, seq_len, 1]
    y_train_tensor = torch.FloatTensor(y_train).to(device)
    X_test_tensor = torch.FloatTensor(X_test).unsqueeze(2).to(device)
    y_test_tensor = torch.FloatTensor(y_test).to(device)
    
    # Create datasets and dataloaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    train_datasets.append(train_dataset)
    
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_datasets.append(test_dataset)
    
    # Create model
    model = WaveletTransformer(
        input_dim=1,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dropout=dropout
    ).to(device)
    models.append(model)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer.to(device)
    optimizers.append(optimizer)
    
    # Train model
    model_start = time.time()
    model.train()
    
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X).squeeze()
            loss = nn.MSELoss()(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        if (epoch + 1) % 10 == 0:
            print(f"Component {i+1}, Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(train_loader):.6f}")
    
    model_end = time.time()
    print(f"✅ Training for component {i+1} completed in {model_end - model_start:.2f} seconds.")

# Total training time
end_time = time.time()
print(f"🔥 Total training time: {end_time - start_time:.2f} seconds.")
