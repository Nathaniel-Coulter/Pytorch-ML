import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# -------------------------
# 1. Generate synthetic dataset
# -------------------------
def generate_data(n_samples=10000, seq_len=100):
    """
    Each sequence: binary sequence of 0s/1s.
    Label = value of first element in sequence.
    """
    X = np.random.randint(0, 2, size=(n_samples, seq_len))
    y = X[:, 0]  # output depends only on the first token
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

seq_len = 100  # try 100, then 200, 500 to make it harder
X, y = generate_data(n_samples=20000, seq_len=seq_len)
dataset = TensorDataset(X, y)
train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

# -------------------------
# 2. Define models
# -------------------------
class SimpleRNN(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=32, num_classes=2):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = x.unsqueeze(-1)  # add feature dimension
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])  # use final hidden state
        return out

class SimpleLSTM(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=32, num_classes=2):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = x.unsqueeze(-1)
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# -------------------------
# 3. Training loop
# -------------------------
def train_model(model, loader, epochs=5, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    history = []

    for epoch in range(epochs):
        correct, total = 0, 0
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)
        acc = correct / total
        history.append(acc)
        print(f"Epoch {epoch+1}/{epochs}, Accuracy: {acc:.4f}")
    return history

# -------------------------
# 4. Run experiments
# -------------------------
print("Training Vanilla RNN")
rnn_model = SimpleRNN()
rnn_acc = train_model(rnn_model, train_loader, epochs=10)

print("\nTraining LSTM")
lstm_model = SimpleLSTM()
lstm_acc = train_model(lstm_model, train_loader, epochs=10)

# -------------------------
# 5. Plot results
# -------------------------
plt.plot(rnn_acc, label="RNN")
plt.plot(lstm_acc, label="LSTM")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title(f"Vanishing Gradients Demo (Sequence length {seq_len})")
plt.show()
