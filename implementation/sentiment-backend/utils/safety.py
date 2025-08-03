import torch
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from torch import nn

# ===================== LSTM Classifier =====================
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(hidden_dim * 2, 64)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        x = self.global_pool(lstm_out.permute(0, 2, 1)).squeeze(-1)
        x = self.dropout(torch.relu(self.fc1(x)))
        return self.sigmoid(self.fc2(x)).squeeze()

# ===================== Load Model & Tokenizer =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("models/safety_model/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

model = LSTMClassifier(vocab_size=10000, embedding_dim=64, hidden_dim=64, output_dim=1).to(device)
model.load_state_dict(torch.load("models/safety_model/pytorch_safety_model.pt", map_location=device))
model.eval()

# ===================== Inference Function =====================
def predict_safety(text):
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=100, padding='post')
    input_tensor = torch.LongTensor(padded).to(device)

    with torch.no_grad():
        prob = model(input_tensor).cpu().item()

    label = "Unsafe" if prob >= 0.5 else "Safe"
    return {"label": label, "confidence": prob}
