import torch
from torch import nn
from transformers import BertModel, BertTokenizer

label_cols = [
    "Wellness & Relaxation",
    "Transportation",
    "Food & Dining",
    "Nature & Activities",
    "Entertainment & Shopping",
    "Accommodation",
    "Crowds & Sustainability",
    "Religious & Historical"
]

class MultiLabelBERT(nn.Module):
    def __init__(self, num_labels=len(label_cols)):
        super(MultiLabelBERT, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.drop = nn.Dropout(0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x = self.drop(outputs.pooler_output)
        return self.out(x)

def load_aspect_model(model_path: str, device):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = MultiLabelBERT()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return tokenizer, model
