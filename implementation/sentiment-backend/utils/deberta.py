import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

class SimpleDebertaClassifier(nn.Module):
    def __init__(self, model_name="microsoft/deberta-v3-base", num_labels=5):
        super(SimpleDebertaClassifier, self).__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0]  # CLS token
        logits = self.classifier(cls_output)
        return logits


def load_deberta_model(model_path, device, num_labels=5):
    """
    Loads the saved DeBERTa model and tokenizer from disk.
    
    Args:
        model_path (str): Path to the directory containing model.pt and tokenizer files.
        device (torch.device): Device to load the model on.
        num_labels (int): Number of classification labels.

    Returns:
        model: Loaded SimpleDebertaClassifier.
        tokenizer: Loaded tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = SimpleDebertaClassifier(num_labels=num_labels)
    model.load_state_dict(torch.load(f"{model_path}/model.pt", map_location=device))
    model.to(device)
    model.eval()
    return model, tokenizer

