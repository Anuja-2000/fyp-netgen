import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

class TriHeadDeberta(nn.Module):
    def __init__(self, model_name="microsoft/deberta-v3-base", num_labels=5, num_token_tags=5):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        self.content_head = nn.Linear(hidden_size, hidden_size)
        self.aspect_head = nn.Linear(hidden_size, hidden_size)
        self.modifier_head = nn.Linear(hidden_size, hidden_size)

        self.attention_weights = nn.Parameter(torch.randn(3))

        self.classifier = nn.Linear(hidden_size, num_labels)
        self.token_classifier = nn.Linear(hidden_size, num_token_tags)

        self.loss_sentiment = nn.CrossEntropyLoss()
        self.loss_token = nn.CrossEntropyLoss(ignore_index=-100)
        # aspect_token_id will be set after tokenizer is loaded

    def extract_aspect_embedding(self, x, input_ids):
        aspect_mask = (input_ids == self.aspect_token_id).unsqueeze(-1)
        aspect_embeddings = (x * aspect_mask).sum(dim=1)
        aspect_counts = aspect_mask.sum(dim=1).clamp(min=1)
        return aspect_embeddings / aspect_counts

    def forward(self, input_ids, attention_mask, labels=None, tag_labels=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        x = outputs.last_hidden_state

        h_content = self.content_head(x)
        h_modifier = self.modifier_head(x)
        aspect_repr = self.extract_aspect_embedding(x, input_ids)
        h_aspect = self.aspect_head(x + aspect_repr.unsqueeze(1))

        weights = torch.softmax(self.attention_weights, dim=0)
        h_final = weights[0]*h_content + weights[1]*h_aspect + weights[2]*h_modifier

        cls_output = h_final[:, 0]
        sentiment_logits = self.classifier(cls_output)
        token_logits = self.token_classifier(h_final)

        loss = None
        if labels is not None and tag_labels is not None:
            loss_sentiment = self.loss_sentiment(sentiment_logits, labels)
            loss_token = self.loss_token(token_logits.view(-1, token_logits.size(-1)), tag_labels.view(-1))
            aspect_focus_penalty = (1.0 - weights[1]) ** 2
            loss = loss_sentiment + 1.0 * loss_token + 0.1 * aspect_focus_penalty

        return {
            "loss": loss,
            "logits": sentiment_logits,
            "token_logits": token_logits,
            "attention_weights": weights.detach().cpu().numpy()
        }

def load_sentiment_model(model_dir: str, device):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = TriHeadDeberta()
    model.encoder.resize_token_embeddings(len(tokenizer))
    model.aspect_token_id = tokenizer.convert_tokens_to_ids("[ASPECT]")
    model.load_state_dict(torch.load(f"{model_dir}/model.pt", map_location=device))
    model.to(device)
    model.eval()
    return tokenizer, model
