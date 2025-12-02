import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig, AutoModel, PreTrainedModel


class DesklibAIDetectionModel(PreTrainedModel):
    config_class = AutoConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = AutoModel.from_config(config)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        last_hidden_state = outputs[0]
        if attention_mask is None:
            pooled_output = last_hidden_state.mean(dim=1)
        else:
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)
            sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
            pooled_output = sum_embeddings / sum_mask
        logits = self.classifier(pooled_output)
        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1), labels.float())
        return {"logits": logits, "loss": loss} if loss is not None else {"logits": logits}


class Detector:
    def __init__(self, model_dir="desklib/ai-text-detector-v1.01", device=None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = DesklibAIDetectionModel.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()

    def _prepare_text(self, title, text):
        if title is None:
            return text
        return title.strip() + " " + text.strip()

    def predict_items(self, items, batch_size=16, threshold=0.65):
        results = []
        texts = [self._prepare_text(i.get("title", ""), i.get("text", "")) for i in items]
        ids = [i["id"] for i in items]

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                enc = self.tokenizer(batch_texts, padding=True, truncation=True, max_length=1024, return_tensors='pt')
                enc = {k: v.to(self.device) for k, v in enc.items()}
                outputs = self.model(**enc)
                logits = outputs["logits"]
                logits = logits.squeeze(-1) if logits.dim() > 1 and logits.size(-1) == 1 else logits
                probs = torch.sigmoid(logits)
                probs = probs.detach().cpu().tolist() if isinstance(probs, torch.Tensor) else list(probs)
                for j, p in enumerate(probs):
                    idx = i + j
                    results.append({"id": ids[idx], "probability": float(p), "label": True if p >= threshold else False})

        return results
