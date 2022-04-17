from transformers.models.auto.modeling_auto import AutoModelForSequenceClassification
from torch import nn
from transformers.modeling_outputs import SequenceClassifierOutput
from torch import Tensor


class BaseModel(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        out: SequenceClassifierOutput = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return self.sigmoid(out.logits).squeeze()
