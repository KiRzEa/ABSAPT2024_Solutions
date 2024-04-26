from torch import nn
from transformers import AutoModel

class ATEBert(nn.Module):
    def __init__(self, model_name, num_labels=3):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.linear = nn.Linear(backbone.config.hidden_size, num_labels)
    
    def forward(self, input_ids, attention_mask, segment_ids, labels):
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = bert_outputs.last_hidden_state
        
        logits = self.linear(sequence_output)

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            labels = labels.view(-1)
            logits = logits.view(-1, num_labels)

            loss = loss_fn(logits, labels)

            return loss, logits
        
        return logits
        