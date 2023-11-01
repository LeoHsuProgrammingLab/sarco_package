import torch.nn as nn
import torch
from transformers import (
   	AutoModelForSequenceClassification,
)
    
class Classifier(nn.Module):
    def __init__(self) -> None:
        super(Classifier, self).__init__()
        self.relu = nn.ReLU()
        self.score_backbone = Score_model('ckiplab/albert-base-chinese-ner')
        self.fc = nn.Sequential(
            nn.Linear(7, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )

    def forward(self, input_ids, attention_mask, d1_score, d2_score):
        scores = [d1_score.to(torch.float32), d2_score.to(torch.float32)]
        for target in ['d4', 'd5', 'd6', 'd7', 'd8']:
            model_path = f'./best_models/score_models/{target}_best.ckpt'
            self.score_backbone.load_state_dict(torch.load(model_path, map_location='cpu'))
            self.score_backbone.eval()
            logits = self.score_backbone(input_ids = input_ids, attention_mask = attention_mask)
            scores.append(logits[0].argmax(dim=-1))

        x = torch.stack(scores,  dim = -1)
        # nn.Linear use default input as torch.float32
        x = self.fc(x)
        
        return x # return logits and loss

class Score_model(nn.Module):
    def __init__(self, model_name: str = None) -> None:
        super(Score_model, self).__init__()
        self.relu = nn.ReLU()
        self.backbone = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels = 4,
            ignore_mismatched_sizes = True
        )
        # freeze the backbone
        # for name, param in self.backbone.named_parameters():
        #     if name.startswith('albert.encoder'):
        #         param.requires_grad = False
        #     # print(name, param.requires_grad)
            
        self.softmax = nn.Softmax(dim = 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, input_ids, attention_mask, target_label = None):
        
        x = self.backbone(input_ids = input_ids, attention_mask = attention_mask, labels = target_label)
        # add softmax
        logits = self.softmax(x.logits)
        return logits, x.loss # return logits and loss