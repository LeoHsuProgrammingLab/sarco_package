from torch.utils.data import Dataset

class InfDataset(Dataset):
    def __init__(self, text, score_bmi, score_age, tokenizer, max_length = 64):
        self.text = text
        self.d1_score = score_bmi
        self.d2_score = score_age
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        text = self.text
        d1_score = self.d1_score 
        d2_score = self.d2_score

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt',
            truncation=True
        )

        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()

        sample = {
            'text': text,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'd1_score': d1_score,
            'd2_score': d2_score,
        }

        return sample