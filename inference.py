import torch
from torch.utils.data import DataLoader
from dataset import InfDataset
from models import Classifier
from transformers import BertTokenizerFast

def tester(model, model_path, data_loader, tokenizer, device):
    print(device)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.to(device)
    model.eval()
    target_names = ['低風險', '中風險', '高風險']
    result = ""
    with torch.no_grad():
        for batch in data_loader:
            logits = model( # Forward pass
                input_ids = batch['input_ids'].to(device), 
                attention_mask = batch['attention_mask'].to(device), 
                d1_score = batch['d1_score'].to(device),
                d2_score = batch['d2_score'].to(device)
            )  

            pred_class = logits.argmax(dim=-1).tolist()
            result = f"你的肌少症狀況： {target_names[pred_class[0]]}"
        print(result)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    while True:
        try:
            bmi = float(input('請輸入病人BMI: '))
        except:
            print("請輸入數字!")
            continue
        try:
            age = float(input('請輸入病人年紀: '))
            break
        except:
            print("請輸入數字!\n現在重新輸入...\n")
            continue
        
    d1_score, d2_score = 0, 0
    if(bmi >= 30) :
        d1_score = 2
    elif(bmi < 30 and bmi >= 25) :
        d1_score = 1
    elif(bmi < 25 and bmi >= 18) :
        bmi = 0
    else :
        bmi = 2

    if(age >= 90) :
        d2_score = 2
    elif (age >= 80 ) :
        d2_score = 1.5
    elif (age >= 65 ) :
        d2_score = 1
    else :
        d2_score = 0

    text = input("請輸入病人狀況描述: ")
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
    dataset = InfDataset(text, d1_score, d2_score, tokenizer)
    dataloader = DataLoader(dataset, batch_size=1)

    model = Classifier()
    model_path = './best_models/gt_models/classifier_best.ckpt'
    tester(model, model_path, dataloader, tokenizer, device)

        