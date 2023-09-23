import pandas as pd
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AdamW

def predict_label(summary):

    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    label_map = {'0': 'center', 1: 'left',2: 'right'}


    model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=len(label_map))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.load_state_dict(torch.load("RObertaForBiasType.pt"))
    model.eval()

    new_tokens = tokenizer.encode_plus(
        summary,
        add_special_tokens=True,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt"
    )

    new_input_ids = new_tokens["input_ids"].to(device)
    new_attention_mask = new_tokens["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(new_input_ids, attention_mask=new_attention_mask)

    predicted_probabilities = torch.softmax(outputs.logits, dim=1)
    predicted_label_index = torch.argmax(predicted_probabilities, dim=1).item()
    predicted_label_text = label_map[predicted_label_index]

    return predicted_label_text



print(predict_label("My name is pratiyush"))
