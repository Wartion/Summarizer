import torch
from transformers import BertTokenizer, BertForSequenceClassification

def predict_bias(summary):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
    label_map = {0: 'Biased', 1: 'Non-biased', 2: 'No agreement'}

    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(label_map))
    model.load_state_dict(torch.load("model.pt"))
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    predicted_score = predicted_probabilities[0][predicted_label_index].item()

    return predicted_label_text 
