import torch, datasets, pandas
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from safetensors.torch import load_file
from sklearn.metrics import f1_score, classification_report
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

import os
import argparse
import warnings

warnings.filterwarnings('ignore')

def collate_fn(batch):
    texts = [item["text"] for item in batch]
    labels = [torch.tensor(item["labels"], dtype=torch.float32) for item in batch]

    # # Tokenize texts
    # inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

    # Pad labels to the maximum label length in the dataset
    labels_padded = pad_sequence(labels, batch_first=True)

    return {"inputs": texts, "labels": labels_padded}

def to_one_hot(indices, num_classes):
    one_hot = torch.zeros((indices.shape[0], num_classes))
    one_hot.scatter_(1, indices.long(), 1)
    return one_hot.long().numpy()


def evaluate_model(model, tokenizer, data_loader, num_labels, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in data_loader:
            inputs = tokenizer(batch['inputs'], padding=True, truncation=True, return_tensors="pt").to(device)

            outputs = model(**inputs)
            preds = (torch.sigmoid(outputs.logits) > 0.5).int()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(to_one_hot(batch["labels"], num_labels))
    return all_preds, all_labels
    

def main():
    ''' set default hyperparams in default_hyperparams.py '''
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument('--model')
    parser.add_argument('--seed', default=1)
    config = parser.parse_args()

    # BASE_DIR = f'../logs/{config.dataset}'
    BASE_DIR = f'/home/panzaresce/Desktop/unibo/year2/tesi/dataset_refactoring/logs/unfair_tos/'


    if os.path.exists(BASE_DIR):
        print(f'{BASE_DIR} exists!')


    # model_name = "nlpaueb/legal-bert-base-uncased"
    safetensors_path = f"{BASE_DIR}/{config.model}/seed_{config.seed}/model.safetensors"

    # copied from utils.config
    LABEL_TO_ID = {
        "fair": 0,
        "a": 1,
        "ch": 2,
        "cr": 3,
        "j": 4,
        "law": 5,
        "ltd": 6,
        "ter": 7,
        "use": 8,
        "pinc": 9
    }
    ID_TO_LABEL = {v:k for k, v in LABEL_TO_ID.items()}

    num_labels = len(ID_TO_LABEL)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(config.model, num_labels=num_labels)
    model.load_state_dict(load_file(safetensors_path))
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(config.model)

    # Load the test dataset
    full_dataset = datasets.load_from_disk("/home/panzaresce/Desktop/unibo/year2/tesi/dataset_refactoring/142_dataset/tos.hf/")
    test_dataset = full_dataset["test"]

    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

    # Run Evaluation
    preds, labels = evaluate_model(model, tokenizer, test_loader, num_labels, device)

    # Metrics
    micro_f1 = f1_score(labels, preds, average="micro", zero_division=0)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)

    # Display results
    print(f"Micro F1 Score: {micro_f1:.4f}")
    print(f"Macro F1 Score: {macro_f1:.4f}")
    print("Confusion Matrix:")
    print(classification_report(labels, preds, zero_division=0, target_names=ID_TO_LABEL.values()))


    report = classification_report(labels, preds, zero_division=0, target_names=ID_TO_LABEL.values(), output_dict=True)
    df = pandas.DataFrame(report).transpose()
    df.to_csv(f"{BASE_DIR}/{config.model}/report.csv")


if __name__ == '__main__':
    main()
