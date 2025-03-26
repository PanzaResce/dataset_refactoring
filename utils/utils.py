import requests, transformers, torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from datasets import load_from_disk
from sklearn.metrics import f1_score

from utils.config import *

def extract_model_name(card):
    return card.split("/")[-1]

def get_bnb_config():
    return transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

def load_dataset(path, split="test"):
    ds = load_from_disk(path)
    return ds[split]

def load_model_and_tokenizer(model_card, bnb_config):
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_card,
        device_map="auto",
        quantization_config=bnb_config,
        trust_remote_code=True,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_card)
    return model, tokenizer

def create_pipeline(model, tokenizer):
    return transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        pad_token_id=tokenizer.eos_token_id,
    )

def generate_responses(pipe, ds_test, base_prompt):
    responses = []
    for clause, label in tqdm(zip(ds_test["text"], ds_test["labels"]), total=len(ds_test)):
        messages = [{"role": "system", "content": base_prompt},
                    {"role": "user", "content": clause}]
        gen_out = pipe(messages, max_new_tokens=50, temperature=0, do_sample=False)[0]
        gen_out["labels"] = label
        responses.append(gen_out)
    return responses

def compute_f1_score(responses, label_to_id):
    y_true, y_pred = [], []
    for r in responses:
        resp_tags = [t for t in r["generated_text"][2]["content"].replace("<", "").replace(">", "").replace(",", "").split(" ")]

        true_sample = [0] * len(label_to_id)
        for t in r["labels"]:
            true_sample[t] = 1

        pred_sample = [0] * len(label_to_id)
        if len(resp_tags) <= len(label_to_id):
            for t in resp_tags:
                try:
                    pred_sample[label_to_id[t]] = 1
                except KeyError as e:
                    print(f"WARNING: {e}, initializing sample as list of zeros")

        y_true.append(true_sample)
        y_pred.append(pred_sample)

    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)
    return {"macro": macro_f1, "micro": micro_f1}

def evaluate_models(endpoints, ds_test, ds_val, base_prompt):
    label_to_id = LABEL_TO_ID
    bnb_config = get_bnb_config()
    models_score = {}

    for name, model_card in endpoints.items():
        print(f"------------------------ Prompting {name} ------------------------")

        model, tokenizer = load_model_and_tokenizer(model_card, bnb_config)
        pipe = create_pipeline(model, tokenizer)

        # Evaluate on test dataset
        test_responses = generate_responses(pipe, ds_test, base_prompt)
        test_scores = compute_f1_score(test_responses, label_to_id)

        if ds_val != None:
            # Evaluate on validation dataset
            validation_responses = generate_responses(pipe, ds_val, base_prompt)
            validation_scores = compute_f1_score(validation_responses, label_to_id)
        else:
            validation_scores = {"macro": 0.0, "micro": 0.0}

        models_score[name] = {
            "test": test_scores,
            "validation": validation_scores
        }

    return models_score


def show_confusion_matrix(confusion_matrix, id2label):
    # Convert tensor to numpy array for processing
    confusion_matrix = confusion_matrix.cpu().numpy()

    # Normalize the confusion matrix per class (row-wise normalization)
    row_sums = confusion_matrix.sum(axis=1, keepdims=True)
    normalized_confusion_matrix = confusion_matrix / row_sums

    # Create labels for the axes
    labels = [id2label[i] for i in range(len(id2label))]

    # Plot the normalized confusion matrix with color intensity based on the normalized values
    fig, ax = plt.subplots(figsize=(8, 8))
    cax = ax.matshow(normalized_confusion_matrix, cmap='Blues')

    # Add a color bar
    plt.colorbar(cax)

    # Set axis labels
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))

    # Label axes with the class names
    ax.set_xticklabels(labels, rotation=45, ha="left")
    ax.set_yticklabels(labels)

    # Add both total count and normalized percentage in each cell
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            count = confusion_matrix[i, j]  # Raw count
            percentage = normalized_confusion_matrix[i, j] * 100  # Percentage
            if count > 0:
                ax.text(j, i, f'{int(count)}\n({percentage:.1f}%)', va='center', ha='center', fontsize=10)

    # Set axis labels
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')

    plt.title('Confusion Matrix with Counts and Percentages')
    plt.tight_layout()
    plt.show()