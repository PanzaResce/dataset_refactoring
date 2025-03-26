import requests, transformers, torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from datasets import load_from_disk
from sklearn.metrics import f1_score

from utils.config import *

def extract_model_name(card):
    return card.split("/")[-1]

def load_dataset(path, split="test"):
    ds = load_from_disk(path)
    return ds[split]


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