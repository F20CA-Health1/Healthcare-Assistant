import json
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from typing import Dict, Tuple

def calculate_metrics(true_labels_path: str, predictions_path: str) -> Tuple[float, float]:
    """Calculation evaluation index
    
    Args:
        true_labels_path: The path of the true labels file
        predictions_path: The path of the predictions file
        
    Returns:
        Tuple[float, float]: (accuracy, f1_score)
    """
    # Read the true labels
    true_labels = []
    with open(true_labels_path, 'r') as f:
        data = json.load(f)
        for value in data.values():
            true_labels.append("(" + chr(65 + value) + ")")  # 0->A, 1->B, 2->C, 3->D

    # Read the predictions
    predictions = []
    with open(predictions_path, 'r') as f:
        for line in f:
            item = json.loads(line.strip())
            predictions.append(item['origin_pred'])

    # Calculate the index
    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average='macro')
    
    return accuracy, f1 