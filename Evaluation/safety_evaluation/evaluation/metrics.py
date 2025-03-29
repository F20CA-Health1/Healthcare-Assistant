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
    predictions = []
    with open(predictions_path, 'r') as f:
        pred_data = json.load(f)
        for value in pred_data.values():
            predictions.append(value)

    # Read the predictions
    true_labels = []
    with open(true_labels_path, 'r') as f:
        dev_data = json.load(f)
        for category in dev_data.values():
            for item in category:
                true_labels.append(item['answer'])

    # Calculate the index
    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average='macro')
    
    return accuracy, f1 
