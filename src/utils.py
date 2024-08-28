import os
import sys
import pickle

import numpy as np
import pandas as pd
from src.exception import CustomException
from sklearn.metrics import recall_score, precision_score, f1_score, fbeta_score, accuracy_score, roc_auc_score


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    

def evaluate_model(y_true, y_pred):
    """
    Evaluate the model using various metrics.
    
    Args:
    y_true (array-like): True labels.
    y_pred (array-like): Predicted labels.

    Returns:
    dict: Dictionary containing the evaluation metrics.
    """
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    f2 = fbeta_score(y_true, y_pred, beta=2, average='weighted')
    auc_roc = roc_auc_score(y_true, y_pred, average='weighted', multi_class='ovr')

    metrics = {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'f2_score': f2,
        'auc_roc': auc_roc
    }

    return metrics
