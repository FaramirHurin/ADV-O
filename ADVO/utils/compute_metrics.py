from sklearn.metrics import PrecisionRecallDisplay, f1_score, accuracy_score, precision_score, recall_score
from typing import List, Dict, Tuple
from .fraud_metrics import pk, pr_auc, precision_recall_curve
import numpy as np
import pandas as pd


def compute_metrics_remove_cards(ytest: pd.Series,
                                 prediction_scores: np.ndarray,
                                 id_cards_test: pd.Series,
                                 id_cards_fraud_train: np.ndarray) -> Tuple[float, float, float, float, float, float, float, float, float, float, float, float]:
    """
    Compute various classification metrics for a model's predictions, including accuracy, precision, recall, f1 score, 
    and partial AUC (pAUC). The pAUCs are computed for different percentiles (pk10, pk20, pk50, pk100, pk300, pk1000) 
    and for the entire population (prauc) and a reference population (cprauc). 

    Parameters:
    - ytest: A pandas Series with the true labels for the test set.
    - prediction_scores: A numpy array with the prediction scores for the test set.
    - id_cards_test: A pandas Series with the card IDs for the test set.
    - id_cards_fraud_train: A numpy array with the card IDs that were labeled as fraud in the training set.

    Returns:
    - A tuple with the values for the following metrics: accuracy, precision, recall, f1 score, pk10, pk20, pk50, pk100, 
      pk300, pk1000, prauc, cprauc.
    """
    ref_prc = 0.0033
    ref_prc_card = 0.0025     
    
#     mask_to_delete = np.isin(id_cards_test.values.astype(int),id_cards_fraud_train.astype(int))
#     y_true = ytest.values[~mask_to_delete]
#     y_pred = prediction_scores[~mask_to_delete]
#     id_cards = id_cards_test.values[~mask_to_delete]
    
    y_pred = prediction_scores 
    y_true = ytest 
    y_pred_int = (y_pred > 0.5).astype(int)
    accuracy = accuracy_score(y_true,y_pred_int)
    precision = precision_score(y_true,y_pred_int)
    recall = recall_score(y_true,y_pred_int)
    f1 = f1_score(y_true,y_pred_int)
    pk10 = pk(y_true,y_pred,10)
    pk20 = pk(y_true,y_pred,20)
    pk50 = pk(y_true,y_pred,50)
    pk100 = pk(y_true,y_pred,100)
    pk300 = None#  pk(y_true,y_pred,300)
    pk1000 = None# pk(y_true,y_pred,1000)
#     cpk100 = pk(y_true,y_pred,100,type="card", cards=id_cards)
#     cpk300 = pk(y_true,y_pred,300,type="card", cards=id_cards)
#     cpk1000 = pk(y_true,y_pred, 1000,type="card", cards=id_cards)
    prauc = pr_auc(y_true,y_pred)
#     prauc_card = pr_auc(y_true,y_pred,type="card", cards=id_cards)
    cprauc = pr_auc(y_true,y_pred,reference_ratio=ref_prc)
#     cprauc_card = pr_auc(y_true,y_pred,type="card", cards=id_cards,reference_ratio=ref_prc_card)
    return accuracy,precision,recall,f1,pk10, pk20,pk50,pk100,pk300,pk1000,prauc,cprauc



def compute_pk(trueY: np.ndarray, predictions: np.ndarray, K: int) -> float:
    """
    Compute the precision for the top K predictions.

    Parameters:
    - trueY: A numpy array with the true labels.
    - predictions: A numpy array with the prediction scores.
    - K: An integer specifying the number of top predictions to consider.

    Returns:
    - The precision for the top K predictions.
    """
    ordered_predicitons = np.sort(predictions)
    value = ordered_predicitons[-K]
    indices = np.where(predictions >= value)
    return sum(trueY[indices])/len(indices[0]) * K
    


def evaluate_models(predictions_proba: List[np.ndarray], discrete_predictions: List[np.ndarray], users: pd.Series, names: List[str], y_test: np.ndarray, K_needed: List[int]) -> Tuple[List[PrecisionRecallDisplay], pd.DataFrame, Dict[str, Dict[str, float]]]:
    """
    This function takes in a list of probability predictions, a list of discrete predictions, a list of names for each model, the true labels for the test set, and a list of K values for the pk metric. It returns a list of PrecisionRecallDisplay objects, a DataFrame with metrics for each model, and a dictionary with pk scores for each model.

    Copy code
    Args:
        predictions_proba: A list of probability predictions for each model
        discrete_predictions: A list of discrete predictions for each model
        names: A list of strings with names for each model
        y_test: The true labels for the test set
        K_needed: A list of K values for the pk metric
        
    Returns:
        A tuple with a list of PrecisionRecallDisplay objects, a DataFrame with metrics for each model, and a dictionary with pk scores for each model.
    """
        
    pk_dictionary = {}
    pr_displays_list = []
    f1_dict = {}
    precison_dict = {}
    recall_dict = {}
    prauc_dict = {}
    prauc_card_dict = {}

    for index, pred in enumerate(predictions_proba):
        
        precision, recall, _ = precision_recall_curve(y_test, pred)
        precision_card, recall_card, _ = precision_recall_curve(y_test, pred, type="card", cards=users)


        pr_displays_list.append(PrecisionRecallDisplay(precision=precision, recall=recall, estimator_name=names[index]))
        auc = -np.trapz(precision, recall)
        auc_card = -np.trapz(precision_card, recall_card)

        y_hat = discrete_predictions[index]
        prauc_dict[names[index]] = auc
        prauc_card_dict[names[index]] = auc_card
        recall_dict[names[index]] = recall_score(y_test, y_hat)
        precison_dict[names[index]] = precision_score(y_test, y_hat)
        f1_dict[names[index]] = f1_score(y_test, y_hat)
        pk_dictionary[names[index]] = {f"PK{k}": pk(np.array(y_test), pred, k) for k in K_needed}      
    
    metrics_dict = {}
    metrics_dict['PRAUC'] = prauc_dict
    metrics_dict['PRAUC_Card'] = prauc_card_dict
    metrics_dict['Precision'] = precison_dict
    metrics_dict['Recall'] = recall_dict
    metrics_dict['F1 score'] = f1_dict

    metrics_df = pd.DataFrame(metrics_dict).T
    pk_df = pd.DataFrame(pk_dictionary)
    all_metrics = pd.concat([metrics_df, pk_df])

    return pr_displays_list, all_metrics