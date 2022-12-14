from fraud_metrics import *
from sklearn.metrics import r2_score, precision_recall_curve, PrecisionRecallDisplay, f1_score, accuracy_score, precision_score, recall_score

def compute_metrics_remove_cards(ytest,prediction_scores,id_cards_test,id_cards_fraud_train):
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



def compute_pk(trueY, predictions, K):
    ordered_predicitons = np.sort(predictions)
    value = ordered_predicitons[-K]
    indices = np.where(predictions >= value)
    return sum(trueY[indices])/len(indices[0]) * K
    

