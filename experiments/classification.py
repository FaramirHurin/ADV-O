import pandas as pd
import numpy as np

from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from method.advo import ADVO
from generator.generator import Generator
#from ctgan_wrapper import CTGANOverSampler
from imblearn.over_sampling import SMOTE, RandomOverSampler, KMeansSMOTE
from imblearn.ensemble import BalancedRandomForestClassifier

from utils.fraud_metrics import pk


SAMPLE_STRATEGY = 0.18
N_JOBS = 12

def make_classification():
    generator = Generator(n_customers=50000, n_terminals=1000)
    generator.generate()
    generator.export()
    
    transactions_df = generator.transactions_df.merge(generator.terminal_profiles_table, left_on='TERMINAL_ID', right_on='TERMINAL_ID', how='left')
    X, y = transactions_df.drop(columns=['TX_FRAUD']), transactions_df['TX_FRAUD']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 
    
    sel = ['x_terminal_id', 'y_terminal_id', 'TX_AMOUNT']
    X_kmeans, y_kmeans= KMeansSMOTE(n_jobs=N_JOBS, sampling_strategy=SAMPLE_STRATEGY, cluster_balance_threshold=0.1).fit_resample(X_train[sel], y_train)
    X_SMOTE, y_SMOTE = SMOTE(n_jobs=N_JOBS, sampling_strategy=SAMPLE_STRATEGY).fit_resample(X_train[sel], y_train)
    X_random, y_random = RandomOverSampler(sampling_strategy=SAMPLE_STRATEGY).fit_resample(X_train[sel], y_train)
    #X_CTGAN, y_CTGAN = CTGANOverSampler(sampling_strategy=SAMPLE_STRATEGY).fit_resample(X_train[sel], y_train)
    X_ADVO, y_ADVO = ADVO(n_jobs=N_JOBS,sampling_strategy=SAMPLE_STRATEGY).fit_resample(X_train, y_train)

    learner = BalancedRandomForestClassifier(n_estimators=50 ,n_jobs=N_JOBS) 
    
    learner.fit(X_train[sel], y_train)
    y_hat_baseline = learner.predict(X_test[sel])
    y_hat_baseline_proba = learner.predict_proba(X_test[sel])[:,1]

    learner.fit(X_kmeans, y_kmeans)
    y_hat_kmeans = learner.predict(X_test[sel])
    y_hat_kmeans_proba = learner.predict_proba(X_test[sel])[:,1]

    learner.fit(X_SMOTE, y_SMOTE)
    y_hat_SMOTE = learner.predict(X_test[sel])
    y_hat_SMOTE_proba = learner.predict_proba(X_test[sel])[:,1]

    learner.fit(X_random, y_random)
    y_hat_random = learner.predict(X_test[sel])
    y_hat_random_proba = learner.predict_proba(X_test[sel])[:,1]

    #learner.fit(X_CTGAN, y_CTGAN)
    #y_hat_CTGAN = learner.predict(X_test)

    learner.fit(X_ADVO, y_ADVO)
    y_hat_ADVO = learner.predict(X_test[sel])
    y_hat_ADVO_proba = learner.predict_proba(X_test[sel])[:,1]

    predictions_proba = [y_hat_baseline_proba, y_hat_SMOTE_proba, y_hat_random_proba, y_hat_kmeans_proba, y_hat_ADVO_proba]
    discrete_predictions = [y_hat_baseline, y_hat_SMOTE, y_hat_random, y_hat_kmeans, y_hat_ADVO]
    names = ['Baseline', 'SMOTE', 'Random', 'KMeansSMOTE', 'ADVO']
    ######
    # First Cell 
    ######

    disp = []
    index = 0

    pk_dictionary = {}
    K = [ 50, 100, 200, 500, 1000, 2000]

    for pred in predictions_proba:
        
        precision, recall, thresholds = precision_recall_curve(y_test, pred)
        print(str([names[index]]) +': '+ str(-np.trapz(precision, recall)))
        this_pk = {}
        for k in K:
            this_pk['PK' + str(k)] = pk(np.array(y_test), pred, k)
    
        pk_dictionary[names[index]] = this_pk
        
        disp.append(PrecisionRecallDisplay(precision = precision, recall=recall, estimator_name=str([names[index]])))
        pk750 = pk(np.array(y_test), pred, 750)
        print(str([names[index]]) +'. pk750 is  '+ str(pk750))
        index +=1


    ######
    # Second Cell
    ######

        
    index = 0

    f1_dict = {}
    precison_dict = {}
    recall_dict = {}

    pk_local_dict = {}
        
    for pred in predictions_proba:
        print (names[index])
        
        pred2 = discrete_predictions[index]
        recall_dict[names[index]] = recall_score(y_test, pred2)
        precison_dict[names[index]] = precision_score(y_test, pred2)
        f1_dict[names[index]] = f1_score(y_test, pred2)
        print(f1_score(y_test, pred2))
        #print(pr_auc_from_precision_recall_curve(test_Y, pred))
        # plot_f_score_curve(test_Y, pred, title=names[index], plot_save_path=names[index])
        index +=1

    pk_local_dict['Precision'] = precison_dict
    pk_local_dict['Recall'] = recall_dict
    pk_local_dict['F1 score'] = f1_dict


    pk_local_dict = pd.DataFrame(pk_local_dict).T
    
    print(pk_local_dict)





if __name__ == '__main__':

    #sys.path.append('/home/gianmarco/git/ProjectPaperOversampling')

    make_classification()