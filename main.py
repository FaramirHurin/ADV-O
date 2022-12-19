from sklearn.model_selection import train_test_split
# from advo.advo import ADVO
# from advo.generator.generator import Generator

from imblearn.over_sampling import SMOTE, RandomOverSampler, KMeansSMOTE
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import MiniBatchKMeans


from advo.generator import Generator
from advo.advo import ADVO
from advo.utils import evaluate_models, compute_kde_difference_auc, fraud_metrics

# from advo.utils.compute_metrics import evaluate_models
# from advo.utils.kde import compute_kde_difference_auc
#from experiments.ctgan_wrapper import CTGANOverSampler
#import torch

import numpy as np

import sys

SAMPLE_STRATEGY = 0.18
N_JOBS = 35
N_TREES = 20
N_USERS = 10000
N_TERMINALS = 1000
RANDOM_STATE = 42

RANDOM_GRID_RF = {'n_estimators': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000], 'max_features': [1, 'sqrt', 'log2'], 'max_depth': [5, 16, 28, 40, None], 'min_samples_split': [10, 25, 50], 'min_samples_leaf': [4, 8, 32], 'bootstrap': [True, False]}
RANDOM_GRID_RIDGE = {'alpha': [int(x) for x in np.linspace(start = 0.001, stop = 1, num = 100)], 'fit_intercept': [True, False]}
RANDOM_GRID_NN = {'hidden_layer_sizes': [int(x) for x in np.linspace(start = 1, stop = 41, num = 80)], 'alpha': [int(x) for x in np.linspace(start = 0.005, stop = 0.02, num = 100)]}


CANDIDATE_REGRESSORS = [MLPRegressor(max_iter=2000, random_state=RANDOM_STATE), Ridge(random_state=RANDOM_STATE), RandomForestRegressor(random_state=RANDOM_STATE)]
CANDIDATE_GRIDS = [RANDOM_GRID_NN, RANDOM_GRID_RIDGE, RANDOM_GRID_RF]

def fit_predict(X_train,y_train,learner, X_test, predictions_proba, discrete_predictions):
    learner.fit(X_train, y_train)
    y_hat = learner.predict(X_test)
    y_hat_proba = learner.predict_proba(X_test)[:,1]
    predictions_proba.append(y_hat_proba)
    discrete_predictions.append(y_hat)
    
def make_classification():

    np.random.seed(RANDOM_STATE)
    #torch.manual_seed(RANDOM_STATE)


    # Generate transactions data using the GENERATOR instance
    gen = Generator(n_customers=N_USERS, n_terminals=N_TERMINALS)
    #generator.generate()
    #generator.export()

    gen.load()


    # Train Test Split 
    transactions_df = gen.transactions_df.merge(gen.terminal_profiles_table, left_on='TERMINAL_ID', right_on='TERMINAL_ID', how='left')
    X, y = transactions_df.drop(columns=['TX_FRAUD']), transactions_df['TX_FRAUD']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE) 
    
    # Initialize useful variables
    sel = ['x_terminal_id', 'y_terminal_id', 'TX_AMOUNT']
    predictions_proba = []
    discrete_predictions = []

    advo = ADVO(n_jobs=N_JOBS,sampling_strategy=SAMPLE_STRATEGY,random_state=RANDOM_STATE)
    advo.set_transactions(X_train, y_train)
    advo.create_couples()
    advo.select_best_regressor(candidate_regressors=CANDIDATE_REGRESSORS,parameters_set=CANDIDATE_GRIDS)
    advo.tune_best_regressors()
    advo.fit_regressors()
    advo.transactions_df = advo.enrich_dataframe(advo.transactions_df)
    advo_tuple = advo.transactions_df[advo.useful_features], advo.transactions_df['TX_FRAUD']
    
    kmeans_smote = KMeansSMOTE(n_jobs=N_JOBS, kmeans_estimator=MiniBatchKMeans(n_init=3),sampling_strategy=SAMPLE_STRATEGY, cluster_balance_threshold=0.1, random_state=RANDOM_STATE).fit_resample(X_train[sel], y_train)
    smote = SMOTE(k_neighbors=NearestNeighbors(n_jobs=N_JOBS),sampling_strategy=SAMPLE_STRATEGY,random_state=RANDOM_STATE).fit_resample(X_train[sel], y_train)
    random = RandomOverSampler(sampling_strategy=SAMPLE_STRATEGY, random_state=RANDOM_STATE).fit_resample(X_train[sel], y_train)
    #ctgan = CTGANOverSampler(sampling_strategy=SAMPLE_STRATEGY).fit_resample(X_train[sel], y_train)

    # Specify oversampling strategies to compare 
    Xy_resampled = [kmeans_smote, 
                    smote, 
                    random, 
                    #ctgan, 
                    advo_tuple]

    # Add not oversampled data as first element
    Xy = [(X_train[sel], y_train)] + Xy_resampled 
    
    # Fit and predict using standard Random Forest for not-oversampled data only 
    names = ['Baseline', 
            'Baseline_balanced', 
            'SMOTE', 
            'Random', 
            'KMeansSMOTE', 
            #'CTGAN', 
            'ADVO']
            
    fit_predict(X_train[sel],y_train, RandomForestClassifier(n_estimators=N_TREES ,n_jobs=N_JOBS, random_state=RANDOM_STATE) , X_test[sel], predictions_proba, discrete_predictions)
    # Fit and predict using Balanced Random Forest for not-oversampled data AND oversampled data
    for X, y in Xy:
        fit_predict(X,y,BalancedRandomForestClassifier(n_estimators=N_TREES ,n_jobs=N_JOBS, random_state=RANDOM_STATE) , X_test[sel], predictions_proba, discrete_predictions)

    trapzs = compute_kde_difference_auc(Xy,sel, names)
    
    #save trapz in a file
    trapzs.to_csv('trapz.csv', index=False)

    # Compute metrics
    K_needed = [50, 100, 200, 500, 1000, 2000]
    _, all_metrics = evaluate_models(predictions_proba, discrete_predictions, X_test['CUSTOMER_ID'], names, y_test, K_needed)

    ranking = all_metrics.rank(axis=1, ascending=False)

    #save all_metrics in a file
    all_metrics.to_csv('all_metrics.csv', index=False)
    #save ranking in a file
    ranking.to_csv('ranking.csv', index=False)

    print(trapzs)
    print(ranking)
    print(all_metrics)


if __name__ == '__main__':
    make_classification()