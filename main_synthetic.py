import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, RandomOverSampler, KMeansSMOTE
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import MiniBatchKMeans
from datetime import timedelta
from ADVO.generator import Generator
from ADVO.oversampler import ADVO, TimeGANOverSampler, CTGANOverSampler
from ADVO.utils import evaluate_models, compute_kde_difference_auc
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

SAMPLE_STRATEGY = 0.1
N_JOBS = 1
N_TREES = 20
N_USERS = 28000
N_TERMINALS = 1000
RANDOM_STATE = 42
COMPROMISSION_PROBABILITY = 0.001
DAYS=100

RANDOM_GRID_RF = {'n_estimators': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000], 'max_features': [1, 'sqrt', 'log2'], 'max_depth': [5, 16, 28, 40, None], 'min_samples_split': [10, 25, 50], 'min_samples_leaf': [4, 8, 32], 'bootstrap': [True, False]}
RANDOM_GRID_RIDGE = {'alpha': [int(x) for x in np.linspace(start = 0.001, stop = 1, num = 100)], 'fit_intercept': [True, False]}
#RANDOM_GRID_NN = {'hidden_layer_sizes': [int(x) for x in np.linspace(start = 1, stop = 41, num = 80)], 'alpha': [int(x) for x in np.linspace(start = 0.005, stop = 0.02, num = 100)]}


CANDIDATE_REGRESSORS = [ Ridge(random_state=RANDOM_STATE), RandomForestRegressor(random_state=RANDOM_STATE)]  #MLPRegressor(max_iter=10000, random_state=RANDOM_STATE, hidden_layer_sizes=10),
CANDIDATE_GRIDS = [RANDOM_GRID_RIDGE, RANDOM_GRID_RF]  #RANDOM_GRID_NN,

def fit_predict(X_train,y_train,learner, X_test, predictions_proba, discrete_predictions):
    learner.fit(X_train, y_train)
    y_hat = learner.predict(X_test)
    y_hat_proba = learner.predict_proba(X_test)[:,1]
    predictions_proba.append(y_hat_proba)
    discrete_predictions.append(y_hat)

def run_advo(X_train, y_train, window_counter):
    advo = ADVO(n_jobs=N_JOBS,sampling_strategy=SAMPLE_STRATEGY,random_state=RANDOM_STATE, mimo=False)
    advo.set_transactions(X_train, y_train)
    advo.create_couples()
    regressor_scores = advo.select_best_regressor(candidate_regressors=CANDIDATE_REGRESSORS,parameters_set=CANDIDATE_GRIDS)
    advo.tune_best_regressors()
    advo.fit_regressors()
    advo.transactions_df = advo.insert_synthetic_frauds(advo.transactions_df)
    regressor_scores.to_csv('results_synthetic/regressor_scores_'+str(window_counter)+'.csv', index=False)
    return advo

def make_classification(train_size_days=10, test_size_days=10, load = True):

    
    if load:
        transactions_df = pd.read_csv('utils/dataset_six_months.csv', parse_dates=['TX_DATETIME'])
    else:
        transactions_df = Generator(n_jobs=1, radius=8).generate(filename='dataset_six_months.csv', nb_days_to_generate=DAYS, max_days_from_compromission=7, n_terminals = N_TERMINALS, n_customers=N_USERS, compromission_probability= COMPROMISSION_PROBABILITY)
        

    start_date, end_date = transactions_df['TX_DATETIME'].min(), transactions_df['TX_DATETIME'].max()
    
    window_start, window_end, window_counter  = start_date, start_date + timedelta(days=train_size_days), 0
    while window_end <= end_date:
        print('Window: ', window_counter, ' - ', window_start, ' - ', window_end)

        # Split data into train and test according to the window
        train_mask, test_mask = (transactions_df['TX_DATETIME'] >= window_start) & (transactions_df['TX_DATETIME'] < window_end), (transactions_df['TX_DATETIME'] >= window_end) & (transactions_df['TX_DATETIME'] < window_end + timedelta(days=test_size_days))
        X_train, y_train, X_test, y_test = transactions_df[train_mask].drop(columns=['TX_FRAUD']), transactions_df[train_mask]['TX_FRAUD'], transactions_df[test_mask].drop(columns=['TX_FRAUD']), transactions_df[test_mask]['TX_FRAUD']
        training_variables, predictions_proba, discrete_predictions = ['X_TERMINAL', 'Y_TERMINAL', 'TX_AMOUNT'], [], []

        # Oversample data using ADVO, SMOTE, RandomOverSampler and KMeansSMOTE
        advo = run_advo(X_train, y_train, window_counter)
        kmeans_smote = KMeansSMOTE(n_jobs=N_JOBS, kmeans_estimator=MiniBatchKMeans(n_init=3),sampling_strategy=SAMPLE_STRATEGY, cluster_balance_threshold=0.001, random_state=RANDOM_STATE).fit_resample(X_train[training_variables], y_train)
        smote = SMOTE(k_neighbors=NearestNeighbors(n_jobs=N_JOBS),sampling_strategy=SAMPLE_STRATEGY,random_state=RANDOM_STATE).fit_resample(X_train[training_variables], y_train)
        random = RandomOverSampler(sampling_strategy=SAMPLE_STRATEGY, random_state=RANDOM_STATE).fit_resample(X_train[training_variables], y_train)
        timegan = TimeGANOverSampler(sampling_strategy=SAMPLE_STRATEGY, epochs=100, seq_len=4, n_seq=3, hidden_dim=24, gamma=1, noise_dim = 32, dim = 128, batch_size = 32, log_step = 100, learning_rate = 5e-4,random_state=RANDOM_STATE).fit_resample(X_train[training_variables+['CUSTOMER_ID']], y_train)
        ctgan = CTGANOverSampler(sampling_strategy=SAMPLE_STRATEGY,random_state=RANDOM_STATE).fit_resample(X_train[training_variables], y_train)
    
        names = ['Baseline','Baseline_balanced', 'SMOTE','Random', 'KMeansSMOTE', 'CTGAN','TIMEGAN', 'ADVO']
        Xy = [(X_train[training_variables], y_train), kmeans_smote, smote, random, ctgan, timegan, (advo.transactions_df[advo.useful_features], advo.transactions_df['TX_FRAUD'])]

        fit_predict(X_train[training_variables],y_train, RandomForestClassifier(n_estimators=N_TREES ,n_jobs=N_JOBS, random_state=RANDOM_STATE) , X_test[training_variables], predictions_proba, discrete_predictions)
        for X, y in Xy:
            fit_predict(X,y, BalancedRandomForestClassifier(n_estimators=N_TREES ,n_jobs=N_JOBS, random_state=RANDOM_STATE) , X_test[training_variables], predictions_proba, discrete_predictions)

        # Compute metrics
        _, all_metrics = evaluate_models(predictions_proba, discrete_predictions, X_test['CUSTOMER_ID'], names, y_test, K_needed = [50, 100, 200, 500, 1000, 2000])
        all_metrics.to_csv('results_synthetic/all_metrics_'+str(window_counter)+'.csv', index=False)
        #trapzs = compute_kde_difference_auc(Xy, training_variables, names)
        #trapzs.to_csv('results_synthetic/trapz_'+str(window_counter)+'.csv', index=False)
        

        window_start, window_end, window_counter  = window_end, window_end + timedelta(days=train_size_days), window_counter + 1
        print('Window ', window_counter, ' done')

if __name__ == '__main__':
    np.random.seed(RANDOM_STATE)
    make_classification(train_size_days=20, test_size_days=20, load=True)
