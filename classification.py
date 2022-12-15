from sklearn.model_selection import train_test_split
from method.advo import ADVO
from generator.generator import Generator

from imblearn.over_sampling import SMOTE, RandomOverSampler, KMeansSMOTE
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import MiniBatchKMeans

from utils.compute_metrics import evaluate_models
from utils.kde import compute_kde_difference_auc
from experiments.ctgan_wrapper import CTGANOverSampler

import sys

SAMPLE_STRATEGY = 0.18
N_JOBS = 12
N_TREES = 20
N_USERS = 10000
N_TERMINALS = 1000

#RANDOM_GRID_RF = {'n_estimators': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000], 'max_features': [1, 'sqrt', 'log2'], 'max_depth': [5, 16, 28, 40, None], 'min_samples_split': [10, 25, 50], 'min_samples_leaf': [4, 8, 32], 'bootstrap': [True, False]}

def fit_predict(X_train,y_train,learner, X_test, predictions_proba, discrete_predictions):
    learner.fit(X_train, y_train)
    y_hat = learner.predict(X_test)
    y_hat_proba = learner.predict_proba(X_test)[:,1]
    predictions_proba.append(y_hat_proba)
    discrete_predictions.append(y_hat)
    
def make_classification():
    # Generate transactions data using the GENERATOR instance
    generator = Generator(n_customers=N_USERS, n_terminals=N_TERMINALS)
    generator.generate()
    generator.export()

    # Train Test Split 
    transactions_df = generator.transactions_df.merge(generator.terminal_profiles_table, left_on='TERMINAL_ID', right_on='TERMINAL_ID', how='left')
    X, y = transactions_df.drop(columns=['TX_FRAUD']), transactions_df['TX_FRAUD']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 
    
    # Initialize useful variables
    sel = ['x_terminal_id', 'y_terminal_id', 'TX_AMOUNT']
    predictions_proba = []
    discrete_predictions = []

    # Specify oversampling strategies to compare 
    Xy_resampled = [KMeansSMOTE(n_jobs=N_JOBS, kmeans_estimator=MiniBatchKMeans(n_init=3),sampling_strategy=SAMPLE_STRATEGY, cluster_balance_threshold=0.1).fit_resample(X_train[sel], y_train),
               SMOTE(k_neighbors=NearestNeighbors(n_jobs=N_JOBS),sampling_strategy=SAMPLE_STRATEGY).fit_resample(X_train[sel], y_train),
               RandomOverSampler(sampling_strategy=SAMPLE_STRATEGY).fit_resample(X_train[sel], y_train),
               CTGANOverSampler(sampling_strategy=SAMPLE_STRATEGY).fit_resample(X_train[sel], y_train),
               ADVO(n_jobs=N_JOBS,sampling_strategy=SAMPLE_STRATEGY).fit_resample(X_train, y_train)]
    # Add not oversampled data as first element
    Xy = [(X_train[sel], y_train)] + Xy_resampled 
    
    # Fit and predict using standard Random Forest for not-oversampled data only 
    names = ['Baseline', 'Baseline_balanced', 'SMOTE', 'Random', 'KMeansSMOTE', 'CTGAN', 'ADVO']
    fit_predict(X_train[sel],y_train, RandomForestClassifier(n_estimators=N_TREES ,n_jobs=N_JOBS) , X_test[sel], predictions_proba, discrete_predictions)
    # Fit and predict using Balanced Random Forest for not-oversampled data AND oversampled data
    for X, y in Xy:
        fit_predict(X,y,BalancedRandomForestClassifier(n_estimators=N_TREES ,n_jobs=N_JOBS) , X_test[sel], predictions_proba, discrete_predictions)

    #trapzs = compute_kde_difference_auc(Xy,sel, names)    

    # Compute metrics
    K_needed = [50, 100, 200, 500, 1000, 2000]
    _, all_metrics = evaluate_models(predictions_proba, discrete_predictions, X_test['CUSTOMER_ID'], names, y_test, K_needed)

    ranking = all_metrics.rank(axis=1, ascending=False)
    print(ranking)

    print(all_metrics)


if __name__ == '__main__':

    sys.path.append('/home/gianmarco/git/ProjectPaperOversampling')

    make_classification()