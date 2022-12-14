import pandas as pd

from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from method.advo import ADVO
from generator.generator import Generator
#from ctgan_wrapper import CTGANOverSampler
from imblearn.over_sampling import SMOTE, RandomOverSampler, KMeansSMOTE
from imblearn.ensemble import BalancedRandomForestClassifier
import sys


SAMPLE_STRATEGY = 0.18
N_JOBS = 6

def make_classification():
    generator = Generator(n_customers=100, n_terminals=10)
    generator.generate()
    transactions_df = generator.transactions_df.merge(generator.terminal_profiles_table, left_on='TERMINAL_ID', right_on='TERMINAL_ID', how='left')
    X, y = transactions_df.drop(columns=['TX_FRAUD']), transactions_df['TX_FRAUD']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 
    
    sel = ['x_terminal_id', 'y_terminal_id', 'TX_AMOUNT']
    X_border, y_border= KMeansSMOTE(n_jobs=N_JOBS, sampling_strategy=SAMPLE_STRATEGY, cluster_balance_threshold=0.1).fit_resample(X_train[sel], y_train)
    X_SMOTE, y_SMOTE = SMOTE(n_jobs=N_JOBS, sampling_strategy=SAMPLE_STRATEGY).fit_resample(X_train[sel], y_train)
    X_random, y_random = RandomOverSampler(sampling_strategy=SAMPLE_STRATEGY).fit_resample(X_train[sel], y_train)
    #X_CTGAN, y_CTGAN = CTGANOverSampler(sampling_strategy=SAMPLE_STRATEGY).fit_resample(X_train[sel], y_train)
    X_ADVO, y_ADVO = ADVO(n_jobs=N_JOBS,sampling_strategy=SAMPLE_STRATEGY).fit_resample(X_train, y_train)

    learner = BalancedRandomForestClassifier(n_jobs=N_JOBS) 
    
    learner.fit(X_train[sel], y_train)
    y_hat_baseline = learner.predict(X_test[sel])
    y_hat_baseline_proba = learner.predict_proba(X_test[sel])[:,1]

    learner.fit(X_border, y_border)
    y_hat_border = learner.predict(X_test[sel])
    y_hat_border_proba = learner.predict_proba(X_test[sel])[:,1]

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

    print('Baseline: ', f1_score(y_test, y_hat_baseline))
    print('Border: ', f1_score(y_test, y_hat_border))
    print('SMOTE: ', f1_score(y_test, y_hat_SMOTE))
    print('Random: ', f1_score(y_test, y_hat_random))
    #print('CTGAN: ', f1_score(y_test, y_hat_CTGAN))
    print('ADVO: ', f1_score(y_test, y_hat_ADVO))


if __name__ == '__main__':

    #sys.path.append('/home/gianmarco/git/ProjectPaperOversampling')

    make_classification()