import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from method.advo import ADVO
from generator import Generator

from imblearn.over_sampling import SMOTE, RandomOverSampler, KMeansSMOTE
from imblearn.ensemble import BalancedRandomForestClassifier


from ctgan import CTGANOverSampler

SAMPLE_STRATEGY = 0.18
N_JOBS = 6

def make_classification():
    generator = Generator()
    generator.generate_transactions(n_customers=100, n_terminals=10)
    X, y = generator.transactions_df.drop(columns=['TX_FRAUD']), generator.transactions_df['TX_FRAUD']
    X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 

    X_border, y_border= KMeansSMOTE(n_jobs=N_JOBS, sampling_strategy=SAMPLE_STRATEGY, cluster_balance_threshold=0.1).fit_resample(X_train, y_train)
    X_SMOTE, y_SMOTE = SMOTE(n_jobs=N_JOBS, sampling_strategy=SAMPLE_STRATEGY).fit_resample(X_train, y_train)
    X_random, y_random = RandomOverSampler(sampling_strategy=SAMPLE_STRATEGY).fit_resample(X_train, y_train)
    X_CTGAN, y_CTGAN = CTGANOverSampler(sampling_strategy=SAMPLE_STRATEGY).fit_resample(X_train, y_train)
    X_ADVO, y_ADVO = ADVO(n_jobs=N_JOBS,sampling_strategy=SAMPLE_STRATEGY).fit_resample(X_train, y_train)

    learner = BalancedRandomForestClassifier(n_jobs=N_JOBS) 
    
    learner.fit(X_train, y_train)
    y_hat_baseline = learner.predict(X_test)

    learner.fit(X_border, y_border)
    y_hat_border = learner.predict(X_test)

    learner.fit(X_SMOTE, y_SMOTE)
    y_hat_SMOTE = learner.predict(X_test)

    learner.fit(X_random, y_random)
    y_hat_random = learner.predict(X_test)

    learner.fit(X_CTGAN, y_CTGAN)
    y_hat_CTGAN = learner.predict(X_test)

    learner.fit(X_ADVO, y_ADVO)
    y_hat_ADVO = learner.predict(X_test)

    print('Baseline: ', r2_score(y_test, y_hat_baseline))
    print('Border: ', r2_score(y_test, y_hat_border))
    print('SMOTE: ', r2_score(y_test, y_hat_SMOTE))
    print('Random: ', r2_score(y_test, y_hat_random))
    print('CTGAN: ', r2_score(y_test, y_hat_CTGAN))
    print('ADVO: ', r2_score(y_test, y_hat_ADVO))


if __name__ == '__main__':
    make_classification()