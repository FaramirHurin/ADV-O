import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

from method.advo import ADVO

from imblearn.over_sampling import SMOTE, RandomOverSampler, KMeansSMOTE
from imblearn.ensemble import BalancedRandomForestClassifier

from ctgan import CTGANOverSampler

SAMPLE_STRATEGY = 0.18

def make_classification():

    advo = ADVO(n_jobs=6)

    # Generate transactions data using the ADVO instance
    advo.generate_transactions(n_customers=100, n_terminals=10)

    # Create pairs of fraudulent transactions using the ADVO instance
    advo.create_couples()

    # Tune the hyperparameters of a Ridge regression model using a GridSearchCV instance
    searcher = GridSearchCV(Ridge(), {'alpha': [0.1, 1, 10]}, cv=3)
    advo.tune_regressors(searcher, Ridge)

    # Fit the tuned regression models to the data using the ADVO instance
    advo.fit_regressors(r2_score)

    original_df = advo.transactions_df.copy()
    enriched_df = advo.enrich_dataframe(advo.transactions_df,2)


    X_border, y_border= KMeansSMOTE(n_jobs=advo.n_jobs, sampling_strategy=SAMPLE_STRATEGY, cluster_balance_threshold=0.1).fit_resample(train_X, train_Y)
    X_SMOTE, y_SMOTE = SMOTE(n_jobs=advo.n_jobs, sampling_strategy=SAMPLE_STRATEGY).fit_resample(train_X, train_Y)
    X_random, y_random = RandomOverSampler(sampling_strategy=SAMPLE_STRATEGY).fit_resample(train_X, train_Y)
    X_CTGAN, y_CTGAN = CTGANOverSampler(sampling_strategy=SAMPLE_STRATEGY).fit_resample(train_X, train_Y)



    
    learner = BalancedRandomForestClassifier(n_estimators=50, random_state=1, n_jobs=10) 

if __name__ == '__main__':
    make_classification()