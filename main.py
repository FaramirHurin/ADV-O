import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

from method.advo import ADVO

def main():
    # Create an ADVO instance
    advo = ADVO()

    # Generate transactions data using the ADVO instance
    advo.generate_transactions(n_customers=100, n_terminals=10)

    # Create pairs of fraudulent transactions using the ADVO instance
    advo.create_couples()

    # Tune the hyperparameters of a Ridge regression model using a GridSearchCV instance
    searcher = GridSearchCV(Ridge(), {'alpha': [0.1, 1, 10]}, cv=3)
    advo.tune_regressors(searcher, Ridge)

    # Fit the tuned regression models to the data using the ADVO instance
    advo.fit_regressors(r2_score)

    final_df = advo.enrich_dataframe(advo.transactions_df,2)


if __name__ == '__main__':
    main()