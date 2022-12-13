from generator import Generator
import pandas as pd
from pandarallel import pandarallel
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV



class ADVO():
    
    """ ADVO: An Adversary model of fraudsters behaviour to improve oversampling in credit card fraud detection
    """

    def __init__(self, generator=None, n_jobs=1, training_frac=0.8, random_state=1):
        self.generator = generator
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.training_frac = training_frac
        self.useful_features = ['x_terminal_id', 'y_terminal_id', 'TX_AMOUNT']
        self.regressors = {}

        # TODO: let the user choose progress_bar and use_memory_fs
        pandarallel.initialize(nb_workers=self.n_jobs, progress_bar=True, use_memory_fs=False)

    def generate_transactions(self, n_customers=50, n_terminals=10):

        self.generator = Generator(n_customers = n_customers, n_terminals=n_terminals, random_state=self.random_state)
        self.generator.generate()

    def load_trasactions(self, filename):
        self.generator = Generator()
        self.generator.load(filename)

    def _make_couples(group):
        group = group.sort_values(['TX_DATETIME'], axis=0, ascending=True)
        couples_df = pd.DataFrame(columns = [*'prev_' + group.columns, *'next_' + group.columns])
        if group.shape[0] > 1: 
            group_as_list = group.values
            for first_tr, second_tr in zip(group_as_list, group_as_list[1:]):
                a_series = pd.Series([*first_tr,*second_tr], index = couples_df.columns).to_frame().T
                couples_df = pd.concat([couples_df,a_series])
            return couples_df
        else: 
            return couples_df
            
    def create_couples(self):
        
        full_transactions_table =  self.generator.transactions_df.merge(self.generator.terminal_profiles_table, 'inner')
        full_frauds_table = full_transactions_table[full_transactions_table['TX_FRAUD'] == 1]

        interestig_columns = ['TX_DATETIME', 'CUSTOMER_ID', 'x_terminal_id', 'y_terminal_id', 'TX_AMOUNT', 'TX_FRAUD']
        clean_frauds_df = full_frauds_table[interestig_columns].sort_values(['TX_DATETIME'], axis=0, ascending=True)

        grouped = clean_frauds_df.groupby('CUSTOMER_ID')
        if self.n_jobs == 1:
            results = grouped.apply(self._make_couples)
        else:
            results = grouped.parallel_apply(self._make_couples)

        results.reset_index(inplace=True, drop=True)
        results.drop(['prev_CUSTOMER_ID', 'next_CUSTOMER_ID'], axis = 1, inplace=True)
        
        self.couples = results


    def tune_regressors(self, searcher, search_parameters, regressor):
        
        training_set = self.couples.sample(frac= 0.8, random_state=self.random_state)
        features_prev = list(map(lambda x: 'prev_' + str(x),  self.useful_features))
        X_train = training_set[features_prev]

        for feature_to_predict in self.useful_features:
            y_train = training_set['next_'+feature_to_predict]
            search =  searcher(regressor, **search_parameters, n_jobs=self.n_jobs, random_state=self.random_state)
            search.fit(X_train, y_train)
            regressor = search.best_estimator_
            self.regressors[feature_to_predict] = regressor



    def fit_regressors(self, metric):

        
        training_set = self.couples.sample(frac= 0.8, random_state=self.random_state)
        test_set = self.couples[~self.couples.isin(training_set)].dropna()

        features_prev = list(map(lambda x: 'prev_' + str(x),  self.useful_features))

        X_train = training_set[features_prev]
        X_test = test_set[features_prev]

        for feature_to_predict in self.useful_features:
            regressor = self.regressors[feature_to_predict]
            
            y_train = training_set['next_'+feature_to_predict]
            y_test = test_set['next_'+feature_to_predict]
            y_naive = X_test['prev_'+feature_to_predict]
            
            regressor.fit(X_train, y_train)
            
            y_pred = regressor.predict(X_test)

            score = metric(y_test, y_pred)
            naive_score = metric(y_test, y_naive)

            regressor.score = score
            regressor.naive_score = naive_score
            regressor.feature_names = X_train.columns





