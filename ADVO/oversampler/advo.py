import pandas as pd
import numpy as np
from pandarallel import pandarallel
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from typing import Dict, Type, Any, Callable
from sklearn.base import BaseEstimator
from sklearn.model_selection import BaseCrossValidator
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
import pickle
from ..generator import  Generator

class ADVO():
    

    """
    ADVO: An Adversary model of fraudsters behaviour to improve oversampling in credit card fraud detection.

    This class implements an adversarial model of fraudsters' behavior that can be used to improve oversampling in credit
    card fraud detection. The model uses transactions data to create pairs of fraudulent transactions for each customer, and
    trains regression models to predict future fraudulent transactions for each customer.

    The class provides methods for generating transactions data, loading transactions from a file, creating pairs of
    fraudulent transactions, tuning the hyperparameters of regression models, fitting the regression models to data, and
    generating new synthetic transactions using the trained models.

    Attributes:
        transactions_df (pd.DataFrame): a dataframe containing the transactions data.
        n_jobs (int): the number of cores to use when running certain methods in parallel.
        random_state (int): the random seed to use when generating transactions.
        verbose (bool): whether to print progress bars when running certain methods in parallel.
        training_frac (float): the fraction of transactions to use for training the regression models.
        sampling_strategy (float): the fraction of fraudulent transactions to achieve when oversampling.
        useful_features (List[str]): a list of the names of the features to use when training the regression models.
        regressors (Dict[str, Any]): a dictionary containing the trained regression models, with the features to predict as
            keys.
        searcher (Type[BaseSearchCV]): the search cross-validation object to use for hyperparameter tuning.
        search_parameters (Dict[str, Any]): a dictionary containing the parameters to use when tuning the regression models.
        regressor (Type[BaseEstimator]): the regressor object to use for training the regression models.
        mimo (bool): whether to use a multi-input multi-output regressor.
    """


    def __init__(self, 
                transactions_df=None, 
                n_jobs=1, 
                training_frac=0.8,
                sampling_strategy=0.5, 
                random_state=1, 
                verbose=False,
                searcher: Type[BaseCrossValidator] = RandomizedSearchCV,
                candidate_regressors = None,
                search_parameters: Dict[str, Any] = {'alpha': [0.1, 1, 2, 5, 10], 'fit_intercept': [True, False]},
                regressor: Type[BaseEstimator] = Ridge(),
                mimo = False):
        self.transactions_df = transactions_df
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.training_frac = training_frac
        self.sampling_strategy = sampling_strategy
        #TODO: let the user choose the features to use
        self.useful_features = ['X_TERMINAL', 'Y_TERMINAL', 'TX_AMOUNT']
        self.regressors = {}
        self.searcher = searcher
        self.candidate_regressors = candidate_regressors
        self.search_parameters = search_parameters
        self.regressor = regressor
        self.mimo = mimo
  

        # TODO: let the user choose progress_bar and use_memory_fs
        pandarallel.initialize(nb_workers=self.n_jobs, progress_bar=verbose, use_memory_fs=False, verbose=0)

    def generate_transactions(self, n_customers=50, n_terminals=10):
        """
        Generate transactions after creating a Generator instance.

        This method generates transactions after creating a `Generator` instance to hold the data. The
        `Generator` instance is stored in the `generator` attribute of the current instance.

        Args:
            n_customers (int): the number of customers to generate.
            n_terminals (int): the number of terminals to generate.
        """

        generator = Generator(n_customers = n_customers, n_terminals=n_terminals, random_state=self.random_state)
        generator.generate()
        self.transactions_df = generator.transactions_df

    def set_transactions(self, X_train, y_train) -> None:
        """ Standard setter method for the attribute transactions_df. 
        
        This method takes as input a set of transactions features and labels, merges them, and stores them in the attribute transactions_df.

        Args:
            X_train (pd.DataFrame): a DataFrame containing the features of the transactions.
            y_train (pd.Series): a Series containing the labels of the transactions.
        """
        self.transactions_df = pd.concat([X_train, y_train], axis=1)

    def load_trasactions(self, filename: str) -> None:
        """
        Load transactions from a file.

        This method loads transactions from a file. The file can be either a csv or a pickle file. The loaded transactions
        are stored in the `transactions_df` attribute of the current instance.

        Args:
            filename (str): the name of the file to load the transactions from.
        """

        if  filename.endswith('.csv'):
            full_transactions_table = pd.read_csv(filename)
        elif filename.endswith('.pkl'):
            with open(filename, 'rb') as f:
                full_transactions_table = pickle.load(f)
        else:
            raise ValueError('Invalid file format')
        
        self.transactions_df = full_transactions_table[['TX_DATETIME', 'CUSTOMER_ID', 'TERMINAL_ID', 'TX_AMOUNT', 'TX_TIME_SECONDS', 'TX_FRAUD']]
        self.transactions_df = self.transactions_df.drop_duplicates()
        self.transactions_df = self.transactions_df.reset_index(drop=True)


    def _make_consecutive_couples(self, group: pd.DataFrame) -> pd.DataFrame: 
        """
        Create couples of fraudulent transactions for a given customer group.

        This method takes a DataFrame representing a group of fraudulent transactions for a single customer. It sorts the
        transactions by date, and then creates couples of transactions by iterating through the transactions and pairing each
        transaction with the subsequent transaction in the list. The resulting couples are stored in a DataFrame and returned.

        Args:
            group (pandas.DataFrame): a DataFrame representing a group of fraudulent transactions for a single customer.

        Returns:
            pandas.DataFrame: a DataFrame containing couples of fraudulent transactions for the given customer.
        """
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
            
    def _make_all_couples(self, group: pd.DataFrame) -> pd.DataFrame: 
        """
        Create all possible couples of fraudulent transactions for a given customer group.

        This method takes a DataFrame representing a group of fraudulent transactions for a single customer. It sorts the
        transactions by date, and then creates all possible couples of transactions by iterating through the transactions
        and pairing each transaction with all subsequent transactions in the list. The resulting couples are stored in a
        DataFrame and returned.

        Args:
            group (pandas.DataFrame): a DataFrame representing a group of fraudulent transactions for a single customer.

        Returns:
            pandas.DataFrame: a DataFrame containing all possible couples of fraudulent transactions for the given customer.
        """
        group = group.sort_values(['TX_DATETIME'], axis=0, ascending=True)
        couples_df = pd.DataFrame(columns=[*'prev_' + group.columns, *'next_' + group.columns])
        if group.shape[0] > 1:
            group_as_list = group.values
            for i, first_tr in enumerate(group_as_list[:-1]):
                for second_tr in group_as_list[i+1:]:
                    a_series = pd.Series([*first_tr, *second_tr], index=couples_df.columns).to_frame().T
                    couples_df = pd.concat([couples_df, a_series])
            return couples_df
        else:
            return couples_df


    def create_couples(self):
        """
        Create couples of fraudulent transactions for each customer.

        This method uses the `transactions_df` and `terminal_profiles_table` attributes of the `generator` attribute to create
        a table of fraudulent transactions. It then groups the transactions by customer, and applies the `_make_couples` method
        to each group in order to create couples of fraudulent transactions for each customer. The results are stored in the
        `couples` attribute of the current instance.

        This method can run in parallel using multiple cores if the `n_jobs` attribute is set to a value greater than 1.
        """
        
        full_frauds_table = self.transactions_df[self.transactions_df['TX_FRAUD'] == 1]

        #TODO: let the user choose the columns to keep...
        interestig_columns = ['TX_DATETIME', 'CUSTOMER_ID', 'X_TERMINAL', 'Y_TERMINAL', 'TX_AMOUNT', 'TX_FRAUD']
        clean_frauds_df = full_frauds_table[interestig_columns].sort_values(['TX_DATETIME'], axis=0, ascending=True)

        #TODO: let the user choose the columns to keep...
        grouped = clean_frauds_df.groupby('CUSTOMER_ID')
        if self.n_jobs == 1:
            results = grouped.apply(self._make_consecutive_couples)
        else:
            results = grouped.parallel_apply(self._make_consecutive_couples)

        results.reset_index(inplace=True, drop=True)
        results.drop(['prev_CUSTOMER_ID', 'next_CUSTOMER_ID'], axis = 1, inplace=True)
        
        self.couples = results

    def tune_best_regressors(self):
        training_set = self.couples
        features_prev = list(map(lambda x: 'prev_' + str(x),  self.useful_features))
        X_train = training_set[features_prev]

        if not self.mimo:
            for feature_to_predict in self.useful_features:
                y_train = training_set['next_'+feature_to_predict]
                search =  self.searcher(self.regressor, self.search_parameters, n_jobs=self.n_jobs, random_state=self.random_state)
                search.fit(X_train, y_train)
                regressor = search.best_estimator_
                self.regressors[feature_to_predict] = regressor
        else:
            y_train = training_set[['next_'+feature for feature in self.useful_features]]
            search =  self.searcher(self.regressor, self.search_parameters, n_jobs=self.n_jobs, random_state=self.random_state)
            search.fit(X_train, y_train)
            regressor = search.best_estimator_
            self.regressors = regressor

    def tune_regressors(self, metric: Callable[[np.ndarray, np.ndarray], float] = r2_score) -> None:
        """
        Tunes the hyperparameters of one regressor per feature specified in the self.useful_features attribute. 
        All regressors belong to the same family specified by the 'self.regressor' parameter. 
        The set of hyperparameter-tuned regressors is stored in the self.regressors attribute.

        Args:
            metric: the evaluation metric for the score of the regressors
        
        Returns:
            None
        """
        
        training_set = self.couples.sample(frac=self.training_frac, random_state=self.random_state)
        test_set = self.couples[~self.couples.isin(training_set)].dropna()
        features_prev = list(map(lambda x: 'prev_' + str(x),  self.useful_features))
        X_train = training_set[features_prev]
        X_test = test_set[features_prev]

        if not self.mimo:
            for feature_to_predict in self.useful_features:
                y_train = training_set['next_'+feature_to_predict]
                y_test = test_set['next_'+feature_to_predict]
                y_naive = X_test['prev_'+feature_to_predict]
                search =  self.searcher(self.regressor, self.search_parameters, n_jobs=self.n_jobs, random_state=self.random_state)
                search.fit(X_train, y_train)
                regressor = search.best_estimator_
                y_hat = regressor.predict(X_test)
                regressor.score = metric(y_test, y_hat)
                regressor.naive_score = metric(y_test, y_naive)
                regressor.feature_names = X_train.columns
                self.regressors[feature_to_predict] = regressor
            
        else:
            y_train = training_set[['next_'+feature for feature in self.useful_features]]
            y_test = test_set[['next_'+feature for feature in self.useful_features]]
            y_naive = X_test[['prev_'+feature for feature in self.useful_features]]
            search =  self.searcher(self.regressor, self.search_parameters, n_jobs=self.n_jobs, random_state=self.random_state)
            search.fit(X_train, y_train)
            regressor = search.best_estimator_
            y_hat = regressor.predict(X_test)
            #TODO: check if metric is a function of the whole vector or of each element
            regressor.score = metric(y_test, y_hat)
            regressor.naive_score = metric(y_test, y_naive)
            regressor.feature_names = X_train.columns
            self.regressors = regressor
        #save regressors in pickle file
        with open("utils/regressors.pkl", 'wb') as f:
            pickle.dump(self.regressors, f)

    def _oversample_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Oversamples a dataframe of transactions by predicting the values for certain features using machine learning models.
        
        Args:
            df: A pandas DataFrame containing transaction data.
        
        Returns:
            A new pandas DataFrame with the original transaction data and the predicted values for the specified features.
        """
        for feature in self.regressors.keys():
            df.loc[:, 'new_'+str(feature)] = self.regressors[feature].predict(df[self.useful_features].rename(columns=lambda x: 'prev_'+str(x)))
        df.loc[:, 'new_TX_FRAUD'] = 1
        df.loc[:,'new_CUSTOMER_ID'] = df['CUSTOMER_ID']
        filter_col = [col for col in df if col.startswith('new')]
        new_frauds = df[filter_col].copy()
        df=df.drop(filter_col, axis=1)
        new_frauds.columns = [column[len('new_'):] for column in new_frauds.columns]
        
        #new_frauds.loc[:, 'TX_AMOUNT'] = round( new_frauds.loc[:, 'TX_AMOUNT'], 2)
        #new_frauds.loc[:, 'x_terminal_id'] = new_frauds.loc[:, 'x_terminal_id'].apply(lambda x: max(0, min(100, x)))
        #new_frauds.loc[:, 'y_terminal_id'] = new_frauds.loc[:, 'y_terminal_id'].apply(lambda x: max(0, min(100, x)))
        
        new_frauds.loc[:, 'TX_AMOUNT'] = new_frauds.loc[:, 'TX_AMOUNT'].round(2)
        new_frauds.loc[:, 'X_TERMINAL'].clip(0, 100, inplace=True)
        new_frauds.loc[:, 'Y_TERMINAL'].clip(0, 100, inplace=True)
        

        return new_frauds

    def insert_synthetic_frauds(self, transactions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds multiple layers of predicted fraudulent transactions to a dataframe of transactions until a specified percentage of fraud is reached.

        This method filters the input dataframe to only include rows with a value of 1 in the 'TX_FRAUD' column,
        and groups the remaining rows by the values in the 'CUSTOMER_ID' column. It then applies the '_oversample_df'
        method to each group of transactions, adding the predicted fraudulent transactions to the original dataframe.
        This process is repeated until the specified percentage of fraud is reached. The resulting dataframe is then
        sampled to ensure that the exact percentage of fraud is maintained.

        Args:
        transactions_df: A pandas DataFrame containing transaction data.
        percentage: A float specifying the target percentage of fraud in the resulting dataframe.

        Returns:
        A new pandas DataFrame with the original transaction data and the added layers of predicted fraudulent transactions.
        """
        transactions_df_copy = transactions_df.copy()
        frauds_df = transactions_df[transactions_df['TX_FRAUD'] == 1]
        frauds_groups = frauds_df.groupby('CUSTOMER_ID', axis=0 )
        total_count = len(transactions_df)
        initial_count = len(frauds_df)
        target_count = (total_count - initial_count)  * self.sampling_strategy
        current_count = len(frauds_df)
        layers = 0
        generated_frauds_counter = 0
        only_synthetic_frauds_df = pd.DataFrame()
        while current_count < target_count:
            if self.n_jobs == 1:
                to_verify_fraud_list = frauds_groups.apply(self._oversample_df).reset_index(drop=True)
            else:
                to_verify_fraud_list = frauds_groups.parallel_apply(self._oversample_df).reset_index(drop=True)
            generated_frauds_counter += len(to_verify_fraud_list)
            transactions_df = pd.concat([transactions_df, to_verify_fraud_list], ignore_index=True)
            only_synthetic_frauds_df = pd.concat([only_synthetic_frauds_df, to_verify_fraud_list], ignore_index=True)
            frauds_df = transactions_df[transactions_df['TX_FRAUD'] == 1]
            frauds_groups = frauds_df.groupby('CUSTOMER_ID', axis=0 )
            current_count = len(frauds_df)
            layers += 1
        only_synthetic_frauds_df = only_synthetic_frauds_df.sample(int(target_count) - initial_count, random_state=self.random_state)
        transactions_df_copy = pd.concat([transactions_df_copy, only_synthetic_frauds_df], ignore_index=True)
        return transactions_df_copy

   
    def fit_regressors(self) -> None:

        features_prev = list(map(lambda x: 'prev_' + str(x),  self.useful_features))

        X_train = self.couples[features_prev]

        if not self.mimo:
            for feature_to_predict in self.useful_features:
                regressor = self.regressors[feature_to_predict]
                y_train = self.couples['next_'+feature_to_predict]
                regressor.fit(X_train, y_train)
        else:
            y_train = self.couples[['next_'+feature for feature in self.useful_features]]
            self.regressor.fit(X_train, y_train)
                
    def select_best_regressor(self, candidate_regressors, parameters_set):
            best_score = np.NINF
            naive_scores = {}
            scores_df = pd.DataFrame()
            for idx, candidate_regressor in enumerate(candidate_regressors):

                self.regressor = candidate_regressors[idx]
                self.search_parameters = parameters_set[idx]
                self.tune_regressors()
                self.fit_regressors()

                scores = {}
                for feature_to_predict in self.useful_features:                    
                    regressor = self.regressors[feature_to_predict]
                    scores[feature_to_predict] = regressor.score
                    naive_scores[feature_to_predict] = regressor.naive_score
                single_scores_df = pd.DataFrame(scores, index=[candidate_regressor])
                scores_df = pd.concat([scores_df, single_scores_df], axis=0)

                mean_value = np.array(list(scores.values())).mean()

                if mean_value > best_score:
                    best_score = mean_value
                    self.best_regressor = candidate_regressor
            merged_scores_df = pd.concat([scores_df, pd.DataFrame(naive_scores, index=['Naive'])], axis=0)
            self.regressor = self.best_regressor
            self.search_parameters = parameters_set[candidate_regressors.index(self.best_regressor)]
            return merged_scores_df

    def fit_resample(self, X_train, y_train):

        self.set_transactions(X_train, y_train)
        self.create_couples()
        
        self.tune_regressors()
        self.fit_regressors()
        self.transactions_df = self.insert_synthetic_frauds(self.transactions_df)
        return self.transactions_df[self.useful_features], self.transactions_df['TX_FRAUD']


