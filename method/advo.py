from generator import Generator
import pandas as pd
import numpy as np
from pandarallel import pandarallel
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from typing import Dict, Type, Any, Callable
from sklearn.model_selection import BaseSearchCV
from sklearn.base import BaseEstimator


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
        transaction_df (pd.DataFrame): a dataframe containing the transactions data.
        n_jobs (int): the number of cores to use when running certain methods in parallel.
        random_state (int): the random seed to use when generating transactions.
        training_frac (float): the fraction of transactions to use for training the regression models.
        useful_features (List[str]): a list of the names of the features to use when training the regression models.
        regressors (Dict[str, Any]): a dictionary containing the trained regression models, with the features to predict as
            keys.
    """

    def __init__(self, transaction_df=None, n_jobs=1, training_frac=0.8, random_state=1):
        self.transaction_df = transaction_df
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.training_frac = training_frac
        self.useful_features = ['x_terminal_id', 'y_terminal_id', 'TX_AMOUNT']
        self.regressors = {}

        # TODO: let the user choose progress_bar and use_memory_fs
        pandarallel.initialize(nb_workers=self.n_jobs, progress_bar=True, use_memory_fs=False)

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
        self.transaction_df = generator.transactions_df

    def load_trasactions(self, filename: str) -> None:
        """
        Load transactions from a file after creating a Generator instance.

        This method loads transactions from the specified file after creating a `Generator` instance to hold the data. The
        `Generator` instance is stored in the `generator` attribute of the current instance.

        Args:
            filename (str): the name of the file containing the transactions to load.
        """
        generator = Generator()
        #TODO: decouple the generator from the transactions
        generator.load(filename)
        self.transaction_df = generator.transactions_df

    def _make_couples(self, group: pd.DataFrame) -> pd.DataFrame: 
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
            
    def create_couples(self):
        """
        Create couples of fraudulent transactions for each customer.

        This method uses the `transactions_df` and `terminal_profiles_table` attributes of the `generator` attribute to create
        a table of fraudulent transactions. It then groups the transactions by customer, and applies the `_make_couples` method
        to each group in order to create couples of fraudulent transactions for each customer. The results are stored in the
        `couples` attribute of the current instance.

        This method can run in parallel using multiple cores if the `n_jobs` attribute is set to a value greater than 1.
        """
        
        full_transactions_table =  self.generator.transactions_df.merge(self.generator.terminal_profiles_table, 'inner')
        full_frauds_table = full_transactions_table[full_transactions_table['TX_FRAUD'] == 1]

        #TODO: let the user choose the columns to keep...
        interestig_columns = ['TX_DATETIME', 'CUSTOMER_ID', 'x_terminal_id', 'y_terminal_id', 'TX_AMOUNT', 'TX_FRAUD']
        clean_frauds_df = full_frauds_table[interestig_columns].sort_values(['TX_DATETIME'], axis=0, ascending=True)

        #TODO: let the user choose the columns to keep...
        grouped = clean_frauds_df.groupby('CUSTOMER_ID')
        if self.n_jobs == 1:
            results = grouped.apply(self._make_couples)
        else:
            results = grouped.parallel_apply(self._make_couples)

        results.reset_index(inplace=True, drop=True)
        results.drop(['prev_CUSTOMER_ID', 'next_CUSTOMER_ID'], axis = 1, inplace=True)
        
        self.couples = results

    
    def tune_regressors(
            self,
            searcher: Type[BaseSearchCV],
            search_parameters: Dict[str, Any],
            regressor: Type[BaseEstimator],
        ) -> None:
        """Tunes the hyperparameters of a set of regression models using a search algorithm.
        The regressor will be part of the self.regressors dictionary, with the feature to predict as key.

        Args:
            searcher: An object that implements a search algorithm, such as GridSearchCV or RandomizedSearchCV.
            search_parameters: A dictionary of hyperparameters for the searcher.
            regressor: A regression model to be tuned.
        
        Returns:
            None
        """
        
        training_set = self.couples.sample(frac= 0.8, random_state=self.random_state)
        features_prev = list(map(lambda x: 'prev_' + str(x),  self.useful_features))
        X_train = training_set[features_prev]

        for feature_to_predict in self.useful_features:
            y_train = training_set['next_'+feature_to_predict]
            search =  searcher(regressor, **search_parameters, n_jobs=self.n_jobs, random_state=self.random_state)
            search.fit(X_train, y_train)
            regressor = search.best_estimator_
            self.regressors[feature_to_predict] = regressor

    def _oversample_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Oversamples a dataframe of transactions by predicting the values for certain features using machine learning models.
        
        Args:
            df: A pandas DataFrame containing transaction data.
        
        Returns:
            A new pandas DataFrame with the original transaction data and the predicted values for the specified features.
        """
        old_columns = df.columns
        for feature in self.regressors.keys():
            #TODO: make it feature agnostic!
            feature_true_name = feature[18:]
            df['new_'+str(feature_true_name)] = self.regressors[feature].predict(df[self.useful_features].values)
        df['new_TX_FRAUD'] = 1
        df['new_CUSTOMER_ID'] = df['CUSTOMER_ID']
        filter_col = [col for col in df if col.startswith('new')]
        new_frauds = df[filter_col]
        df=df.drop(filter_col, axis=1)
        new_frauds = new_frauds.rename(dict(zip(filter_col, old_columns)), axis = 1)
        new_frauds['TX_AMOUNT'] = round( new_frauds['TX_AMOUNT'], 2)
        new_frauds['x_terminal_id'] = new_frauds['x_terminal_id'].apply(lambda x: max(0, min(100, x)))
        new_frauds['y_terminal_id'] = new_frauds['y_terminal_id'].apply(lambda x: max(0, min(100, x)))
        return new_frauds

    def _enrich_dataframe(self, transactions_df: pd.DataFrame, layers: int) -> pd.DataFrame:
        """Adds multiple layers of predicted fraudulent transactions to a dataframe of transactions.
        
        This method filters the input dataframe to only include rows with a value of 1 in the 'TX_FRAUD' column, 
        and groups the remaining rows by the values in the 'CUSTOMER_ID' column. It then applies the '_oversample_df' 
        method to each group of transactions, adding the predicted fraudulent transactions to the original dataframe. 
        This process is repeated for the number of iterations specified by the 'layers' argument.
        
        Args:
            transactions_df: A pandas DataFrame containing transaction data.
            layers: An integer specifying the number of times to apply the '_oversample_df' method and add 
                    the predicted fraudulent transactions to the dataframe.
        
        Returns:
            A new pandas DataFrame with the original transaction data and the added layers of predicted fraudulent transactions.
        """
    
        frauds_df = transactions_df[transactions_df['TX_FRAUD'] == 1]
        frauds_groups = frauds_df.groupby(['CUSTOMER_ID'], axis=0 )    
        
        for _ in range(layers):

                if self.n_jobs == 1:
                    to_verify_fraud_list = frauds_groups.apply(self._oversample_df).reset_index(drop=True)
                else:
                    to_verify_fraud_list = frauds_groups.parallel_apply(self._oversample_df).reset_index(drop=True)
                
                
                transactions_df = transactions_df.append(to_verify_fraud_list, ignore_index=True)
                frauds_df = transactions_df[transactions_df['TX_FRAUD'] == 1]
                frauds_groups = frauds_df.groupby(['CUSTOMER_ID'], axis=0 )
        
        return transactions_df


    def fit_regressors(self, metric: Callable[[np.ndarray, np.ndarray], float]) -> None:
        """Trains a set of regression models to predict future values of features,
        i.e. the transaction amount, the next terminal id, etc. of the next fraudulent transaction.
    
        Args:
            metric: A function that takes two arrays and returns a single numeric value,
                    representing the quality of the prediction.
        
        Returns:
            None
        """

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





