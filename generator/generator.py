import random 
import numpy as np 
import pandas as pd
import pickle
from .functions import *


class Generator():
    
    def __init__(self, n_customers=50, n_terminals=10, radius=20, nb_days=8, start_date="2018-04-01", random_state = 2, \
        max_days_from_compromission=3):

        self.n_customers = n_customers
        self.n_terminals = n_terminals
        self.radius = radius
        self.nb_days = nb_days
        self.start_date = start_date
        self.random_state = random_state
        self.max_days_from_compromission = max_days_from_compromission
    
        self.terminal_profiles_table = generate_terminal_profiles_table(self.n_customers, self.random_state)
        self.customer_profiles_table = generate_customer_profiles_table(self.n_terminals, self.random_state)
        self.x_y_terminals = self.terminal_profiles_table[['x_terminal_id', 'y_terminal_id']].values.astype(float)
        self.customer_profiles_table['available_terminals'] = self.customer_profiles_table.apply\
            (lambda x: get_list_terminals_within_radius(x, x_y_terminals=self.x_y_terminals, r=self.radius), axis=1)

        self.fraudsters_mean = np.random.normal(np.mean(self.customer_profiles_table['mean_amount'])) * 1.1
        self.fraudsters_var = np.random.normal(np.mean(self.customer_profiles_table['std_amount'])) * 0.8
    

        
    def generate(self) -> None:
        """Generates transactions data for the customers in this dataset.

        This function groups the customer profiles by customer ID and then generates
        transactions data for each group using the `_generate_transactions_table`
        function. The generated transactions data is stored in the `transactions_df`
        attribute of this object.

        Returns:
            None
        """
        
        groups = self.customer_profiles_table.groupby('CUSTOMER_ID')
        self.transactions_df = groups.apply(lambda x: self._generate_transactions_table(customer_profile=x.iloc[0])).reset_index(drop=True)    

    def export(self, filename: str ='transactions.csv', format: str ='csv') -> None:
        """Exports the transactions data to a file.

        This function saves the transactions data stored in the `transactions_df`
        attribute of this object to a file with the specified `filename` and
        `format`. If `format` is 'csv', the data is saved as a CSV file. If
        `format` is 'pkl', the data is pickled and saved to a binary file.

        Args:
            filename (str): The name of the file to save the data to.
            format (str): The format to use for saving the data. Must be 'csv' or
                'pkl'.

        Returns:
            None
        """
        if format=='csv':
            self.transactions_df.to_csv(filename,index=False)
        elif format=='pkl':
            with open(filename, 'wb') as f:
                pickle.dump(self.transactions_df,f)


    def _generate_transactions_table(self, customer_profile):
        days_from_compromission = 0  # Number of max frauds days
        customer_transactions = []
        random.seed(int(customer_profile['CUSTOMER_ID']))
        # For all days
        for day in range(self.nb_days):
            # Random number of transactions for that day
            nb_tx = np.random.poisson(customer_profile.mean_nb_tx_per_day) + 1  #+1 to avoid 0

            if customer_profile['compromised'] == 1 and len(customer_transactions) > 0 and days_from_compromission < self.max_days_from_compromission:
                customer_transactions = generate_fraudulent_transactions(customer_transactions, nb_tx, customer_profile, day, self.fraudsters_mean, self.fraudsters_var, self.terminal_profiles_table, self.x_y_terminals)
                days_from_compromission += 1

            else:
                customer_transactions = generate_genuine_transactions(customer_transactions, nb_tx, customer_profile, day)
                customer_profile['compromised'] = compromise_user(customer_profile)

        customer_transactions = pd.DataFrame(customer_transactions,
                                             columns=['TX_TIME_SECONDS_INSIDE_DAY', 'TX_TIME_DAYS', 'CUSTOMER_ID',
                                                      'TERMINAL_ID', 'TX_AMOUNT', 'TX_FRAUD'])

        if customer_transactions.shape[0] > 0:
            customer_transactions['TX_TIME_SECONDS'] = customer_transactions["TX_TIME_SECONDS_INSIDE_DAY"] + \
                                                       customer_transactions["TX_TIME_DAYS"] * 86400
            customer_transactions['TX_DATETIME'] = pd.to_datetime(customer_transactions['TX_TIME_SECONDS'], unit='s',
                                                                  origin=self.start_date)
            customer_transactions = customer_transactions[
                ['TX_DATETIME', 'CUSTOMER_ID', 'TERMINAL_ID', 'TX_AMOUNT', 'TX_TIME_SECONDS', 'TX_FRAUD']]

        return customer_transactions