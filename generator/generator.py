import random 
import numpy as np 
import pandas as pd
from .functions import *


class Generator():
    
    def __init__(self, n_customers, n_terminals, radius=20, nb_days=8, start_date="2018-04-01", random_state = 2):
        self.n_customers = n_customers
        self.n_terminals = n_terminals
        self.radius = radius
        self.nb_days = nb_days
        self.start_date = start_date
        self.random_state = random_state
        
    
    def _generate_transactions_table(self, customer_profile):
        days_from_comprmission = 0  # Number of max frauds days
        customer_transactions = []
        random.seed(int(customer_profile['CUSTOMER_ID']))
        # For all days
        for day in range(self.nb_days):
            # Random number of transactions for that day
            nb_tx = np.random.poisson(customer_profile.mean_nb_tx_per_day)
            if nb_tx == 0:
                continue

            if customer_profile['compromised'] == 1 and len(customer_transactions) > 0 and days_from_comprmission < 3:
                customer_transactions = generate_fraudulent_transactions(customer_transactions, nb_tx, customer_profile, day, self.fraudsters_mean, self.fraudsters_var, self.terminal_profiles_table, self.x_y_terminals)
                days_from_comprmission += 1

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
    
    
    def initialize(self):
        self.terminal_profiles_table = generate_terminal_profiles_table(self.n_customers, self.random_state)
        self.customer_profiles_table = generate_customer_profiles_table(self.n_terminals, self.random_state)
        self.x_y_terminals = self.terminal_profiles_table[['x_terminal_id', 'y_terminal_id']].values.astype(float)
        self.customer_profiles_table['available_terminals'] = self.customer_profiles_table.apply\
            (lambda x: get_list_terminals_within_radius(x, x_y_terminals=self.x_y_terminals, r=self.radius), axis=1)

        self.fraudsters_mean = np.random.normal(np.mean(self.customer_profiles_table['mean_amount'])) * 1.1
        self.fraudsters_var = np.random.normal(np.mean(self.customer_profiles_table['std_amount'])) * 0.8
    
    
    def generate(self):
        
        groups = self.customer_profiles_table.groupby('CUSTOMER_ID')
        transactions_df=groups.apply(lambda x: self._generate_transactions_table(customer_profile=x.iloc[0])).reset_index(drop=True)
        return transactions_df
    