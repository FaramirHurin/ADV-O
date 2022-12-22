import numpy as np 
import pandas as pd
from ADVO.generator.entities import Terminal, Customer


class Generator():

    def __init__(self, random_state = 42):
        self.random_state = np.random.RandomState(random_state)
        self.terminals = []
        self.customers = []
        self.transactions = []

    def generate_terminals(self, n_terminals = 100):
        self.terminals = []
        for terminal_id in range(n_terminals):
            x_terminal_id, y_terminal_id = np.random.uniform(0, 100), np.random.uniform(0, 100)
            terminal = Terminal(terminal_id, x_terminal_id, y_terminal_id, random_state=self.random_state)
            self.terminals.append(terminal)

    def generate_customers(self, n_customers=500, radius=20, max_days_from_compromission=3, compromission_probability=0.03):
        if not len(self.terminals):
            raise ValueError("You need to generate terminals before generating customers")
        
        self.customers = []
        for customer_id in range(n_customers):
            
            x_customer_id, y_customer_id = np.random.uniform(0, 100), np.random.uniform(0, 100)
            mean_amount = np.random.uniform(5, 100)  
            std_amount = mean_amount / 2  
            mean_nb_tx_per_day = np.random.uniform(0, 4)
            
            customer = Customer(customer_id, x_customer_id, y_customer_id, radius, mean_amount, std_amount, mean_nb_tx_per_day, max_days_from_compromission, compromission_probability, random_state=self.random_state)
            customer.set_available_terminals(self.terminals)
            self.customers.append(customer)        
            
    def generate_transactions(self, nb_days_to_generate = 8, start_date="2018-04-01") -> None:
        self.start_date = start_date
        #TODO: see how to parallelize
        for customer in self.customers:
            customer.generate_transactions(nb_days_to_generate)
            self.transactions.extend(customer.transactions)

    def get_transactions_df(self) -> pd.DataFrame:
        transactions_list = [transaction.get_dataframe() for transaction in self.transactions]
        return pd.concat(transactions_list).round(2)

    def get_terminals_df(self) -> pd.DataFrame:
        terminals_list = [terminal.get_dataframe() for terminal in self.terminals]
        return pd.concat(terminals_list).round(2)

    def get_customers_df(self) -> pd.DataFrame:
        customers_list = [customer.get_dataframe() for customer in self.customers]
        return pd.concat(customers_list).round(2)


