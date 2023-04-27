import numpy as np 
import pandas as pd
from datetime import datetime, timedelta
from ADVO.generator.entities import Terminal, Customer
import matplotlib.pyplot as plt

from multiprocessing import Pool

class Generator():

    def __init__(self, radius=10, random_state = 42, n_jobs = 1):
        self.random_state = np.random.RandomState(random_state)
        self.terminals = []
        self.customers = []
        self.transactions = []
        self.n_jobs = n_jobs
        self.radius = radius
        

    def generate_object(self):
        r= np.random.beta(a=60, b=40) * self.radius
        theta = np.random.beta(a=50, b=50) * self.radius
        #self.x = radius * np.cos(theta)  # np.random.beta(a=70, b=30) * 100
        #self.y = radius * np.sin(theta)  # np.random.beta(a=20, b=80) * 100
        #r = np.random.random() * self.radius
        #theta = np.random.random() * 2 * np.pi
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return x, y

    def generate_terminal(self, terminal_id):
        x_terminal_id, y_terminal_id = self.generate_object()
        terminal = Terminal(terminal_id, x_terminal_id, y_terminal_id, random_state=self.random_state)
        print('Terminal {} generated'.format(terminal_id), end='\r')
        return terminal

    def generate_terminals(self, n_terminals=100):
        self.terminals = []

        with Pool(self.n_jobs) as pool:
            self.terminals = pool.map(self.generate_terminal, range(n_terminals))
        
        # x = [terminal.x for terminal in self.terminals]
        # y = [terminal.y for terminal in self.terminals]
        # terminal_profiles_table = pd.DataFrame()
        # terminal_profiles_table['x'] = x
        # terminal_profiles_table['y'] = y
        # # Plot locations of terminals
        # plt.scatter(terminal_profiles_table['x'] ,
        #            terminal_profiles_table['y'], s=0.1,
        #            color='blue')
        # plt.title('Locations of terminals')

    def generate_customer(self, customer_id, terminals, radius, mean_amount, std_amount, mean_nb_tx_per_day, max_days_from_compromission, compromission_probability):
        x_customer_id, y_customer_id = self.generate_object()
        customer = Customer(customer_id, x_customer_id, y_customer_id, radius, mean_amount, std_amount, mean_nb_tx_per_day, max_days_from_compromission, compromission_probability, random_state=self.random_state, terminals=terminals, radius_to_be=self.radius)
        print('Customer {} generated'.format(customer_id), end='\r')
        return customer

    def generate_customers(self, n_customers=200, radius=20, max_days_from_compromission=3, compromission_probability=0.03):
        if not len(self.terminals):
            raise ValueError("You need to generate terminals before generating customers")
        
        self.customers = []

        params = []
        for customer_id in range(n_customers):
            mean_amount = np.random.uniform(5, 100)
            std_amount = mean_amount / 2
            mean_nb_tx_per_day = np.random.uniform(0, 4)

            params.append((customer_id, self.terminals, radius, mean_amount, std_amount, mean_nb_tx_per_day, max_days_from_compromission, compromission_probability))

        with Pool(self.n_jobs) as pool:
            self.customers = pool.starmap(self.generate_customer, params)

    # def generate_customers(self, n_customers=200, radius=20, max_days_from_compromission=3, compromission_probability=0.03):
    #     if not len(self.terminals):
    #         raise ValueError("You need to generate terminals before generating customers")
        
    #     self.customers = []
    #     for customer_id in range(n_customers):
            
    #         x_customer_id, y_customer_id = self.generate_object() # np.random.uniform(0, 100), np.random.uniform(0, 100)
    #         mean_amount = np.random.uniform(5, 100)  
    #         std_amount = mean_amount / 2  
    #         mean_nb_tx_per_day = np.random.uniform(0, 4)
            
    #         customer = Customer(customer_id, x_customer_id, y_customer_id, radius, mean_amount, std_amount, mean_nb_tx_per_day, max_days_from_compromission, compromission_probability, random_state=self.random_state, terminals=self.terminals)
    #         self.customers.append(customer)
        # x = [customer.x for customer in self.customers]
        # y = [customer.y for customer in self.customers]
        # customer_profiles_table = pd.DataFrame()
        # customer_profiles_table['x'] = x
        # customer_profiles_table['y'] = y
        # plt.scatter(customer_profiles_table['x'],
        #             customer_profiles_table['y'], s=0.1,
        #             color='red')
        # plt.title('Locations of customers')

    def generate_transactions_for_customer(self, customer, nb_days_to_generate):
        customer.generate_transactions(nb_days_to_generate)
        print("Customer {} transactions generated".format(customer.customer_id), end='\r')
        return customer.transactions

    def generate_transactions(self, nb_days_to_generate=180, start_date="2018-04-01") -> None:
        self.start_date = start_date
        self.transactions = []

        with Pool(self.n_jobs) as pool:
            try:
                results = pool.starmap(self.generate_transactions_for_customer, [(customer, nb_days_to_generate) for customer in self.customers])
            except:
                DEBUG =0
                results = pool.starmap(self.generate_transactions_for_customer, [(customer, nb_days_to_generate) for customer in self.customers])

        for result in results:
            self.transactions.extend(result)

    def generate(self, filename='dataset.csv', n_terminals = 100, n_customers=200, radius=20, max_days_from_compromission=3, compromission_probability=0.02, nb_days_to_generate = 180, start_date="2018-04-01"):
        self.generate_terminals( n_terminals)
        self.generate_customers( n_customers, radius, max_days_from_compromission, compromission_probability)
        self.generate_transactions( nb_days_to_generate, start_date)

        transactions_df = self.get_transactions_df().merge(self.get_terminals_df(), left_on='TERMINAL_ID', right_on='TERMINAL_ID', how='left')
        transactions_df['TX_DATETIME'] =  datetime.strptime(start_date, '%Y-%m-%d') +pd.to_timedelta(transactions_df['TX_DAY'], unit='d') + pd.to_timedelta(transactions_df['TX_TIME'], unit='s')
        transactions_df.to_csv('utils/'+filename, index=False)
        return transactions_df

    def get_terminals_df(self, decimals=2) -> pd.DataFrame:
        terminals_list = [terminal.get_dataframe() for terminal in self.terminals]
        terminal_df = pd.concat(terminals_list).infer_objects()
        terminal_df.loc[:,terminal_df.select_dtypes(['float64']).columns] = terminal_df.select_dtypes(['float64']).round(decimals)
        return terminal_df.reset_index(drop=True)

    def get_customers_df(self, decimals=2) -> pd.DataFrame:
        customers_list = [customer.get_dataframe() for customer in self.customers]
        customers_df = pd.concat(customers_list).infer_objects()
        customers_df.loc[:,customers_df.select_dtypes(['float64']).columns] = customers_df.select_dtypes(['float64']).round(decimals)
        return customers_df.reset_index(drop=True)

    def get_transactions_df(self, decimals=2) -> pd.DataFrame:
        transactions_list = [transaction.get_dataframe() for transaction in self.transactions]
        transactions_df = pd.concat(transactions_list).infer_objects()
        transactions_df.loc[:,transactions_df.select_dtypes(['float64']).columns] = transactions_df.select_dtypes(['float64']).round(decimals)
        return transactions_df.reset_index(drop=True)

    def get_dataframes(self, decimals=2):
        return self.get_terminals_df(decimals), self.get_customers_df(decimals), self.get_transactions_df(decimals)