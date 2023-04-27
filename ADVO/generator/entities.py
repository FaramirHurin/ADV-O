import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple
import numpy as np
from scipy.spatial import KDTree

FRAUDULENT_MEAN_AMOUNT_FACTOR = 1.05
FRAUDULENT_STD_AMOUNT_FACTOR = 0.8

class Customer():
    def __init__(self, customer_id: int, x: float, y: float,
                radius: float, mean_amt: float, std_amt: float,
                mean_n_transactions:int, max_days_from_compromission:int,
                compromission_probability: float, random_state: np.random.RandomState, 
                compromised: bool = False, terminals: List['Terminal'] = [], radius_to_be=10):
        self.customer_id = customer_id
        self.x = x
        self.y = y
        self.x_history = {'genuine': [], 'fraud': []}
        self.y_history = {'genuine': [], 'fraud': []}
        self.radius = radius
        self.mean_amt = mean_amt
        self.std_amt = std_amt
        self.mean_n_transactions = mean_n_transactions
        self.compromised = compromised
        self.max_days_from_compromission = max_days_from_compromission
        self.compromission_probability = compromission_probability
        self.random_state = random_state
        self.all_terminals = terminals
        self.available_terminals = []
        self.available_terminals_weights = []
        self.transactions = []

        self.kdtree = KDTree([(terminal.x, terminal.y) for terminal in self.all_terminals])

        self.set_available_terminals()
        self.radius_to_be = radius_to_be

    def get_closest_terminal(self):
        
        # Find the index of the closest terminal and its distance
        _, closest_index = self.kdtree.query((self.x, self.y))

        # Get the closest terminal's coordinates
        closest_terminal = self.all_terminals[closest_index]

        return closest_terminal    

    def set_available_terminals(self):
        self.available_terminals = []
        self.available_terminals_weights = []
        assert self.radius >0
        # Find indices of terminals within the radius
        try:

            indices_within_radius = self.kdtree.query_ball_point([self.x, self.y], self.radius)
        except:
            print('X and Y are: ')
            print(self.x, self.y)
            indices_within_radius = self.kdtree.query_ball_point([self.x, self.y], self.radius)


        for index in indices_within_radius:
            terminal = self.all_terminals[index]
            dist = terminal.distance_to_point(self.x, self.y) ** 3
            self.available_terminals.append(terminal)
            self.available_terminals_weights.append(1/dist)

        # Normalize weights so that they sum to 1
        if len(self.available_terminals_weights) > 0:
            self.available_terminals_weights = [weight / sum(self.available_terminals_weights) for weight in self.available_terminals_weights]
        else:
            DEBUG=0


    #TODO: Add a parameter to specify the number of days to generate transactions for
    def generate_transactions(self, n_days):
        
        if not len(self.available_terminals):  # Check if there are available terminals
            print('X and Y are' + str(self.x) + ' '+ str(self.y))
            print(self.x, self.y)
            raise ValueError(f"Customer {self.customer_id} does not have available terminals, make sure you have called the set_available_terminals functions and that the radius is not too small fjiedrfj0jedWSTFIJKRUY+LOPèòED3wsfjrtuy247+iklop890èò")
        
        self.transactions = []
        days_from_compromission = 0  # Number of max frauds days before card blocking

        for day in range(n_days):
            today_n_transactions = self.random_state.poisson(self.mean_n_transactions)
            if today_n_transactions > 0:
                if self.compromised and len(self.transactions) and days_from_compromission < self.max_days_from_compromission:
                    self._generate_fraudulent_transactions(today_n_transactions, day)
                    days_from_compromission += 1 
                else:
                    self._generate_genuine_transactions(today_n_transactions, day)
                    self.compromised = self.random_state.binomial(1, self.compromission_probability)
        DEBUG = 0

    def _generate_genuine_transactions(self, today_n_transactions, day):
        for _ in range(today_n_transactions):
            time_tx_seconds = int(np.clip(np.random.normal(86400 / 2, 20000), 0, 86400)) # Mean 12pm, std 5.5h
            amount = np.round(np.clip(np.random.normal(self.mean_amt, self.std_amt),0, None), decimals=2)
            
            # terminal = self.random_state.choice(self.available_terminals, p=self.available_terminals_weights)
            terminal = self.get_closest_terminal()

            self._update_coordinate_history(terminal)
            transaction = Transaction(time_tx_seconds,day, self, terminal, amount, False)
            self.transactions.append(transaction)
 
    def _generate_fraudulent_transactions(self, today_n_transactions, day):
        for _ in range(today_n_transactions):
            if not self.transactions[-1].is_fraud: 
                # If the last transaction was not fraud, generate a first fraud
                radius = np.random.beta(a=70, b=30) * self.radius_to_be
                theta = np.random.beta(a=45, b=55) * self.radius_to_be
                self.x = radius * np.cos(theta) # np.random.beta(a=70, b=30) * 100
                self.y = radius * np.sin(theta) # np.random.beta(a=20, b=80) * 100
                
                self.mean_amt = np.random.normal(self.mean_amt) * FRAUDULENT_MEAN_AMOUNT_FACTOR
                self.std_amt = np.random.normal(self.std_amt) * FRAUDULENT_STD_AMOUNT_FACTOR
                self.set_available_terminals()
                
                time_tx_seconds = int(np.clip(np.random.normal(86400 / 2, 20000), 0, 86400))
                while time_tx_seconds < self.transactions[-1].tx_time:
                    time_tx_seconds = int(np.clip(np.random.normal(86400 / 2, 20000), 0, 86400))
                amount = np.round(np.clip(np.random.normal(loc=self.mean_amt, scale=self.std_amt), 0, None), decimals=2)
                
                # terminal = self.random_state.choice(self.available_terminals, p=self.available_terminals_weights)
                terminal = self.get_closest_terminal()

                self._update_coordinate_history(terminal, fraud=True)
                transaction = Transaction(time_tx_seconds,day, self, terminal, amount, True)
                self.transactions.append(transaction)
            else: 
                # If the last transaction was fraud, generate a following fraud
                last_transaction = self.transactions[-1]
                last_terminal_x, last_terminal_y, last_fraud_day, last_fraud_time, last_fraud_amount = last_transaction.terminal.x, last_transaction.terminal.y, last_transaction.day, last_transaction.tx_time, last_transaction.amount
                self.x, self.y = self._set_following_fraudster_coordinates(last_terminal_x, last_terminal_y)
                
                self.set_available_terminals()
                if day == last_fraud_day: # if not the first fraud of the day
                    time_tx_seconds = int(np.clip(np.random.normal(86400 / 2, 20000), 0, 86400))
                    while time_tx_seconds < self.transactions[-1].tx_time:
                        time_tx_seconds = int(np.clip(np.random.normal(86400 / 2, 20000), 0, 86400))
                    # time_tx_seconds = int(np.clip(last_fraud_time + abs(np.random.normal(loc=0, scale=30000)), 0, 86400))
                else:
                    time_tx_seconds = int(np.clip(np.random.normal(86400 / 2, 20000), 0, 86400))
                amount = np.abs(np.random.normal(0.95 * last_fraud_amount + 0.2 * last_terminal_x + 0.07 + last_terminal_y , self.std_amt / 2))
                # terminal = self.random_state.choice(self.available_terminals, p=self.available_terminals_weights)
                terminal = self.get_closest_terminal()
                
                self._update_coordinate_history(terminal, fraud=True)
                transaction = Transaction(time_tx_seconds,day, self, terminal, amount, True)
                self.transactions.append(transaction)

    def _set_following_fraudster_coordinates(self, last_terminal_x, last_terminal_y):
        X = (last_terminal_x +(np.sin(last_terminal_x) + np.cos(last_terminal_y))/2 * self.radius_to_be + np.random.rand(1)[0]/2) / 2
        Y = ( last_terminal_x + (np.sin(last_terminal_y) + np.cos(last_terminal_x))/2 * self.radius_to_be + np.random.rand(1)[0]/2) / 2
        return  X,Y

    def _update_coordinate_history(self, terminal, fraud=False):
        if fraud:
            self.x_history['fraud'].append(terminal.x)
            self.y_history['fraud'].append(terminal.y)
        else:
            self.x_history['genuine'].append(terminal.x)
            self.y_history['genuine'].append(terminal.y)

    def get_dataframe(self) -> pd.DataFrame:
        customers_df = pd.DataFrame(data=[self.customer_id, self.x, self.y, self.mean_amt, self.std_amt, self.mean_n_transactions, bool(self.compromised) ] , index=['CUSTOMER_ID', 'X_CUSTOMER', 'Y_CUSTOMER', 'MEAN_AMT', 'STD_AMT', 'MEAN_NB_TX_PER_DAY', 'COMPROMISED']).T
        return customers_df



class Terminal():

    def __init__(self, terminal_id: int, x: float, y: float, random_state: np.random.RandomState):
        self.terminal_id = terminal_id
        self.x = x
        self.y = y
        self.random_state = random_state

    def distance_to_point(self, x: float, y: float) -> float:
        squared_diff_x_y = np.square((x - self.x, y - self.y))
        return np.sqrt(np.sum(squared_diff_x_y))

    def get_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([[self.terminal_id, self.x, self.y]], columns=['TERMINAL_ID', 'X_TERMINAL', 'Y_TERMINAL'])


class Transaction():
    def __init__(self, tx_time: int, day: int, customer: Customer, terminal: Terminal, amount: float, is_fraud: bool):
        self.tx_time = tx_time
        self.day = day
        self.customer = customer
        self.terminal = terminal
        self.amount = amount
        self.is_fraud = is_fraud        
    
    def get_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([[self.tx_time, self.day, self.customer. customer_id,self. terminal.terminal_id, self.amount, self.is_fraud]], columns=['TX_TIME', 'TX_DAY', 'CUSTOMER_ID', 'TERMINAL_ID','TX_AMOUNT', 'TX_FRAUD'])
