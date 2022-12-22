import numpy as np
import pandas as pd
from typing import List, Tuple

FRAUDULENT_MEAN_AMOUNT_FACTOR = 0.9
FRAUDULENT_STD_AMOUNT_FACTOR = 1.3

class Customer():
    def __init__(self, customer_id: int, x: float, y: float, 
                radius: float, mean_amt: float, std_amt: float,
                mean_n_transactions:int, max_days_from_compromission:int,
                compromission_probability: float, random_state: np.random.RandomState, 
                compromised: bool = False):
        self.customer_id = customer_id
        self.x = x
        self.y = y
        self.radius = radius
        self.mean_amt = mean_amt
        self.std_amt = std_amt
        self.mean_n_transactions = mean_n_transactions
        self.compromised = compromised
        self.max_days_from_compromission = max_days_from_compromission
        self.compromission_probability = compromission_probability
        self.random_state = random_state
        self.all_terminals = []
        self.available_terminals = []
        self.available_terminals_weights = []
        self.transactions = []
        
    def set_available_terminals(self, terminals):
        self.all_terminals = terminals
        self.available_terminals = []
        self.available_terminals_weights = []
        
        for terminal in self.all_terminals:
            dist = terminal.distance_to_point(self.x, self.y)
            if dist < self.radius:
                self.available_terminals.append(terminal)
                self.available_terminals_weights.append(np.exp(-dist**2))
        # Normalize weights so that they sum to 1
        self.available_terminals_weights = [weight / sum(self.available_terminals_weights) for weight in self.available_terminals_weights]

    #TODO: Add a parameter to specify the number of days to generate transactions for
    def generate_transactions(self, n_days):
        
        if not len(self.available_terminals):  # Check if there are available terminals
            raise ValueError("This customer does not have available terminals, make sure you have called the set_available_terminals functions and that the radius is not too small")
        
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

    def _generate_genuine_transactions(self, today_n_transactions, day):
        for _ in range(today_n_transactions):
            time_tx_seconds = int(np.clip(np.random.normal(86400 / 2, 20000), 0, 86400)) # Mean 12pm, std 5.5h
            amount = np.round(np.clip(np.random.normal(self.mean_amt, self.std_amt),0, None), decimals=2)
            terminal = self.random_state.choice(self.available_terminals, p=self.available_terminals_weights)
            transaction = Transaction(time_tx_seconds,day, self, terminal, amount, False)
            self.transactions.append(transaction)
 
    def _generate_fraudulent_transactions(self, today_n_transactions, day):
        for _ in range(today_n_transactions):
            if not self.transactions[-1].is_fraud: 
                # If the last transaction was not fraud, generate a first fraud 
                self.x = np.random.beta(a=70, b=30) * 100
                self.y = np.random.beta(a=20, b=80) * 100
                self.mean_amt = np.random.normal(self.mean_amt) * FRAUDULENT_MEAN_AMOUNT_FACTOR
                self.std_amt = np.random.normal(self.std_amt) * FRAUDULENT_STD_AMOUNT_FACTOR
                self.set_available_terminals(self.all_terminals)
                
                time_tx_seconds = int(np.clip(np.random.normal(86400 / 2, 20000), 0, 86400))
                amount = np.round(np.clip(np.random.normal(loc=self.mean_amt, scale=self.std_amt), 0, None), decimals=2)
                terminal = self.random_state.choice(self.available_terminals, p=self.available_terminals_weights)
                transaction = Transaction(time_tx_seconds,day, self, terminal, amount, True)
                self.transactions.append(transaction)
            else: 
                # If the last transaction was fraud, generate a following fraud
                last_transaction = self.transactions[-1]
                last_terminal_x, last_terminal_y, last_fraud_day, last_fraud_time, last_fraud_amount = last_transaction.terminal.x, last_transaction.terminal.y, last_transaction.day, last_transaction.tx_time, last_transaction.amount
                self.x, self.y = self._set_following_fraud_coordinates(last_terminal_x, last_terminal_y)
                self.set_available_terminals(self.all_terminals)
                if day == last_fraud_day: # if not the first fraud of the day 
                    time_tx_seconds = int(np.clip(last_fraud_time + abs(np.random.normal(loc=0, scale=30000)), 0, 86400))
                else:
                    time_tx_seconds = int(np.clip(np.random.normal(86400 / 2, 20000), 0, 86400))
                amount = np.abs(np.random.normal(1.1 * last_fraud_amount - 0.2 * last_terminal_x + 0.7 + last_terminal_y * 0.1, self.std_amt / 2))
                terminal = self.random_state.choice(self.available_terminals, p=self.available_terminals_weights)
                transaction = Transaction(time_tx_seconds,day, self, terminal, amount, True)
                self.transactions.append(transaction)

    def _set_following_fraud_coordinates(self, last_terminal_x, last_terminal_y):
        small_x = last_terminal_x / 100
        small_y = last_terminal_y / 100

        X = (small_x * 0.75 - 0.06 * small_y + 0.08 * small_x ** 2 + 0.5 * small_y ** 2 - small_x * small_y * 0.3) * 100
        Y = (small_x * 0.3 + 0.85 * small_y - 0.3 * small_x ** 2 + 0.1 * small_y ** 2 - small_x * small_y * 0.3) * 100

        return  np.random.beta(a=X, b=5) * 100, np.random.beta(a=Y, b=5) * 100

    def get_dataframe(self) -> pd.DataFrame:
        customers_df = pd.DataFrame(data=[self.customer_id, self.x, self.y, self.mean_amt, self.std_amt, self.mean_n_transactions, self.compromised ] , index=['customer_id', 'x_customer', 'y_customer', 'mean_amount', 'std_amount', 'mean_nb_tx_per_day', 'compromised'])
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
        return pd.DataFrame([[self.terminal_id, self.x, self.y]], columns=['terminal_id', 'x_terminal', 'y_terminal'])


class Transaction():
    def __init__(self, tx_time: int, day: int, customer: Customer, terminal: Terminal, amount: float, is_fraud: bool):
        self.tx_time = tx_time
        self.day = day
        self.customer = customer
        self.terminal = terminal
        self.amount = amount
        self.is_fraud = is_fraud
    
    def get_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([[self.tx_time, self.day, self.customer. customer_id,self. terminal.terminal_id, self.amount, self.is_fraud]], columns=['tx_time', 'tx_day', 'customer_id', 'terminal_id','amount', 'is_fraud'])
