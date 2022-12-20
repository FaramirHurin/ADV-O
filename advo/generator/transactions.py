import numpy as np
import pandas as pd 
import random
from .utils import *
from .radius import *

from typing import List, Union

class Transaction():
    def __init__(self, transaction_id, x, y, amount, time, terminal_id, customer_id):
        self.transaction_id = transaction_id
        self.x = x
        self.y = y
        self.amount = amount
        self.time = time
        self.terminal_id = terminal_id
        self.customer_id = customer_id

    def __str__(self):
        return "Transaction ID: " + str(self.transaction_id) + ", x: " + str(self.x) + ", y: " + str(self.y) + ", amount: " + str(self.amount) + ", time: " + str(self.time) + ", terminal ID: " + str(self.terminal_id) + ", customer ID: " + str(self.customer_id)

    def get_dataframe(self):
        return pd.DataFrame({'transaction_id': [self.transaction_id], 'x': [self.x], 'y': [self.y], 'amount': [self.amount], 'time': [self.time], 'terminal_id': [self.terminal_id], 'customer_id': [self.customer_id]})

    def get_distance_to_terminal(self, terminal):
        return self.get_distance_to_point(terminal.x, terminal.y)

    def get_distance_to_point(self, x, y):
        return np.sqrt(np.square(self.x - x) + np.square(self.y - y))

    def get_distance_to_customer(self, customer):
        return self.get_distance_to_point(customer.x, customer.y)

    def get_distance_to_transaction(self, transaction):
        return self.get_distance_to_point(transaction.x, transaction.y)

    def get_distance(self, x, y):
        return np.sqrt(np.square(self.x - x) + np.square(self.y - y))


# TODO: refactor this 
def compromise_user(customer_profile: dict, compromission_probability: float = 0.03) -> int:

    """Determines whether a customer has been compromised.

    This method determines whether a customer has been compromised based on the given customer profile and the
    specified compromission probability. If the customer has already been compromised, the method returns 1. Otherwise,
    the method uses a binomial distribution to determine whether the customer has been compromised based on the
    compromission probability.

    Args:
        customer_profile (dict): The customer profile to check.
        compromission_probability (float, optional): The probability of a customer being compromised. Defaults to 0.03.

    Returns:
        int: 1
    """

    if customer_profile['compromised'] == 0:
        compromised = np.random.binomial(n=1, p=compromission_probability)
    else:
        compromised = 1
    return compromised


def generate_genuine_transactions(customer_transactions: pd.DataFrame, nb_tx: int, customer_profile: dict, day: int) -> pd.DataFrame:

    """Generates genuine transactions for a customer.

    This method generates a specified number of genuine transactions for a given customer profile and day. The transactions
    are generated using a normal distribution for the transaction amount, and a uniform distribution for the transaction
    time. The generated transactions are appended to the given customer transactions dataframe.

    Args:
        customer_transactions (pd.DataFrame): The existing customer transactions.
        nb_tx (int): The number of transactions to generate.
        customer_profile (dict): The customer profile to use for generating transactions.
        day (int): The day on which the transactions occurred.

    Returns:
        pd.DataFrame: The updated customer transactions dataframe.
    """


    # If nb_tx positive, let us generate transactions
    if nb_tx > 0:

        for tx in range(nb_tx):

            # Time of transaction: Around noon, std 20000 seconds. This choice aims at simulating the fact that
            # most transactions occur during the day.
            time_tx = int(np.random.normal(86400 / 2, 20000))

            # If transaction time between 0 and 86400, let us keep it, otherwise, let us discard it
            if (time_tx > 0) and (time_tx < 86400):

                # Amount is drawn from a normal distribution
                amount = np.random.normal(customer_profile.mean_amount, customer_profile.std_amount)

                # If amount negative, draw from a uniform distribution
                if amount < 0:
                    amount = np.random.uniform(0, customer_profile.mean_amount * 2)

                amount = np.round(amount, decimals=2)

                if len(customer_profile.available_terminals) > 0:
                    terminal_id = random.choice(customer_profile.available_terminals)
                    is_fraud = 0

                    customer_transactions.append([time_tx, day,
                                                  customer_profile.CUSTOMER_ID,
                                                  terminal_id, amount, is_fraud])
    return customer_transactions



def generate_first_fraud(customer_profile: pd.DataFrame, day: int, fraudsters_mean: float, fraudsters_var: float, terminal_profiles_table: pd.DataFrame, r: int = 20) -> List[Union[int, str, float]]:
    """Generates the first fraudulent transaction for a compromised customer.

    This method generates the first fraudulent transaction for a compromised customer. The transaction is generated using a
    normal distribution for the transaction amount, and a uniform distribution for the transaction location. The location
    is selected randomly from a set of terminals within a specified radius of a randomly chosen point.

    Args:
        customer_profile (pd.DataFrame): The customer profile to use for generating the transaction.
        day (int): The day on which the transaction occurred.
        fraudsters_mean (float): The mean amount of fraudulent transactions.
        fraudsters_var (float): The variance of fraudulent transactions.
        terminal_profiles_table (pd.DataFrame): The table of terminal profiles to use for selecting the transaction location.
        r (int, optional): The radius around the randomly chosen point to use for selecting the transaction location. Defaults to 20.

    Returns:
        List[Union[int, str, float]]: The generated fraudulent transaction.
    """
    new_centreX, new_centreY = compute_first_centre()
    terminals_xy = terminal_profiles_table[['x_terminal_id', 'y_terminal_id']]
    available_terminals, weights = get_list_terminals_within_radius_from_point(new_centreX, new_centreY,
                                                                               terminals_xy, r)
    terminal_id = random.choices(available_terminals, weights=weights)[0]
    time_tx = compute_first_time()  # Time
    assert time_tx > 0
    amount = compute_first_amount(fraudsters_mean, fraudsters_var)
    is_fraud = 1
    fraud = [time_tx, day, customer_profile.CUSTOMER_ID, terminal_id, amount, is_fraud]
    # print('First Fraud is ' + str(fraud))
    return fraud


def generate_following_fraud(
    previous_fraud: pd.DataFrame, 
    terminal_profiles_table: pd.DataFrame, 
    customer_profile: pd.DataFrame, 
    day: int, 
    fraudsters_mean: float,
    fraudsters_var: float, 
    x_y_terminals: np.ndarray, 
    r: float = 20
) -> Tuple[List[Union[int, float]], int]:
    """
    This function takes the previous fraud transaction, the terminal profiles table, the customer profile, 
    the day of the transaction, the mean and variance of fraudsters, the x and y coordinates of terminals, 
    and a radius, and returns the next fraud transaction and the transaction time.

    Parameters:
    previous_fraud (pd.DataFrame): A DataFrame representing the previous fraud transaction.
    terminal_profiles_table (pd.DataFrame): A DataFrame containing the terminal profiles.
    customer_profile (pd.DataFrame): A DataFrame containing the customer profile.
    day (int): The day of the transaction.
    fraudsters_mean (float): The mean of fraudsters.
    fraudsters_var (float): The variance of fraudsters.
    x_y_terminals (np.ndarray): A 2D array containing the x and y coordinates of terminals.
    r (float): The radius to search within (defaults to 20).

    Returns:
    Tuple[List[Union[int, float]], int]: A tuple containing the next fraud transaction and the transaction time.
    """

    previous_terminal_ID = previous_fraud[3]  # TERMINAL_ID
    previous_terminal = terminal_profiles_table[terminal_profiles_table['TERMINAL_ID'] == previous_terminal_ID]
    previous_terminal = previous_terminal.squeeze()
    previous_X = previous_terminal['x_terminal_id']
    previous_Y = previous_terminal['y_terminal_id']
    new_centreX, new_centreY = compute_new_centre(previous_X, previous_Y)
    available_terminals, weights = get_list_terminals_within_radius_from_point(new_centreX, new_centreY, x_y_terminals, r)
    time_tx = compute_time(previous_fraud, day)  # Time

    # If transaction time between 0 and 86400, let us keep it, otherwise, let us discard it

    terminal_id = random.choices(available_terminals, weights=weights)[0]

    assert terminal_id in available_terminals
    is_fraud = 1
    amount = compute_new_amt(previous_fraud[4], previous_X, previous_Y, fraudsters_var)  # TX_AMOUNT

    fraud = [time_tx, day, customer_profile.CUSTOMER_ID, terminal_id, amount, is_fraud]
    # print('New Fraud is ' + str(fraud))
    return fraud, time_tx


def generate_fraudulent_transactions(
    customer_transactions: pd.DataFrame, 
    nb_txm: int, 
    customer_profile: pd.DataFrame, 
    day: int, 
    fraudsters_mean: float,
    fraudsters_var: float, 
    terminal_profiles_table: pd.DataFrame, 
    x_y_terminals: np.ndarray
) -> pd.DataFrame:
    """
    This function takes the customer transactions, the number of transactions to generate, the customer profile, 
    the day of the transaction, the mean and variance of fraudsters, the terminal profiles table, 
    and the x and y coordinates of terminals, and returns a DataFrame of fraudulent transactions.

    Parameters:
    customer_transactions (pd.DataFrame): A DataFrame of customer transactions.
    nb_txm (int): The number of transactions to generate.
    customer_profile (pd.DataFrame): A DataFrame containing the customer profile.
    day (int): The day of the transaction.
    fraudsters_mean (float): The mean of fraudsters.
    fraudsters_var (float): The variance of fraudsters.
    terminal_profiles_table (pd.DataFrame): A DataFrame containing the terminal profiles.
    x_y_terminals (np.ndarray): A 2D array containing the x and y coordinates of terminals.

    Returns:
    pd.DataFrame: A DataFrame of fraudulent transactions.
    """

    time_tx = 0
    for index in range(nb_txm):
        if customer_transactions[-1][-1] == 0:  # TX_FRAUD
            new_fraud = generate_first_fraud(customer_profile, day, fraudsters_mean, fraudsters_var,terminal_profiles_table)
            if not (new_fraud is None):
                customer_transactions.append(new_fraud)
        else:
            new_fraud, this_time = generate_following_fraud(customer_transactions[-1], terminal_profiles_table,
                                                            customer_profile, day, fraudsters_mean, fraudsters_var, x_y_terminals)
            customer_transactions.append(new_fraud)
    return customer_transactions
