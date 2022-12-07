import numpy as np
import random
from .utils import *
from .radius import *

def compromise_user(customer_profile, compromission_probability=0.03):
    if customer_profile['compromised'] == 0:
        compromised = np.random.binomial(n=1, p=compromission_probability)
    else:
        compromised = 1
    return compromised


def generate_genuine_transactions(customer_transactions, nb_tx, customer_profile, day):
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



def generate_first_fraud(customer_profile, day, fraudsters_mean, fraudsters_var, terminal_profiles_table, r=20):
    # print('-')
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


def generate_following_fraud(previous_fraud, terminal_profiles_table, customer_profile, day, fraudsters_mean,
                                     fraudsters_var, x_y_terminals, r=20):

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


def generate_fraudulent_transactions(customer_transactions, nb_txm, customer_profile, day, fraudsters_mean,
                                     fraudsters_var, terminal_profiles_table, x_y_terminals):

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
