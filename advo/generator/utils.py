import numpy as np
from typing import Tuple

# TODO: fix this function, why the conditions? LOC and SCALE as parameters
def compute_first_centre() -> Tuple[float, float]:
    """
    Compute the coordinates of the first center.
    
    Returns
    -------
    Tuple[float, float]
        A tuple containing the x and y coordinates of the first center.
    """
    x, y = np.random.normal(loc=70, scale=4), np.random.normal(loc=20, scale=6)
    while x > 100 or x < 0:
        x = np.random.normal(loc=70, scale=4)
    while y > 100 or y < 0:
        y = np.random.normal(loc=20, scale=4)
    return x, y


def compute_first_time() -> int:
    """
    Compute the first time of transaction.
    
    Returns
    -------
    int
        An integer representing the first time of transaction.
    """
    time_tx = abs(int(np.random.normal(86400 / 8, 20000)))
    return time_tx



def compute_first_amount(fraudsters_mean: float, fraudsters_var: float) -> float:
    """
    Compute the first amount of transaction.
    
    Parameters
    ----------
    fraudsters_mean: float
        The mean amount of transaction for fraudsters.
    fraudsters_var: float
        The variance of the amount of transaction for fraudsters.
    
    Returns
    -------
    float
        A float representing the first amount of transaction.
    """
    first_amount = np.random.normal(loc=fraudsters_mean, scale=fraudsters_var)
    first_amount = np.round(first_amount, decimals=2)
    # print('Amount is' + str(first_amount))
    return first_amount


def compute_new_centre(previous_X: float, previous_Y: float) -> Tuple[float, float]:
    """
    Compute the coordinates of the new center.
    
    Parameters
    ----------
    previous_X: float
        The previous x coordinate of the center.
    previous_Y: float
        The previous y coordinate of the center.
    
    Returns
    -------
    Tuple[float, float]
        A tuple containing the x and y coordinates of the new center.
    """
    small_x = previous_X / 100
    small_y = previous_Y / 100

    X = (small_x * 0.75 - 0.06 * small_y + 0.08 * small_x ** 2 + 0.5 * small_y ** 2 - small_x * small_y * 0.3) * 100
    Y = (small_x * 0.3 + 0.85 * small_y - 0.3 * small_x ** 2 + 0.1 * small_y ** 2 - small_x * small_y * 0.3) * 100

    new_centreX = np.random.normal(loc=X, scale=5)
    new_centreY = np.random.normal(loc=Y, scale=5)

    while new_centreX > 100 or new_centreX < 0:
        # print('+')
        new_centreX = np.random.normal(loc=X, scale=10)
    while new_centreY > 100 or new_centreY < 0:
        new_centreY = np.random.normal(loc=Y, scale=10)
        # print('+')
    return new_centreX, new_centreY


def compute_new_amt(previous_AMT: float, previous_X: float, previous_Y: float, fraudsters_var: float) -> float:
    """
    Compute the new amount of transaction.
    
    Parameters
    ----------
    previous_AMT: float
        The previous amount of transaction.
    previous_X: float
        The previous x coordinate of the center.
    previous_Y: float
        The previous y coordinate of the center.
    fraudsters_var: float
        The variance of the amount of transaction for fraudsters.
    
    Returns
    -------
    float
        A float representing the new amount of transaction.
    """
    value = np.random.normal(1.1 * previous_AMT - 0.2 * previous_X + 0.7 + previous_Y * 0.1, fraudsters_var / 2)
    return np.abs(value)


# TODO. why not separating the computation of the first time from the computation
# of the second time similarly to the other functions? The condition
# if previous_fraud[1] == day is very unclear.

# If first fraud of the day, go random, oterwise take from previous fraud.
def compute_time(previous_fraud: tuple, day: int) -> int:
    """
    Computes the time of a transaction.

    Args:
        previous_fraud (tuple): A tuple containing the previous time of fraud and the day on which it occurred.
        day (int): The current day.

    Returns:
        int: The time of the transaction.
    """
    if previous_fraud[1] == day:  # DAY
        time_tx = int(previous_fraud[0] + abs(np.random.normal(loc=0, scale=30000)))
        while time_tx > 86400:
            time_tx = int(previous_fraud[0] + abs(np.random.normal(loc=0, scale=30000)))
    else:
        time_tx = compute_first_time()
    return time_tx
