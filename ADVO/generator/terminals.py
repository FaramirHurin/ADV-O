import numpy as np
import pandas as pd
from typing import List, Tuple

def generate_terminal_profiles_table(n_terminals: int, random_state: int = 0) -> pd.DataFrame:

    """Generates a table of terminal profiles.

    This method generates a table of terminal profiles with the specified number of terminals. Each terminal is assigned
    a unique identifier and x,y coordinates. The x,y coordinates are generated randomly from uniform distributions.

    Args:
        n_terminals (int): The number of terminals to generate.
        random_state (int, optional): The seed to use for the random number generator. Defaults to 0.

    Returns:
        pd.DataFrame: A table containing the generated terminal profiles.
    """

    np.random.seed(random_state)

    terminal_id_properties = []

    # Generate terminal properties from random distributions
    for terminal_id in range(n_terminals):
        x_terminal_id = np.random.uniform(0, 100)
        y_terminal_id = np.random.uniform(0, 100)

        terminal_id_properties.append([terminal_id,
                                       x_terminal_id, y_terminal_id])

    terminal_profiles_table = pd.DataFrame(terminal_id_properties, columns=['TERMINAL_ID','x_terminal_id', 'y_terminal_id'])

    return terminal_profiles_table


def get_list_terminals_within_radius(
    customer_profile: pd.DataFrame, 
    x_y_terminals: np.ndarray, 
    r: float
) -> List[int]:
    """
    This function takes a customer's profile, a 2D array containing the x and y coordinates of terminals, 
    and a radius, and returns a list of terminal IDs that are within the given radius from the customer's location.

    Parameters:
    customer_profile (pd.DataFrame): A DataFrame containing the customer's profile.
    x_y_terminals (np.ndarray): A 2D array containing the x and y coordinates of terminals.
    r (float): The radius to search within.

    Returns:
    List[int]: A list of terminal IDs that are within the given radius from the customer's location.
    """
    # Use numpy arrays in the following to speed up computations

    # Location (x,y) of customer as numpy array
    x_y_customer = customer_profile[['x_customer_id', 'y_customer_id']].values.astype(float)

    # Squared difference in coordinates between customer and terminal locations
    
    squared_diff_x_y = np.square(x_y_customer - x_y_terminals)

    # Sum along rows and compute suared root to get distance
    dist_x_y = np.sqrt(np.sum(squared_diff_x_y, axis=1))

    # Get the indices of terminals which are at a distance less than r
    available_terminals = list(np.where(dist_x_y < r)[0])

    # Return the list of terminal IDs
    return available_terminals


def get_list_terminals_within_radius_from_point(
    x_centre: float, 
    y_centre: float, 
    x_y_terminals: np.ndarray, 
    r: float
) -> Tuple[List[int], np.ndarray]:
    """
    This function takes the x and y coordinates of a point, a 2D array containing the x and y coordinates of terminals, 
    and a radius, and returns a list of terminal IDs that are within the given radius from the point, 
    as well as an array of corresponding weights.

    Parameters:
    x_centre (float): The x coordinate of the point.
    y_centre (float): The y coordinate of the point.
    x_y_terminals (np.ndarray): A 2D array containing the x and y coordinates of terminals.
    r (float): The radius to search within.

    Returns:
    Tuple[List[int], np.ndarray]: A tuple containing a list of terminal IDs that are within the given radius from the point, 
    and an array of corresponding weights.
    """
    # Use numpy arrays in the following to speed up computations

    # Location (x,y) of customer as numpy array
    x_y_customer = [x_centre, y_centre]

    

    # Squared difference in coordinates between customer and terminal locations
    squared_diff_x_y = np.square(x_y_customer - x_y_terminals)

    weights = np.sum(squared_diff_x_y, axis=1)

    # Sum along rows and compute suared root to get distance
    dist_x_y = np.sqrt(weights)

    # Get the indices of terminals which are at a distance less than r
    available_terminals = list(np.where(dist_x_y < r)[0])

    # Return the list of terminal IDs
    return available_terminals, np.exp(-weights[available_terminals])
