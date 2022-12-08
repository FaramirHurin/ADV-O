import numpy as np
import pandas as pd 

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

