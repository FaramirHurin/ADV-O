import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from typing import List, Tuple

def compute_kde_difference_auc(Xy: List[Tuple[pd.DataFrame, pd.DataFrame]],
                               columns: List[str],
                               names: List[str]) -> pd.DataFrame:
    """
    Compute the area under the curve (AUC) of the absolute difference between the kernel density estimates (KDE) of 
    multiple pandas DataFrames for the specified columns.

    Parameters:
    - Xy: A list of tuples, where each tuple consists of a pandas DataFrame (X) and a pandas DataFrame (y). The X 
          DataFrame should contain the columns for which the KDEs will be computed.
    - columns: A list of strings specifying the names of the columns for which the KDEs will be computed.
    - names: A list of strings to use as the row names in the resulting pd.DataFrame. The first two names will be 
             discarded, as they correspond to the X and y DataFrames.

    Returns:
    - A pandas DataFrame with the AUCs of the absolute difference between the KDEs of the specified columns in the X
      DataFrames. The rows are named using the elements of the 'names' parameter, starting from the third element. The
      columns are named using the elements of the 'columns' parameter.
    """


    trapzs = pd.DataFrame(columns=columns, index=names[2:])
    for column in columns:
            mins = [X[column].min() for X,_ in Xy]
            maxs = [X[column].max() for X,_ in Xy]
            grid = np.linspace(np.min(mins), np.max(maxs), 501) 

            kdes = [gaussian_kde(X[column]) for X,_ in Xy]
            aucs = np.zeros(len(Xy) - 1)
            
            baseline = kdes[0]
            for i in range(1, len(Xy)):
                kde_i = kdes[i]
                diff = np.abs(kde_i(grid) - baseline(grid))
                aucs[i-1] = np.trapz(diff, grid)

            trapzs[column] = aucs

    return trapzs