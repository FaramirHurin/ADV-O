import os
import pandas as pd 
import numpy as np
from scipy.stats import rankdata
from scipy.stats import f
import matplotlib.pyplot as plt
import seaborn as sns
import time
import math

from utils.orange_library import *

RESULTS_FOLDER = "results/"

def perform_friedman_nemenyi_test(filename, dataframe, alpha = 0.05):
    
    nrows = dataframe.shape[0]
    ncols = dataframe.shape[1]
    
    #perform ranking of the results
    ranking = pd.DataFrame(dataframe)
    for i in range(nrows):
        ranking.iloc[i,:] = rankdata(dataframe.iloc[i,:])
    
    #compute average of rankings
    avranks = [0]*ncols
    for i in range(ncols):
        avranks[i] = ranking.iloc[:,i].mean() 
    
    #compute Friedman and Iman tests, with respective degrees of freedom
    (f_stat, dof), (im, fdistdof) = compute_friedman(avranks, nrows)

    df1 = fdistdof[0]
    df2 = fdistdof[1]

    critical_value = f.ppf(q = 1 - alpha, dfn = df1, dfd = df2)

    if (im > critical_value):
        print("H0 Rejected, proceeding")
        
        cd = compute_CD(avranks, nrows, str(alpha))
        path = RESULTS_FOLDER+filename+"_CD.png"
        graph_ranks(avranks, dataframe.columns, cd=cd, width=6, textspace=1, filename=path, bbox_inches="tight")
        
    else:
        print("Cannot reject H0, stop.")



if __name__ == "__main__":

    # RESULTS PREPROCESSING 

    # for all files in results folder load the one that starts with "all_metrics"
    results_list = []
    for filename in os.listdir(RESULTS_FOLDER):
        if filename.startswith("all_metrics"):
            dataframe = pd.read_csv(os.path.join(RESULTS_FOLDER, filename))
            results_list.append(dataframe)

    
    # create a dictionary where each key is a dataset name and the value is the corresponding dataframe
    dataframes_dict = {}
    for i, df in enumerate(results_list):
        dataset_name = f"dataset{i+1}"
        dataframes_dict[dataset_name] = df

    # create a new dataframe where each row is a method, each column is a dataset, and the values are just the last metric
    methods = dataframe.columns
    result_df = pd.DataFrame(columns=dataframes_dict.keys(), index=methods)

    for dataset_name, df in dataframes_dict.items():
        #TODO: call it 'PRAUC' in the results rather than picking the first row here
        result_df[dataset_name] = df.iloc[1,:]

    #TODO: remove random noise when we have the real results
    result_df = result_df.applymap(lambda x: x + np.random.rand() / 10)
    result_df = result_df.T

    print(result_df)
    
    ## FRIEDMAN TEST
    perform_friedman_nemenyi_test("friedman_test", result_df)