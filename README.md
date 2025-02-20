<h1 align="center">Welcome to ADV-O 👋</h1>
<p>
  <img alt="Version" src="https://img.shields.io/badge/version-0.2.0-blue.svg?cacheSeconds=2592000" />
  <a href="docs" target="_blank">
    <img alt="Documentation" src="https://img.shields.io/badge/documentation-yes-brightgreen.svg" />
  </a>
  <a href="https://opensource.org/licenses/Apache-2.0" target="_blank">
    <img alt="License: Apache License, Version 2.0" src="https://img.shields.io/badge/license-Apache%202-blue" /
  </a>
  <a href="https://github.com/FaramirHurin/ADV-O" target="_blank">
    <img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/FaramirHurin/ADV-O?style=social">
  </a>
   

  
<!--
  <a href="https://codecov.io/gh/kefranabg/readme-md-generator">
    <img src="https://codecov.io/gh/kefranabg/readme-md-generator/branch/master/graph/badge.svg" />
  </a>
-->
</p>

> An adversary model of fraudsters’ behaviour to improve oversampling in credit card fraud detection <br>
This is the repository for the code of the paper "An Adversary model of fraudsters’ behaviour to improve oversampling in credit card fraud detection" by Daniele Lunghi, Gian Marco Paldino, Olivier Caelen, and Gianluca Bontempi. <br>
This repository is intended to make the experiments in the paper reproducible. <br>
The repository is expected to be extended in the future. <br>


## Visit our 🏠 [Homepage](https://FaramirHurin.github.io/ADV-O/) for the documentation!

Tested on Python 3.10
This work is based on the [transaction data simulator](https://fraud-detection-handbook.github.io/fraud-detection-handbook/Chapter_3_GettingStarted/SimulatedDataset.html) described in the [Fraud Detection Handbook](https://fraud-detection-handbook.github.io/fraud-detection-handbook/Foreword.html) by [Le Borgne et al.](https://fraud-detection-handbook.github.io/fraud-detection-handbook/Foreword.html#authors)

If you use this work, you should cite: 
```
@ARTICLE{10332176,
  author={Lunghi, Daniele and Paldino, Gian Marco and Caelen, Olivier and Bontempi, Gianluca},
  journal={IEEE Access}, 
  title={An Adversary Model of Fraudsters’ Behavior to Improve Oversampling in Credit Card Fraud Detection}, 
  year={2023},
  volume={11},
  number={},
  pages={136666-136679},
  keywords={Fraud;Behavioral sciences;Credit cards;Time series analysis;Classification algorithms;Training;Prediction algorithms;Fraud detection;imbalance learning;machine learning;oversampling;synthetic data;time series;threat model},
  doi={10.1109/ACCESS.2023.3337635}}

@book{leborgne2022fraud,
title={Reproducible Machine Learning for Credit Card Fraud Detection - Practical Handbook},
author={Le Borgne, Yann-A{\"e}l and Siblini, Wissam and Lebichot, Bertrand and Bontempi, Gianluca},
url={https://github.com/Fraud-Detection-Handbook/fraud-detection-handbook},
year={2022},
publisher={Universit{\'e} Libre de Bruxelles}
}
```
  
## 🖥️ Install and Usage

```sh
git clone https://github.com/FaramirHurin/ADV-O.git
cd ADV-O
pip install -r requirements.txt
python main_synthetic.py
```

### 🦾 Demo

The code will execute the experiments on synthetic data that have been included in the paper. <br>
The output of the code will provide Table 6, 7, 8 of the paper. <br>
N.B. CTGAN is disabled by default, because it requires a specific Python version, pytorch, and makes the experiments slower.<br>
It can be added by uncommenting the corresponding lines. <br>
```sh
Table 6: Synthetic data: R2 scores for the predicted features for various regressors.
                                              x_terminal_id  y_terminal_id  TX_AMOUNT
MLPRegressor(max_iter=2000, random_state=42)           0.85           0.59       0.94
Ridge(random_state=42)                                 0.85           0.58       0.93
RandomForestRegressor(random_state=42)                 0.85           0.59       0.90
Naive                                                  0.39           0.54       0.91


Table 7: Synthetic data: accuracy of oversampling algorithms. All oversampling algorithms have been tested using a Balanced Random Forest. No oversampling has been tested with a classic Random Forest ('Baseline'),  and a Balanced Random Forest ('Baseline balanced').
            Baseline  Baseline_balanced  SMOTE  Random  KMeansSMOTE  ADVO
PRAUC           0.32               0.37   0.36    0.37         0.36  0.37
PRAUC_Card      0.45               0.50   0.46    0.49         0.48  0.48
Precision       0.34               0.23   0.27    0.26         0.25  0.27
Recall          0.29               0.89   0.68    0.72         0.73  0.69
F1 score        0.31               0.36   0.39    0.38         0.37  0.39
PK50            0.76               0.36   0.56    0.30         0.40  0.42
PK100           0.78               0.37   0.52    0.38         0.39  0.45
PK200           0.74               0.38   0.50    0.44         0.36  0.55
PK500           0.61               0.40   0.50    0.40         0.40  0.55
PK1000          0.48               0.42   0.46    0.44         0.40  0.48
PK2000          0.36               0.38   0.40    0.39         0.38  0.41


Table 8: Synthetic data: AUC of absolute differences between kde
             x_terminal_id  y_terminal_id  TX_AMOUNT
SMOTE                 0.11           0.10       0.18
Random                0.05           0.11       0.02
KMeansSMOTE           0.05           0.10       0.02
ADVO                  0.09           0.12       0.03
```
## 🧠 The Generator
The generator simulates genuine and fraudulent transactions and is based on a customer-terminal-transaction structure, where a group of customers selects a set of customers to perform various transactions. A two-dimensional vector, represented as a location, characterizes each terminal. A location also describes each customer.
Customers then iteratively choose among the terminals close to them they use to generate transactions. Terminals must be within a max distance from the customer, and their probability of being selected is higher the closer they are to the customer. The amount of each transaction is independently drawn from a Normal distribution, whose variance and mean depend only on the encoded habits of the user. 
Then, as simulation time goes by, a portion of cardholders switches from the genuine to the fraudster category. 
We represent the compromission of a user as an abrupt change in her location and spending habits, which are drawn from a different multivariate distribution representing the fraudsters population.
We then model the dependency between any two consecutive frauds as a change in the features of the fraudster performing them, where the new location and spending habits are a nondeterministic function of the transaction just conducted
  
## 💡 General Idea
The code begins by setting some constants, such as `SAMPLE_STRATEGY`, `N_JOBS`, `N_TREES`, `N_USERS`, `N_TERMINALS`, and `RANDOM_STATE`, which are used later in the code. The `RANDOM_GRID_*` variables define sets of hyperparameters that will be used to train and evaluate machine learning models using cross-validation. The `CANDIDATE_REGRESSORS` list specifies three machine learning models that will be trained and evaluated: a multi-layer perceptron regressor, a Ridge regressor, and a random forest regressor. The `CANDIDATE_GRIDS` list specifies the sets of hyperparameters that will be used for each of these models.

It begins by loading the generated data. It then splits the data into a training set and a test set using the `train_test_split` function from sklearn.model_selection. The code then creates instances of several over-sampling techniques, including `SMOTE`, `RandomOverSampler`, and `KMeansSMOTE`, `CTGAN` (disabled by default), and `ADVO`, the proposed methodology. The code also creates instances of two ensemble classifiers: `BalancedRandomForestClassifier` and `RandomForestClassifier`.

For `ADVO`, it also trains and evaluates the machine learning models specified in the `CANDIDATE_REGRESSORS` list, using the hyperparameter grids specified in `CANDIDATE_GRIDS`. 

The code then iterates over the over-sampling techniques and the ensemble classifiers, training and evaluating each combination on the training data.

The code uses the `evaluate_models` function to evaluate the trained models and compute various metrics, including `AUC` (area under the curve), precision, and recall.

Finally, the code uses the `compute_kde_difference_auc` function to compute the AUC for the difference between the kernel density estimates of the predicted probabilities for the fraudulent and non-fraudulent classes, and the `fraud_metrics` function to compute additional metrics for evaluating the performance of the fraud detection models.



## ✅ Run tests

```sh
python -m unittest
```

## 👩🏻‍💻 Authors

👤 **Daniele Lunghi**

* Website: [ResearchGate](https://www.researchgate.net/profile/Daniele-Lunghi)
* Github: [@FaramirHurin](https://github.com/FaramirHurin)
* LinkedIn: [@daniele-lunghi-7b06b91a2](https://linkedin.com/in/daniele-lunghi-7b06b91a2)

👤 **Gian Marco Paldino**

* Website: [ResearchGate](https://www.researchgate.net/profile/Gian-Marco-Paldino-2)
* Github: [@gmpal](https://github.com/gmpal)
* LinkedIn: [@gianmarcopaldino](https://linkedin.com/in/gianmarcopaldino)


## 🤝 Contributing

Contributions, issues and feature requests are welcome!<br />Feel free to check [issues page](https://github.com/FaramirHurin/ADV-O/issues). You can also take a look at the [contributing guide](https://github.com/FaramirHurin/ADV-O/blob/main/contributing.rst).

## Show your support

Give a ⭐️ if this project helped you!

## 📝 License

Copyright © 2022 [Daniele Lunghi](https://github.com/FaramirHurin).<br />
This project is [Apache License, Version 2.0](https://opensource.org/licenses/Apache-2.0) licensed.
