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

> An Adversary model of fraudsters’ behaviour to improve oversampling in credit card fraud detection <br>
This is the repository for the code of the paper "An Adversary model of fraudsters’ behaviour to improve oversampling in credit card fraud detection" by Daniele Lunghi, Gian Marco Paldino, Olivier Caelen, and Gianluca Bontempi. <br>
The code in this repository is intended to make the experiments in the paper reproducible. <br>
The repository is expected to be extended in the future. <br>


### 🏠 [Homepage](https://gmpal.github.io/ADV-O/)


## 🖥️ Install and Usage

```sh
git clone https://github.com/FaramirHurin/ADV-O.git
cd ADV-O
pip install -r requirements.txt
python main.py
```

### 🦾 Demo

The code will execute the experiments on synthetic data that have been included in the paper. <br>
The output of the code will provide Table 6, 7, 8 of the paper. <br>
N.B. CTGAN is disabled by default, because it requires a specific Python version, pytorch, and makes the experiments slower.<br>
It can be added by uncommenting the corresponding lines. <br>
```sh
Regressor:  MLPRegressor(max_iter=2000, random_state=42)
             x_terminal_id  y_terminal_id  TX_AMOUNT
score             0.852514       0.585098   0.938087
naive_score       0.392595       0.540160   0.906686
Regressor:  Ridge(random_state=42)
             x_terminal_id  y_terminal_id  TX_AMOUNT
score             0.848494       0.579815   0.925387
naive_score       0.392595       0.540160   0.906686
Regressor:  RandomForestRegressor(random_state=42)
             x_terminal_id  y_terminal_id  TX_AMOUNT
score             0.845694       0.585861   0.903093
naive_score       0.392595       0.540160   0.906686
Best regressor:  MLPRegressor(max_iter=2000, random_state=42)
             x_terminal_id  y_terminal_id  TX_AMOUNT
SMOTE             0.109430       0.100991   0.183195
Random            0.046015       0.105073   0.020810
KMeansSMOTE       0.045930       0.104494   0.021049
ADVO              0.086780       0.116524   0.033137
            Baseline  Baseline_balanced     SMOTE    Random  KMeansSMOTE      ADVO
PRAUC       0.321631           0.374427  0.358808  0.368367     0.357233  0.367125
PRAUC_Card  0.454525           0.500294  0.464808  0.494833     0.480953  0.479956
Precision   0.336567           0.227430  0.268898  0.262045     0.252546  0.267644
Recall      0.290874           0.888746  0.681393  0.719123     0.727830  0.694615
F1 score    0.312057           0.362179  0.385619  0.384119     0.374979  0.386402
PK50        0.760000           0.360000  0.560000  0.300000     0.400000  0.420000
PK100       0.780000           0.370000  0.520000  0.380000     0.390000  0.450000
PK200       0.740000           0.375000  0.505000  0.435000     0.365000  0.545000
PK500       0.612000           0.398000  0.502000  0.404000     0.400000  0.554000
PK1000      0.475000           0.421000  0.462000  0.439000     0.404000  0.485000
PK2000      0.359000           0.383500  0.397000  0.390500     0.378000  0.406500
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

***
_This README was generated with ❤️ by [readme-md-generator](https://github.com/kefranabg/readme-md-generator)_
