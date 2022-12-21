Usage
======

Run
----
.. code-block:: bash

    git clone https://github.com/FaramirHurin/ADV-O.git
    cd ADV-O
    pip install -r requirements.txt
    python main.py

 
Output
------

.. code-block:: bash

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

