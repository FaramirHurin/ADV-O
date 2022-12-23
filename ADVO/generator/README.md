# Welcome to ADVO-Generator

ADVO-Generator is a tool for generating synthetic data for a system of bank terminals and customers. The generated data includes terminal and customer locations, as well as customer transactions at the terminals.

## Initialization
To create a Generator object, you can use the following code:

```py
from ADVO.generator import Generator

generator = Generator(random_state=42)
```

The random_state argument allows you to specify a seed for the random number generator, so that the generated data will be the same each time the class is used with the same seed. If you do not specify a random_state, the generator will use a different seed each time it is run.

## Generating data
To generate data, you can use the following methods:

`generate_terminals(n_terminals=100)`: generates a specified number of terminal objects and stores them in the terminals attribute of the Generator object. The terminal locations are randomly generated.

`generate_customers(n_customers=500, radius=20, max_days_from_compromission=3, compromission_probability=0.03)`: generates a specified number of customer objects and stores them in the customers attribute of the Generator object. The customer locations are randomly generated. The radius parameter specifies the maximum distance that a customer is willing to travel to use a terminal. The max_days_from_compromission parameter specifies the maximum number of days a customer will continue to use a compromised terminal after it has been compromised. The compromission_probability parameter specifies the probability that a customer will compromise a terminal they use.

`generate_transactions(nb_days_to_generate=8, start_date="2018-04-01")`: generates transaction data for the customers over a specified number of days, starting from the specified start_date. The transactions are stored in the transactions attribute of the Generator object.

## Accessing data
To access the data that has been generated, you can use the following methods:

`get_terminals_df()`: returns a pandas DataFrame containing the data for all generated terminal objects.

`get_customers_df()`: returns a pandas DataFrame containing the data for all generated customer objects.

`get_transactions_df()`: returns a pandas DataFrame containing the data for all generated transactions.

## Basic usage
Here is an example of how you might use the `Generator` class to generate and access data:

```py
from ADVO.generator import Generator

# Create a generator object
generator = Generator(random_state=42)

# Generate 100 terminals, 500 customers, and transaction data for 8 days
generator.generate_terminals(n_terminals=100)
generator.generate_customers(n_customers=200, radius=20, max_days_from_compromission=3, compromission_probability=0.03)
generator.generate_transactions(nb_days_to_generate=8, start_date="2022-01-01")

# Access the generated data
terminals_df = generator.get_terminals_df()
customers_df = generator.get_customers_df()
transactions_df = generator.get_transactions_df()
```

A more compact way to obtain the same results as above (using the default parameters) is this one: 
```py
from ADVO.generator import Generator

# Create a generator object
generator = Generator(random_state=42)
# Generate 100 terminals, 500 customers, and transaction data for 8 days
generator.generate()
# Access the generated data
terminals_df, customers_df, transactions_df = generator.get_dataframes()
```

You can then use the DataFrames to analyze the generated data as needed. For example, you might plot the locations of the terminals and customers on a map, or create summary statistics for the transaction data. 
This is what `terminals_df` looks like.
```
   terminal_id  x_terminal  y_terminal
0            0        2.79       95.74
1            1       59.73       73.21
2            2       71.39       28.46
3            3       62.11       60.07
4            4       45.21       26.83
```
This is what `customers_df` looks like.
```
   customer_id  x_customer  y_customer  mean_amount  std_amount  mean_nb_tx_per_day  compromised
0            0       86.96        5.32        12.42        6.21                1.49        False
1            1       73.19       14.70        73.30       36.65                1.68        False
2            2       81.05       32.29        44.52       22.26                2.35        False
3            3       95.37       92.57        84.12       59.59                1.96        False
4            4       28.88       60.49        21.25       10.63                1.51        False
```
This is what `transactions_df` looks like.
```
   tx_time  tx_day  customer_id  terminal_id  amount  is_fraud
0    64714       1            0          110    9.68     False
1    55369       1            0          110    0.00     False
2    29417       4            0          115   11.97     False
3    45963       5            0          110   13.14     False
4    58945       5            0          115   18.67     False
```

## Feature engineering 
You can also run feature engineering on the `transaction_df` as follows 

```py
transactions_df_eng = feature_engineering(generator, transactions_df)
```

the resulting dataframe (shuffled and transposed for clarity), would look like this: 

```
                                                   115                  164                   79                  223                  435
tx_time                                          34313                31424                37519                62507                46308
tx_day                                               7                    7                    6                    5                    2
customer_id                                        131                   21                   44                  171                   57
terminal_id                                         43                   26                  148                   43                   43
amount                                             0.0                 27.4                15.27               404.59               123.15
is_fraud                                         False                False                False                 True                 True
transaction_id                                    2272                  319                  752                 2982                 1023
tx_datetime                        2018-04-08 09:31:53  2018-04-08 08:43:44  2018-04-07 10:25:19  2018-04-06 17:21:47  2018-04-03 12:51:48
is_weekend                                        True                 True                 True                False                False
during_night                                     False                 True                False                False                False
hour                                                 9                    8                   10                   17                   12
day_of_week                                          6                    6                    5                    4                    1
month                                                4                    4                    4                    4                    4
distance_to_terminal                              4.83                  1.7                 5.36                 2.23                 3.64
customer_id_nb_tx_1days                            3.0                  3.0                  5.0                  5.0                  1.0
customer_id_nb_tx_7days                           31.0                 21.0                 24.0                 31.0                 20.0
customer_id_nb_tx_30days                         114.0                 74.0                 88.0                105.0                 89.0
terminal_id_nb_tx_1days                           57.0                 12.0                  6.0                 43.0                 15.0
terminal_id_nb_tx_7days                          232.0                 47.0                 27.0                132.0                 32.0
terminal_id_nb_tx_30days                         332.0                290.0                167.0                246.0                162.0
terminal_id_nb_frauds_1days                       26.0                  7.0                  1.0                 17.0                 12.0
terminal_id_nb_frauds_7days                      122.0                 28.0                  7.0                 79.0                 30.0
terminal_id_nb_frauds_30days                     170.0                  0.0                 43.0                  0.0                  0.0
customer_id_min_tx_1days                           0.0                11.68                15.27               320.32               123.15
customer_id_max_tx_1days                        153.71                 27.4                53.67               427.86               123.15
customer_id_mean_tx_1days                        67.44                21.93                37.21               383.69               123.15
customer_id_std_tx_1days                           0.0                  0.0                  0.0                  0.0                  0.0
customer_id_median_tx_1days                        0.0                  0.0                  0.0                  0.0                  0.0
customer_id_min_tx_7days                           0.0                 3.64                  0.0                 0.65                  0.0
customer_id_max_tx_7days                         422.2                85.74               148.84               427.86               168.53
customer_id_mean_tx_7days                       142.75                 23.5                56.84               125.37                63.62
customer_id_std_tx_7days                         21.35                  0.0                  0.0                  0.0                  0.0
customer_id_median_tx_7days                      124.5                17.74                27.99               150.84                63.07
customer_id_min_tx_30days                          0.0                  0.0                  0.0                  0.0                  0.0
customer_id_max_tx_30days                        422.2                  0.0               148.84               427.86               168.53
customer_id_mean_tx_30days                       60.82                  0.0                34.06                52.68                42.45
customer_id_std_tx_30days                          0.0                  0.0                  0.0                  0.0                  0.0
customer_id_median_tx_30days                     75.77                  0.0                23.26                82.86                38.18
terminal_id_min_tx_1days                           0.0                 1.54                15.27                  0.0                 0.92
terminal_id_max_tx_1days                        205.13               192.57                53.67               427.86               256.42
terminal_id_mean_tx_1days                        57.42                32.02                33.94                90.52                53.69
terminal_id_std_tx_1days                           0.0                  0.0                  0.0                  0.0                  0.0
terminal_id_median_tx_1days                        0.0                  0.0                  0.0                  0.0                  0.0
terminal_id_min_tx_7days                           0.0                 1.54                  0.7                  0.0                 0.92
terminal_id_max_tx_7days                        567.66               563.87               115.71               427.86               563.87
terminal_id_mean_tx_7days                        71.39                53.97                46.12                73.39                63.34
terminal_id_std_tx_7days                         29.62                57.21                  0.0                39.94                52.39
terminal_id_median_tx_7days                      11.27                19.57                36.43                23.12                 29.3
terminal_id_min_tx_30days                          0.0                  0.0                  0.0                  0.0                  0.0
terminal_id_max_tx_30days                       567.66                  0.0               371.33                  0.0                  0.0
terminal_id_mean_tx_30days                       65.02                  0.0                49.57                  0.0                  0.0
terminal_id_std_tx_30days                          0.0                  0.0                  0.0                  0.0                  0.0
terminal_id_median_tx_30days                     40.13                  0.0                44.89                  0.0                  0.0
customer_nb_prev_tx_same_terminal                   26                   14                   18                   16                    2
tx_angle                                         43.93                44.91                37.89                43.93                43.93
tx_location                                 North-East           North-East           North-East           North-East           North-East
```