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

`generate_customers(n_customers=500, radius=20, max_days_from_compromission=3, compromission_probability=0.03)`: generates a specified number of customer objects and stores them in the customers attribute of the Generator object. The customer locations are randomly generated. The radius parameter specifies the maximum distance that a customer is willing to travel to use a terminal. The max_days_from_compromission parameter specifies the maximum number of days a fraudster will continue to use a compromised card after it has been compromised. The compromission_probability parameter specifies the probability that a customer card is stolen i.e. the customer becomes a fraudster.

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
                                                   299                  503                  280                  478
tx_time                                          83557                44733                37698                31721
tx_day                                               5                    7                    5                    7
customer_id                                         55                   60                    5                   59
terminal_id                                         45                  106                   59                   63
amount                                           42.09                 7.86                119.4                70.67
is_fraud                                          True                 True                False                False
transaction_id                                     944                 1037                   76                 1026
tx_datetime                        2022-12-22 23:12:37  2022-12-24 12:25:33  2022-12-22 10:28:18  2022-12-24 08:48:41
is_weekend                                       False                 True                False                 True
during_night                                      True                False                False                 True
hour                                                23                   12                   10                    8
day_of_week                                          3                    5                    3                    5
month                                               12                   12                   12                   12
distance_to_terminal                             13.48                73.76                 7.27                11.19
customer_id_nb_tx_1days                            3.0                  3.0                  3.0                  6.0
customer_id_nb_tx_7days                           19.0                 19.0                 16.0                 25.0
customer_id_nb_tx_30days                          81.0                 76.0                 75.0                 78.0
terminal_id_nb_tx_1days                           32.0                  6.0                 11.0                 29.0
terminal_id_nb_tx_7days                          113.0                 41.0                 33.0                131.0
terminal_id_nb_tx_30days                         251.0                137.0                126.0                291.0
terminal_id_nb_frauds_1days                       19.0                  1.0                  6.0                  6.0
terminal_id_nb_frauds_7days                       71.0                  8.0                 53.0                 41.0
terminal_id_nb_frauds_30days                       0.0                178.0                  0.0                181.0
customer_id_min_tx_1days                         42.09                 0.25                 23.7                  0.0
customer_id_max_tx_1days                         97.82                 7.86                119.4               114.13
customer_id_mean_tx_1days                        69.59                 4.97                 61.7                52.09
customer_id_std_tx_1days                           0.0                  0.0                  0.0                  0.0
customer_id_median_tx_1days                        0.0                  0.0                  0.0                  0.0
customer_id_min_tx_7days                          9.04                  0.0                  1.1                  0.0
customer_id_max_tx_7days                        145.25               114.13                119.4               114.13
customer_id_mean_tx_7days                        67.81                51.52                 48.5                52.97
customer_id_std_tx_7days                           0.0                  0.0                14.31                  0.0
customer_id_median_tx_7days                      40.73                28.47                 18.2                26.21
customer_id_min_tx_30days                          0.0                  0.0                  0.0                  0.0
customer_id_max_tx_30days                       165.62               196.09                  0.0               196.09
customer_id_mean_tx_30days                       54.32                44.29                  0.0                45.82
customer_id_std_tx_30days                          0.0                  0.0                  0.0                  0.0
customer_id_median_tx_30days                     42.75                35.67                  0.0                35.07
terminal_id_min_tx_1days                          0.58                 7.86                 2.55                  0.0
terminal_id_max_tx_1days                        142.88                75.14                192.5               157.86
terminal_id_mean_tx_1days                        35.94                56.56                45.34                67.08
terminal_id_std_tx_1days                           0.0                  0.0                  0.0                  0.0
terminal_id_median_tx_1days                        0.0                  0.0                  0.0                  0.0
terminal_id_min_tx_7days                           0.0                 1.58                  0.0                  0.0
terminal_id_max_tx_7days                        175.84               148.03               196.09               231.84
terminal_id_mean_tx_7days                        34.18                55.68                33.45                50.51
terminal_id_std_tx_7days                         11.01                  0.0                  0.0                 8.89
terminal_id_median_tx_7days                       14.0                30.38                 9.64                15.01
terminal_id_min_tx_30days                          0.0                  0.0                  0.0                  0.0
terminal_id_max_tx_30days                          0.0               231.84                  0.0               231.84
terminal_id_mean_tx_30days                         0.0                 42.4                  0.0                40.57
terminal_id_std_tx_30days                          0.0                  0.0                  0.0                  0.0
terminal_id_median_tx_30days                       0.0                23.77                  0.0                20.06
customer_nb_prev_tx_same_terminal                    8                    0                   10                   11
tx_angle                                         43.95                15.91                 49.0                46.96
tx_location                                 North-East                North           North-East           North-East
```
