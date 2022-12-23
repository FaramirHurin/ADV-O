import pandas as pd
import numpy as np

def feature_engineering(generator, transactions_df):
        #TODO: NO DELAY for now
        #TODO: I believe they look into the future: discuss this 
        transactions_df["transaction_id"] = transactions_df.index
        transactions_df["tx_datetime"] = pd.to_datetime(transactions_df.tx_day * 86400 + transactions_df.tx_time, unit="s", origin=generator.start_date)
        transactions_df["is_weekend"] = transactions_df.tx_datetime.dt.dayofweek.isin([5, 6])
        transactions_df["during_night"] = transactions_df.tx_datetime.dt.hour.isin([0, 1, 2, 3, 4, 5, 6, 7, 8, 22, 23])
        transactions_df["hour"] = transactions_df["tx_datetime"].apply(lambda x: x.hour)
        transactions_df["day_of_week"] = transactions_df["tx_datetime"].apply(lambda x: x.dayofweek)
        transactions_df["month"] = transactions_df["tx_datetime"].apply(lambda x: x.month)
        
        transactions_df["distance_to_terminal"] = transactions_df.apply(lambda x: np.sqrt((generator.customers[x.customer_id].x - generator.terminals[x.terminal_id].x )**2 + (generator.customers[x.customer_id].y - generator.terminals[x.terminal_id].y )**2), axis=1)
        transactions_df["distance_to_terminal"] = transactions_df["distance_to_terminal"].fillna(0)

        time_periods = [1, 7, 30]
        groups = ['customer_id', 'terminal_id']
        calculations = ['nb_tx', 'nb_frauds', 'stats']

        for calculation in calculations:
            for group in groups:
                for time_period in time_periods:
                    if calculation == 'nb_tx':
                        df_rolling = transactions_df.groupby([group, 'tx_day'])[group].count().rolling(time_period).sum().rename(f'{group}_{calculation}_{time_period}days').reset_index()
                        transactions_df = transactions_df.merge(df_rolling, on=[group, 'tx_day'])
                        transactions_df[f'{group}_{calculation}_{time_period}days'].fillna(0, inplace=True)
                    elif calculation == 'stats':
                        df_rolling_min = transactions_df.groupby([group, 'tx_day'])['amount'].min().rolling(time_period).min().reset_index().rename(columns={'amount': f'{group}_min_tx_{time_period}days'})
                        df_rolling_max = transactions_df.groupby([group, 'tx_day'])['amount'].max().rolling(time_period).max().reset_index().rename(columns={'amount': f'{group}_max_tx_{time_period}days'})
                        df_rolling_mean = transactions_df.groupby([group, 'tx_day'])['amount'].mean().rolling(time_period).mean().reset_index().rename(columns={'amount': f'{group}_mean_tx_{time_period}days'})
                        df_rolling_std = transactions_df.groupby([group, 'tx_day'])['amount'].std().rolling(time_period).std().reset_index().rename(columns={'amount': f'{group}_std_tx_{time_period}days'})
                        df_rolling_median = transactions_df.groupby([group, 'tx_day'])['amount'].median().rolling(time_period).std().reset_index().rename(columns={'amount': f'{group}_median_tx_{time_period}days'})

                        transactions_df = transactions_df.merge(df_rolling_min, on=[group, 'tx_day'])
                        transactions_df = transactions_df.merge(df_rolling_max, on=[group, 'tx_day'])
                        transactions_df = transactions_df.merge(df_rolling_mean, on=[group, 'tx_day'])
                        transactions_df = transactions_df.merge(df_rolling_std, on=[group, 'tx_day'])
                        transactions_df = transactions_df.merge(df_rolling_median, on=[group, 'tx_day'])

                        transactions_df[f'{group}_min_tx_{time_period}days'].fillna(0, inplace=True)
                        transactions_df[f'{group}_max_tx_{time_period}days'].fillna(0, inplace=True)
                        transactions_df[f'{group}_mean_tx_{time_period}days'].fillna(0, inplace=True)
                        transactions_df[f'{group}_std_tx_{time_period}days'].fillna(0, inplace=True)
                        transactions_df[f'{group}_median_tx_{time_period}days'].fillna(0, inplace=True)
                    elif calculation == 'nb_frauds' and group == 'terminal_id':
                        df_rolling = transactions_df[transactions_df['is_fraud'] == 1].groupby([group, 'tx_day'])[group].count().rolling(time_period).sum().rename(f'{group}_{calculation}_{time_period}days').reset_index()
                        transactions_df = transactions_df.merge(df_rolling, on=[group, 'tx_day'])
                        transactions_df[f'{group}_{calculation}_{time_period}days'].fillna(0, inplace=True)


        df_grouped = transactions_df.groupby(['customer_id', 'terminal_id'])
        transactions_df['customer_nb_prev_tx_same_terminal'] = df_grouped['transaction_id'].transform(lambda x: x.shift(1).notnull().sum())
        transactions_df['customer_nb_prev_tx_same_terminal'].fillna(0, inplace=True)


        # Number of transactions per day: You could create a new column that represents the average number of transactions that the customer makes per day over a specific time period (e.g., past 7 days, past month, past year). This could be useful if you suspect that customers who make a large number of transactions per day are more likely to be fraudulent.

        # Terminal type: You could create a new column that indicates the type of terminal where the transaction took place (e.g., ATM, point-of-sale terminal, online). This could be useful if you suspect that certain types of terminals are more likely to be used for fraudulent transactions.

        return transactions_df           