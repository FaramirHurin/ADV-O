import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Circle


def feature_engineering(generator, transactions_df, decimals=2):

        transactions_df = transactions_df.copy()
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
        
        x_terminal =  transactions_df['terminal_id'].apply(lambda x: generator.terminals[x].x)
        y_terminal =  transactions_df['terminal_id'].apply(lambda x: generator.terminals[x].y)
        transactions_df['tx_angle'] = np.arctan2(y_terminal, x_terminal).apply(lambda x: np.rad2deg(x))

        # Create a new column called "direction" that indicates the direction of the transaction based
        # on the angle between the terminal and the origin
        transactions_df['tx_location'] = np.where((transactions_df['tx_angle'] > 0) & (transactions_df['tx_angle'] <= 22.5), 'North',
                                                np.where((transactions_df['tx_angle'] > 22.5) & (transactions_df['tx_angle'] <= 67.5), 'North-East',
                                                np.where((transactions_df['tx_angle'] > 67.5) & (transactions_df['tx_angle'] <= 112.5), 'East',
                                                np.where((transactions_df['tx_angle'] > 112.5) & (transactions_df['tx_angle'] <= 157.5), 'South-East',
                                                np.where((transactions_df['tx_angle'] > 157.5) & (transactions_df['tx_angle'] <= 180) | (transactions_df['tx_angle'] < 0) & (transactions_df['tx_angle'] >= -22.5), 'South',
                                                np.where((transactions_df['tx_angle'] < -22.5) & (transactions_df['tx_angle'] >= -67.5), 'South-West',
                                                np.where((transactions_df['tx_angle'] < -67.5) & (transactions_df['tx_angle'] >= -112.5), 'West',
                                                np.where((transactions_df['tx_angle'] < -112.5) & (transactions_df['tx_angle'] >= -157.5), 'North-West', 'Other'))))))))

        transactions_df = transactions_df.infer_objects()
        transactions_df.loc[:,transactions_df.select_dtypes(['float64']).columns] = transactions_df.select_dtypes(['float64']).round(decimals=decimals)
        return transactions_df           


class GeneratedDataPlotter():

    def __init__(self, generator):
        self.generator = generator
        self.terminals_df, self.customers_df, self.transactions_df = generator.get_dataframes()


    def plot_terminals_and_customers(self, figsize=(10, 10), alpha=0.02):
        fig, ax = plt.subplots(figsize=figsize)
        ax.scatter(self.terminals_df['x_terminal'], self.terminals_df['y_terminal'], color='blue', label='Terminal')
        ax.scatter(self.customers_df['x_customer'], self.customers_df['y_customer'], color='red', label='Customer')
        for i, row in self.customers_df.iterrows():
            ax.add_patch(Circle((row['x_customer'], row['y_customer']), self.generator.customers[row['customer_id']].radius, color='green', alpha=alpha))
        ax.legend(loc = 'upper left')
        ax.set_xlim([0, 100])
        ax.set_ylim([0, 100])
        plt.show()

    def plot_num_transactions_per_day(self,  figsize=(10, 10)):
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(self.transactions_df.groupby('tx_day')['amount'].count(), label='Number of transactions')
        ax.set_ylabel('Number of transactions')
        ax.set_xlabel('Day')
        ax.legend()
        plt.show()

    def plot_avg_amount_per_day(self,  figsize=(10, 10)):
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(self.transactions_df.groupby('tx_day').mean()['amount'], label='Average amount per transaction')
        ax.set_ylabel('Average amount per transaction')
        ax.set_xlabel('Day')
        ax.legend()
        plt.show()

    def plot_total_amount_per_day(self,  figsize=(10, 10)):
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(self.transactions_df.groupby('tx_day').sum()['amount'], label='Total amount spent')
        ax.set_ylabel('Total amount spent')
        ax.set_xlabel('Day')
        ax.legend()
        plt.show()

    def plot_coordinate_history(self, customer_id, genuine_color='g', fraud_color='r', arrow_color='k'):
        customer = self.generator.customers[customer_id]
        plt.plot(customer.x_history['genuine'], customer.y_history['genuine'], f'{genuine_color}.')
        plt.plot(customer.x_history['fraud'], customer.y_history['fraud'], f'{fraud_color}.')
        plt.xlabel('X coordinate')
        plt.ylabel('Y coordinate')
        
        for i in range(len(customer.x_history['genuine']) - 1):
            plt.arrow(customer.x_history['genuine'][i], customer.y_history['genuine'][i], 
                    customer.x_history['genuine'][i+1] - customer.x_history['genuine'][i], 
                    customer.y_history['genuine'][i+1] - customer.y_history['genuine'][i],
                    color=arrow_color, width=0.01, head_width=0.5, head_length=2, length_includes_head=True)
        for i in range(len(customer.x_history['fraud']) - 1):
            plt.arrow(customer.x_history['fraud'][i], customer.y_history['fraud'][i], 
                    customer.x_history['fraud'][i+1] - customer.x_history['fraud'][i], 
                    customer.y_history['fraud'][i+1] - customer.y_history['fraud'][i],
                    color=arrow_color, width=0.01, head_width=0.5, head_length=2, length_includes_head=True)
        
        plt.show()
    
