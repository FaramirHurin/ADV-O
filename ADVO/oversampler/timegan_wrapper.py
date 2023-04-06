from os import path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ydata_synthetic.synthesizers import ModelParameters
from ydata_synthetic.synthesizers.timeseries import TimeGAN



class TimeGANOverSampler():
    def __init__(self, sampling_strategy, epochs=10, seq_len=4, n_seq=3, hidden_dim=24, gamma=1, noise_dim = 32, dim = 128, batch_size = 32, log_step = 100, learning_rate = 5e-4):
        self.sampling_strategy = sampling_strategy
        self.epochs = epochs
        self.seq_len = seq_len
        self.n_seq = n_seq
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.noise_dim = noise_dim
        self.dim = dim
        self.batch_size = batch_size
        self.log_step = log_step
        self.learning_rate = learning_rate

    def _cut(self, data, seq_len):
        temp_data = []
        # Cut data by sequence length
        for i in range(0, len(data) - seq_len):
            _x = data[i:i + seq_len]
            temp_data.append(_x)
        return temp_data


    def fit_resample(self, train_X, train_Y, features):
        df = pd.concat([train_X, train_Y], axis=1)
        frauds_df = df.loc[df['TX_FRAUD']==1]
        groups = frauds_df.groupby("CUSTOMER_ID")

        # Loop through each group
        subseries_list = []
        for group_id, group_df in groups:
            if len(group_df) >= self.seq_len:
                subseries = self._cut(group_df[features].values, self.seq_len)
                subseries_list.extend(subseries)    
            else:
                continue

        gan_args = ModelParameters(batch_size=self.batch_size,
                                lr=self.learning_rate,
                                noise_dim=self.noise_dim,
                                layers_dim=self.dim)

        synth = TimeGAN(model_parameters=gan_args, hidden_dim=self.hidden_dim, seq_len=self.seq_len, n_seq=self.n_seq, gamma=self.gamma)
        synth.train(subseries_list, train_steps=self.epochs)
        synth.save('utils/synthesizer.pkl')

        num_frauds = frauds_df.shape[0]
        num_synthetic_frauds = int((self.sampling_strategy)*df.shape[0]) - num_frauds

        synth_data = synth.sample(num_synthetic_frauds//self.seq_len)

        synthetic_data = pd.DataFrame()
        for i in range(len(synth_data)):
            synthetic_data = synthetic_data.append(pd.DataFrame(synth_data[i]), ignore_index=True)

        synthetic_data.columns = features

        #TODO: check from this point on 
        synthetic_data['CUSTOMER_ID'] = np.random.randint(0, 100000, size=synthetic_data.shape[0])
        synthetic_data['TX_FRAUD'] = 1


        augmented_df = pd.concat([synthetic_data, df], axis=0)

        return augmented_df.drop(columns=['TX_FRAUD']), augmented_df['TX_FRAUD']

