from ctgan import CTGAN
import pandas as pd 

class CTGANOverSampler():
    def __init__(self, sampling_strategy, epochs=10, cuda=True):
        self.sampling_strategy = sampling_strategy
        self.epochs = epochs
        self.cuda = cuda

    def fit_resample(self, train_X, train_Y):
        df = pd.concat([train_X, train_Y], axis=1)
        frauds_df = df.loc[df['TX_FRAUD']==1]
        num_frauds = frauds_df.shape[0]
        num_synthetic_frauds = int((self.sampling_strategy)*df.shape[0]) - num_frauds
        ctgan = CTGAN(epochs=self.epochs, cuda=self.cuda)
        ctgan.fit(frauds_df)
        synthetic_data = ctgan.sample(num_synthetic_frauds)

        print(synthetic_data['TX_FRAUD'].value_counts())
        augmented_df = pd.concat([synthetic_data, df], axis=0)

        return augmented_df.drop(columns=['TX_FRAUD']), augmented_df['TX_FRAUD']

