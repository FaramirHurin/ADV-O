

# N users
# T terminals
# D days
# p compromission probability
# NB transactions per day
# tau max days


# Total number of frauds F = D * N * (1 - (1 - p)^D) * NB * tau
# Total number of genuines G = N * NB * np.sum([(1 - p)^d for d in D])
# Desired ratio of frauds = F / G = D * tau *  (1 - (1 - p)^D) /  np.sum([(1 - p)^d for d in D]
# Average number of frauds per compromised card  COMPR = tau * NB

# First constraint, we want to keep COMPR below a certain threshold to avoid divergence, but we can use different values
# Hence, we can select various couples of tau and NB.
# Then, we can select different desired percentage of stolen cards, and find the necessary combination p-D.
# We can therefore run simulations playing with tau, NB, p and D, even though only 3 are independent.
# Finally, we can try runnin the setting with equal, similar and different initial distributions of fraudsters anf genuine.
# We can also split over the time, BUT we must consider that the first week will have a different % of frauds than the last.
# In any case, we have enough simulations to mak the empirical tests, we only need to profile where the generator is slow,
# and select the parameters to minimize the runtime.

# Let us fix various desired ratios of frauds, number of days,







import logging
from ADVO.generator import Generator


logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG)

# Create a generator object
generator = Generator(random_state=42)

logging.debug('Operations starting')
# Generate 100 terminals, 500 customers, and transaction data for 8 days
generator.generate_terminals(n_terminals=200)
generator.generate_customers(n_customers=2000, radius=20, max_days_from_compromission=4, compromission_probability=0.05, avg_tx_per_day = [0,2], trx_std_amt = 2)
generator.generate_transactions(nb_days_to_generate=60, start_date="2022-01-01")
logging.debug('End operations')

# Access the generated data
terminals_df = generator.get_terminals_df()
customers_df = generator.get_customers_df()
transactions_df = generator.get_transactions_df()

print( transactions_df.shape[0], sum(transactions_df['is_fraud']),
                       sum(transactions_df['is_fraud']) / (sum(transactions_df['is_fraud'])+transactions_df.shape[0]) )