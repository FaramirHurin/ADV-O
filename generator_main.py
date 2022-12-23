from ADVO.generator import Generator, feature_engineering

if __name__ == "__main__":

    generator = Generator(random_state = 55)
    generator.generate_terminals(n_terminals = 150)
    generator.generate_customers(n_customers = 200)
    generator.generate_transactions(nb_days_to_generate = 8, start_date = "2022-12-17")
    

    transactions_df = generator.get_transactions_df()
    terminals_df = generator.get_terminals_df()
    customers_df = generator.get_customers_df()

    transactions_df_eng = feature_engineering(generator, transactions_df)

    print(terminals_df.head())
    print(customers_df.head())
    print(transactions_df.head())
    print(transactions_df_eng.sample(frac=1)[150:155].T)
       