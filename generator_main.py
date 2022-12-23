from ADVO.generator import Generator, feature_engineering

if __name__ == "__main__":

    generator = Generator(random_state = 55)
    generator.generate_terminals(n_terminals = 100)
    generator.generate_customers(n_customers = 500)
    generator.generate_transactions(nb_days_to_generate = 8, start_date = "2018-04-01")
    
    transactions_df = generator.get_transactions_df()

    transactions_df_eng = feature_engineering(generator, transactions_df)

    print(transactions_df_eng.head())
       