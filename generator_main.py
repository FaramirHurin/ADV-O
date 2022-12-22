from ADVO.generator import Generator

if __name__ == "__main__":

    generator = Generator(random_state = 55)
    generator.generate_terminals(n_terminals = 100)
    generator.generate_customers(n_customers = 500)
    generator.generate_transactions(nb_days_to_generate = 8, start_date = "2018-04-01")
    
    
    terminals_df = generator.get_terminals_df()
    customers_df = generator.get_customers_df()
    transactions_df = generator.get_transactions_df()

    print(transactions_df.head(10))
