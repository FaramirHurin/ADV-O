from ADVO.generator import Generator, feature_engineering, GeneratedDataPlotter

if __name__ == "__main__":

    generator = Generator(random_state = 55)
    generator.generate_terminals(n_terminals = 200)
    generator.generate_customers(n_customers = 3, radius=50, max_days_from_compromission=3, compromission_probability=0.2)
    generator.generate_transactions(nb_days_to_generate = 10, start_date = "2022-12-17")

    generator.customers[0].plot_coordinate_history()
