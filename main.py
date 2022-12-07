from generator.generator import Generator


generator = Generator(n_customers=100, n_terminals=50)
generator.initialize()
transactions = generator.generate()

print(transactions)

