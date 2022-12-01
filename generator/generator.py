class Generator():
    
    __slots__ = ("n_customers", "n_terminals", "radius" ,"random_state")
    
    def __init__(self, n_customers, n_terminals, radius, random_state = 2):
        self.n_customers = n_customers
        self.n_terminals = n_terminals
        self.radius = radius
        self.random_state = random_state
        
    def generate_customer_profiles_table(self):
        self.customer_profiles_table = functions.generate_customer_profiles_table(self.n_customers, self.random_state)
    
    def generate_terminal_profiles_table(self):
        self.terminal_profiles_table = functions.generate_terminal_profiles_table(self.n_terminals, self.random_state)
    
    
    
