import unittest
import random 
import numpy as np 
import pandas as pd
import pickle
from .functions import *

class TestGenerator(unittest.TestCase):
    def setUp(self):
        # Initialize a Generator instance with default parameters
        self.generator = Generator()
    
    def test_initialize(self):
        # Test that the terminal and customer profiles are correctly generated
        self.assertEqual(self.generator.terminal_profiles_table.shape[0], self.generator.n_terminals)
        self.assertEqual(self.generator.customer_profiles_table.shape[0], self.generator.n_customers)
        
        # Test that the available terminals for each customer are correctly calculated
        for i, row in self.generator.customer_profiles_table.iterrows():
            x, y = row['x_customer_id'], row['y_customer_id']
            available_terminals = row['available_terminals']
            for terminal in available_terminals:
                terminal_x, terminal_y = self.generator.x_y_terminals[terminal]
                distance = np.sqrt((x - terminal_x) ** 2 + (y - terminal_y) ** 2)
                self.assertLessEqual(distance, self.generator.radius)
    
    def test_generate(self):
        # Test that the generate method generates a transactions table with the correct number of rows
        transactions_table = self.generator.generate()
        self.assertEqual(transactions_table.shape[0], self.generator.n_customers * self.generator.nb_days)
        
        # Test that the transaction dates are within the correct date range
        min_date = pd.to_datetime(self.generator.start_date)
        max_date = min_date + pd.Timedelta(days=self.generator.nb_days)
        for date in transactions_table['transaction_date']:
            self.assertGreaterEqual(date, min_date)
            self.assertLessEqual(date, max_date)
    
    def test_export(self):
        # Test that the export method correctly exports the transactions table to a file
        self.generator.export('test_file', 'pickle')
        with open('test_file', 'rb') as f:
            data = pickle.load(f)
        self.assertEqual(data, self.generator.generate())
        
        # Test that an error is raised if an invalid file format is specified
        with self.assertRaises(ValueError):
            self.generator.export('test_file', 'invalid_format')
    
    def test__generate_transactions_table(self):
        # Test that the _generate_transactions_table method generates a table with the correct number of rows
        customer_profile = self.generator.customer_profiles_table.iloc[0]
        transactions_table = self.generator._generate_transactions_table(customer_profile)
        self.assertEqual(transactions_table.shape[0], self.generator.nb_days)
        
        # Test that the transaction dates are within the correct date range
        min_date = pd.to_datetime(self.generator.start_date)
        max_date = min_date + pd.Timedelta(days=self.generator.nb_days)
        for date in transactions_table['transaction_date']:
            self.assertGreaterEqual(date, min_date)
            self.assertLessEqual(date, max_date)
        
        # Test that the transaction amounts are within the correct range
        mean_amount = customer_profile['mean_amount']
        std_amount = customer_profile['std_amount']
        for amount in transactions_table['amount']:
            self.assertGreaterEqual(amount, mean_amount - 3 * std_amount)
            self.assertLessEqual(amount, mean_amount + 3 * std_amount)
        
        # Test that the transaction terminal is within the customer's available terminals
        available_terminals = customer_profile['available_terminals']
        for terminal in transactions_table['terminal_id']:
            self.assertIn(terminal, available_terminals)


    def test_generate(self):
        # Test that the generate method generates a transactions table with the correct number of rows
        transactions_table = self.generator.generate()
        self.assertEqual(transactions_table.shape[0], self.generator.n_customers * self.generator.nb_days)
        
        # Test that the transaction dates are within the correct date range
        min_date = pd.to_datetime(self.generator.start_date)
        max_date = min_date + pd.Timedelta(days=self.generator.nb_days)
        for date in transactions_table['transaction_date']:
            self.assertGreaterEqual(date, min_date)
            self.assertLessEqual(date, max_date)
        
        # Test that the transaction amounts are within the correct range for non-fraudster customers
        non_fraudsters = self.generator.customer_profiles_table[self.generator.customer_profiles_table['fraudster'] == False]
        for i, customer_profile in non_fraudsters.iterrows():
            mean_amount = customer_profile['mean_amount']
            std_amount = customer_profile['std_amount']
            customer_transactions = transactions_table[transactions_table['customer_id'] == i]
            for amount in customer_transactions['amount']:
                self.assertGreaterEqual(amount, mean_amount - 3 * std_amount)
                self.assertLessEqual(amount, mean_amount + 3 * std_amount)
        
        # Test that the transaction amounts are within the correct range for fraudster customers
        fraudsters = self.generator.customer_profiles_table[self.generator.customer_profiles_table['fraudster'] == True]
        for i, customer_profile in fraudsters.iterrows():
            mean_amount = self.generator.fraudsters_mean
            std_amount = np.sqrt(self.generator.fraudsters_var)
            customer_transactions = transactions_table[transactions_table['customer_id'] == i]
            for amount in customer_transactions['amount']:
                self.assertGreaterEqual(amount, mean_amount - 3 * std_amount)
                self.assertLessEqual(amount, mean_amount + 3 * std_amount)
        
        # Test that the transaction terminal is within the customer's available terminals
        for i, customer_profile in self.generator.customer_profiles_table.iterrows():
            available_terminals = customer_profile['available_terminals']
            customer_transactions = transactions_table[transactions_table['customer_id'] == i]
            for terminal in customer_transactions['terminal_id']:
                self.assertIn(terminal, available_terminals)

    def test_load(self):
        # Test that the load method correctly loads a transactions table from a file
        self.generator.export('test_file', 'pickle')
        loaded_generator = Generator.load('test_file')
        self.assertEqual(self.generator.generate(), loaded_generator.generate())
        
        # Test that an error is raised if the file format is invalid
        with self.assertRaises(ValueError):
            Generator.load('test_file', file_format='invalid_format')