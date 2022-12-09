import unittest
import numpy as np
import pandas as pd
from generator import generate_terminal_profiles_table

class TestGenerateTerminalProfilesTable(unittest.TestCase):

    def test_returns_dataframe(self):
        # Test that the function returns a DataFrame
        result = generate_terminal_profiles_table(10)
        self.assertIsInstance(result, pd.DataFrame)

    def test_dataframe_dimensions(self):
        # Test that the DataFrame has the correct number of rows and columns
        result = generate_terminal_profiles_table(10)
        self.assertEqual(result.shape, (10, 3))

    def test_dataframe_column_names(self):
        # Test that the DataFrame has the expected column names
        result = generate_terminal_profiles_table(10)
        self.assertEqual(list(result.columns), ['TERMINAL_ID', 'x_terminal_id', 'y_terminal_id'])

    def test_terminal_id_uniqueness(self):
        # Test that the 'TERMINAL_ID' column contains unique values
        result = generate_terminal_profiles_table(10)
        self.assertEqual(len(result['TERMINAL_ID'].unique()), 10)

    def test_x_y_distributions(self):
        # Test that the 'x_terminal_id' and 'y_terminal_id' columns contain values from uniform distributions
        result = generate_terminal_profiles_table(100)
        x_values = result['x_terminal_id'].values
        y_values = result['y_terminal_id'].values
        self.assertTrue(np.isclose(np.mean(x_values), 50, atol=10))
        self.assertTrue(np.isclose(np.mean(y_values), 50, atol=10))
