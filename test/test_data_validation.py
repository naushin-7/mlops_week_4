import unittest
import pandas as pd
import os, sys 

class TestDataValidation(unittest.TestCase):
    @classmethod 
    def setUpClass(cls):    
        pathname = os.path.dirname(os.path.dirname(sys.argv[0]))
        path = os.path.abspath(pathname)
        data_dir = os.path.join('test_data','iris.csv')
        cls.csv_path = os.path.join(path,data_dir)

    # Check data shape
    def test_data_shape(self):
        df = pd.read_csv(self.csv_path)
        self.assertEqual(df.shape, (75, 5))

    # Check for missing values
    def test_no_missing_values(self):
        df = pd.read_csv(self.csv_path)
        self.assertFalse(df.isnull().values.any())

if __name__ == '__main__':
    unittest.main()
