import unittest
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score
import os,sys

class TestModelEvaluation(unittest.TestCase):
    def setUp(self):
        pathname = os.path.dirname(os.path.dirname(sys.argv[0]))
        path = os.path.abspath(pathname)
        data_dir = os.path.join('test_data','iris.csv')
        csv_path = os.path.join(path,data_dir)
        model_path = os.path.join(path,'model','model.pkl')
        # Load data and model before each test
        self.df = pd.read_csv(csv_path)
        self.X = self.df.drop(columns=["species"])
        self.y = self.df["species"]
        self.model = joblib.load(model_path)

    # UnitTest: check if the accuracy is more than 90%
    def test_accuracy_above_threshold(self):
        y_pred = self.model.predict(self.X)
        acc = accuracy_score(self.y, y_pred)
        self.assertGreater(acc, 0.9, f"Model accuracy < 0.9: {acc}")

if __name__ == '__main__':
    unittest.main()
