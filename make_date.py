import pandas as pd
from sklearn.model_selection import train_test_split


class Data_wraper_for_log_reg:
    def __init__(self, path):

        self.path = path
        self.X = 0
        self.y = 0
        self.data = pd.read_csv(self.path)

    def split_for_train(self, scaler, cat_col_name, test_size, random_state):
        self.X = self.data.drop(cat_col_name)
        self.y = self.data(cat_col_name)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )
        return self.X_train, self.X_test, self.y_train, self.y_test

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
