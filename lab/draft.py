import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, root_mean_squared_error


class CardioPreprocessor:

    def __init__(self):
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None


    def load_data():

        df = pd.read_csv("../data/cardio.csv", sep=";")
        df = df.drop(columns=["id"])
        schema = pd.DataFrame({
            "age": "days",
            "height": "cm",
            "weight": "kg",
            "ap_hi": "systolic blood pressure (mmHg)",
            "ap_lo": "diastolic blood pressure (mmHg)",
            "cholesterol": "1: normal - 2: above normal - 3: well above normal",
            "gluc": "1: normal - 2: above normal - 3: well above normal",
            "smoke": "binary",
            "alco": "binary",
            "active": "binary",
            "cardio": "binary"
        }, index=[0]).T
        numeric_cols = ["age", "height", "weight", "ap_hi", "ap_lo"]
        categoric_cols = ["gender", "cholesterol", "gluc", "smoke", "alco", "active", "cardio"]
        return df, schema, numeric_cols, categoric_cols


    def clean_data(self):
        pass


class CardioEDA:

    def __init__(self):
        pass


    def countplots(df, cat_cols):

        fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(10, 5))
        axes = axes.flatten()
        for ax, col in zip(axes, cat_cols):
            sns.countplot(data=df, x=col, ax=ax)
            ax.set_title(f"\n{col}\n", fontweight="bold")
            ax.set_xlabel(None)
            ax.set_ylabel(None)
        fig.delaxes(axes[-1])
        plt.tight_layout()
        plt.show()


    def boxplots(df, num_cols):

        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(5, 7))
        axes = axes.flatten()
        for ax, col in zip(axes, num_cols):
            sns.boxplot(data=df, y=col, ax=ax)
            ax.set_title(f"\n{col}\n", fontweight="bold")
            ax.set_xlabel(None)
            ax.set_ylabel(None)
        fig.delaxes(axes[-1])
        plt.tight_layout()
        plt.show()


class CardioFunctions:

    def __init__(self):
        pass


    def get_data(self):
        pass


    def clean_data(self):
        pass


    def split_data(self):
        pass


    def train_model(self):
        pass


    def evaluate_model(self):
        pass


    def save_model(self):
        pass


    def load_model(self):
        pass


    def predict(self):
        pass


    def run(self):
        self.get_data()
        self.clean_data()
        self.split_data()
        self.train_model()
        self.evaluate_model()
        self.save_model()
        self.load_model()
        self.predict()