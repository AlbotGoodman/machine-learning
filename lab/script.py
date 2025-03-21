import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


class Processing(self):
    
    def __init__(self):
        pass

    def load_data(self, path):
        pass

    def add_bmi(self, df):
        pass

    def add_blood_pressure(self, df):
        pass

    def clear_outliers(self, df):
        pass

    def create_subsets(self, df):
        """df_a och df_b"""
        pass

    def split_data(self, df):
        pass

    def scale_data(self, df):
        pass


class Visualisation(self):

    def __init__(self):
        pass

    def EDA(self):
        """Alla EDA visuals här kanske?"""
        pass

    def boxplot_comparison(self):
        pass

    def risk_factors(self):
        pass

    def correlation_matrix(self):
        pass


class Models(self):

    def __init__(self):
        """Should param_grids be here?"""
        pass

    def tuning(self):
        pass

    def prediction(self):
        """Not sure I want/need this."""
        pass

    def ensemble(self):
        pass

    def evaluation(self):
        pass


def main():
    """Run it all."""
    pass


if __name__ == "__main__":
    temp = main()