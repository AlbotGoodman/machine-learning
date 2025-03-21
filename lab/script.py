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


class Processing():
    
    def __init__(self, path):
        self._df = pd.read_csv(path, sep=";")
        self._df = self._df.drop(columns=["id"])

    @property
    def df(self):
        return self._df

    def add_bmi(self):
        self._df["bmi"] = self._df["weight"] / (self._df["height"] / 100) ** 2
        self._df["bmi"] = self._df["bmi"].round(1)
        self._df = self._df[["age", "gender", "height", "weight", "bmi", "ap_hi", "ap_lo", "cholesterol", "gluc", "smoke", "alco", "active", "cardio"]]
        self._df = self._df[(self._df["bmi"] > 16) & (self._df["bmi"] <= 40)]
        # TODO: perhaps change the labels below to numbers instead
        bmi_labels = ["underweight", "normal", "overweight", "obese cl1", "obese cl2", "obese cl3"]
        self._df["bmi_cat"] = pd.cut(self._df["bmi"], bins=[16, 18.5, 25, 30, 35, 40, float("inf")], labels=bmi_labels)
        self._df = self._df[["age", "gender", "height", "weight", "bmi", "bmi_cat", "ap_hi", "ap_lo", "cholesterol", "gluc", "smoke", "alco", "active", "cardio"]]
        return self
    
    def clear_outliers(df): # df here will be a specific column
        q1 = df.quantile(0.25)
        q3 = df.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        df = df[(df >= lower) & (df <= upper)]
        return df

    def add_blood_pressure(self):
        bp_conditions = [
            (self._df["ap_hi"] <= 90) | (self._df["ap_lo"] <= 60), # with outlier I need | instead of & otherwise I get unknown values
            ((self._df["ap_hi"] >= 90) & (self._df["ap_hi"] < 120)) & (self._df["ap_lo"] < 80),
            ((self._df["ap_hi"] >= 120) & (self._df["ap_hi"] < 130)) & (self._df["ap_lo"] < 80),
            ((self._df["ap_hi"] >= 130) & (self._df["ap_hi"] < 140)) | ((self._df["ap_lo"] >= 80) & (self._df["ap_lo"] < 90)),
            ((self._df["ap_hi"] >= 140) & (self._df["ap_hi"] < 180)) | ((self._df["ap_lo"] >= 90) & (self._df["ap_lo"] < 120)),
            (self._df["ap_hi"] >= 180) | (self._df["ap_lo"] >= 120)
        ]
        bp_labels = [
            "hypotension", 
            "normal", 
            "elevated", 
            "hypertension st1", 
            "hypertension st2", 
            "hypertension crisis"
        ]
        self._df["ap_cat"] = np.select(bp_conditions, bp_labels, default="unknown")
        self._df = self._df[["age", "gender", "height", "weight", "bmi", "bmi_cat", "ap_hi", "ap_lo", "ap_cat", "cholesterol", "gluc", "smoke", "alco", "active", "cardio"]]
        return self

    def create_subsets(self, df):
        """df_a och df_b"""
        pass

    def split_data(self, df):
        pass

    def scale_data(self, df):
        pass


class Visualisation():

    def __init__(self):
        pass

    def EDA(df):
        rows = 2
        cols = 4
        fig, axes = plt.subplots(rows, cols, figsize=(25, 15))
        cholesterol_labels = ["normal", "above normal", "well above normal"]
        cholesterol_colours = ["forestgreen", "gold", "crimson"]
        cardio_palette = {0: "forestgreen", 1: "crimson"}

        # The addition of "lambda ax:" was from Copilot by highlighting the code and prompting:
        # "These plots won't show in the subplot and the pie plot is shown outside of the subplot."
        plot_functions = [
            lambda ax: sns.countplot(data=df, x="cardio", hue="cardio", palette=["forestgreen", "crimson"], ax=ax, legend=False, zorder=2),
            lambda ax: ax.pie(df["cholesterol"].value_counts(), labels=cholesterol_labels, colors=cholesterol_colours, autopct="%1.1f%%", startangle=180),
            lambda ax: sns.histplot(data=df, x=(df["age"] // 365.25).astype(int), hue="cardio", palette=cardio_palette, ax=ax, zorder=2),
            lambda ax: sns.countplot(data=df, x="smoke", hue="cardio", palette=cardio_palette, ax=ax, legend=False, zorder=2),
            lambda ax: sns.histplot(data=df, x="weight", hue="gender", bins=75, ax=ax, zorder=2),
            lambda ax: sns.histplot(data=df, x="height", hue="gender", bins=60, ax=ax, zorder=2),
            lambda ax: sns.countplot(data=df, x="cardio", hue="gender", ax=ax, zorder=2)
        ]
        titles = [
            "Cardiovascular Disease Distribution\n",
            "Cholesterol Distribution\n",
            "Age Distribution\n",
            "Smoking Distribution\n",
            "Weight Distribution\n",
            "Height Distribution\n",
            "Cardiovascular Disease by Gender\n"
        ]
        c = 0
        for row in range(rows):
            for col in range(cols):
                if c < len(plot_functions):
                    ax = axes[row, col]
                    plot_functions[c](ax)  # Copilot (as mentioned above)
                    if c == 0:
                        ax.set_xticks([0, 1])
                        ax.set_xticklabels(labels=["Healthy", "Sick"])
                    if c == 3:
                        ax.set_xticks([0, 1])
                        ax.set_xticklabels(labels=["Non-Smoker", "Smoker"])
                    ax.set_title(titles[c], fontweight="bold")
                    ax.set_xlabel("")
                    ax.set_ylabel("")
                    ax.grid(axis="y", zorder=1)
                    c += 1
                else:
                    fig.delaxes(axes[row, col])  # Remove unused subplot

        fig.suptitle("\nExploratory Data Analysis\n", fontweight="bold", fontsize=20)
        plt.subplots_adjust(hspace=0.2, wspace=0.3)
        plt.show()

    def boxplot_comparison(self):
        pass

    def risk_factors(df):
        """Haven't tried this yet."""
        rows = 2
        cols = 3
        fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
        chol_gluc_labels = ["normal", "above normal", "well above normal"]
        binary_labels = ["no", "yes"]
        cardio_palette = {0: "forestgreen", 1: "crimson"}
        bmi_labels = ["underweight", "normal", "overweight", "obese cl1", "obese cl2", "obese cl3"]
        bp_labels = ["hypotension", "normal", "elevated", "hypertension st1", "hypertension st2", "hypertension crisis"]

        # The addition of "lambda ax:" was from Copilot by highlighting the code and prompting:
        # "These plots won't show in the subplot and the pie plot is shown outside of the subplot."
        plot_functions = [
            lambda ax: sns.countplot(data=df, x="bmi_cat", order=bmi_labels, hue="cardio", ax=ax, palette=cardio_palette, zorder=2),
            lambda ax: sns.countplot(data=df, x="ap_cat", order=bp_labels, hue="cardio", ax=ax, palette=cardio_palette, zorder=2),
            lambda ax: sns.countplot(data=df, x=(df["age"] // 365.25).astype(int), hue="cardio", ax=ax, palette=cardio_palette, zorder=2),
            lambda ax: sns.countplot(data=df, x="cholesterol", hue="cardio", ax=ax, palette=cardio_palette, zorder=2),
            lambda ax: sns.countplot(data=df, x="gluc", hue="cardio", ax=ax, palette=cardio_palette, zorder=2),
            lambda ax: sns.countplot(data=df, x="active", hue="cardio", ax=ax, palette=cardio_palette, zorder=2)
        ]
        titles = [
            "Body Mass Index\n",
            "Blood Pressure\n",
            "Age\n",
            "Cholesterol Levels\n",
            "Glucose Levels\n",
            "Physically Active\n"
        ]
        c = 0
        for row in range(rows):
            for col in range(cols):
                if c < len(plot_functions):
                    ax = axes[row, col]
                    plot_functions[c](ax)  # Copilot (as mentioned above)
                    if c == 3 or c == 4:
                        ax.set_xticks([1, 2, 3])
                        ax.set_xticklabels(labels=chol_gluc_labels)
                    if c == 5:
                        ax.set_xticks([0, 1])
                        ax.set_xticklabels(labels=binary_labels)
                    ax.set_title(titles[c], fontweight="bold")
                    ax.set_xlabel("")
                    ax.set_ylabel("")
                    ax.grid(axis="y", zorder=1)
                    c += 1
                else:
                    fig.delaxes(axes[row, col])  # Remove unused subplot

        fig.suptitle("\nExploratory Data Analysis\n", fontweight="bold", fontsize=20)
        plt.subplots_adjust(hspace=0.2, wspace=0.3)
        plt.show()

    def correlation_matrix(self):
        pass


class Modelling():

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