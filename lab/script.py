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
    

    def clear_outliers(self, cols): # cols can a single value or a list
        for col in cols:
            if col == "ap_lo":
                self._df = self._df[(self._df[col] >= 0) & (self._df[col] <= 200)]
            if col == "ap_hi":
                self._df = self._df[(self._df[col] >= 0) & (self._df[col] <= 250)]
            q1 = self._df[col].quantile(0.25)
            q3 = self._df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            self._df = self._df[(self._df[col] >= lower) & (self._df[col] <= upper)]
        return self


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


class Visualisation():


    def __init__(self):
        pass


    def EDA(df):
        rows = 2
        cols = 4
        fig, axes = plt.subplots(rows, cols, figsize=(22, 12))
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


    def boxplot_comparison(df):
        fig, ax = plt.subplots(1, 2, figsize=(8, 5))
        sns.boxplot(data=df, y="ap_hi", ax=ax[0])
        sns.boxplot(data=df, y="ap_lo", ax=ax[1])
        ax[0].set_title("Systolic Blood Pressure\n", fontweight="bold")
        ax[1].set_title("Diastolic Blood Pressure\n", fontweight="bold")
        ax[0].grid(axis='y')
        ax[1].grid(axis='y')
        ax[0].set_ylabel("mmHg")
        ax[1].set_ylabel("")
        plt.show()


    def risk_factors(df):
        """Haven't tried this yet."""
        rows = 2
        cols = 3
        fig, axes = plt.subplots(rows, cols, figsize=(17, 12))
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
            lambda ax: sns.countplot(data=df, x=(df["age"] // 365.25).astype(int), hue="cardio", ax=ax, palette=cardio_palette, zorder=2), # TODO: fix the x-axis labelling overlap
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
                        ax.set_xticks([0, 1, 2])
                        ax.set_xticklabels(labels=chol_gluc_labels)
                    if c == 5:
                        ax.set_xticks([0, 1])
                        ax.set_xticklabels(labels=binary_labels)
                    ax.set_title(titles[c], fontweight="bold")
                    if c == 2:
                        ax.tick_params(axis="x", rotation=90)
                    else:
                        ax.tick_params(axis="x", rotation=45)
                    ax.set_xlabel("")
                    ax.set_ylabel("Number of Patients")
                    ax.legend(title="", labels=["healthy", "sick"])
                    ax.grid(axis="y", zorder=1)
                    c += 1
        fig.suptitle("\nRisk Factors for Cardiovascular Disease\n", fontweight="bold", fontsize=20)
        plt.subplots_adjust(hspace=0.4, wspace=0.3)
        plt.show()


    def correlation_matrix(df):
        df_corr = df.copy()
        bmi_cat_dict = {
            "underweight": 1,
            "normal": 2,
            "overweight": 3,
            "obese cl1": 4,
            "obese cl2": 5,
            "obese cl3": 6
        }
        ap_cat_dict = {
            "hypotension": 1, 
            "normal": 2, 
            "elevated": 3, 
            "hypertension st1": 4, 
            "hypertension st2": 5, 
            "hypertension crisis": 6
        }
        df_corr["bmi_cat"] = df_corr["bmi_cat"].map(bmi_cat_dict)
        df_corr["ap_cat"] = df_corr["ap_cat"].map(ap_cat_dict)
        plt.figure(figsize=(10, 7), dpi=75)
        sns.heatmap(df_corr.corr(), annot=True, vmin=-1, vmax=1, fmt='.2f', cmap='RdBu')
        plt.title("\nCorrelation Matrix\n", fontweight="bold", fontsize=17)
        plt.show()


class Modelling():


    def __init__(self, df):
        self._X = df.drop(columns=["cardio"])
        self._y = df["cardio"]
        self._X_train = None
        self._X_val = None
        self._X_test = None
        self._y_train = None
        self._y_val = None
        self._y_test = None
        self._scores = {}
        self._param_grids = {
            "log_reg": {
                "model": LogisticRegression(),
                "params": {
                    "C": [
                        0.001,
                        0.01,
                        0.1,
                        1,
                        10,
                        100,
                        1000
                    ],
                    "penalty": ["l1", "l2", "elasticnet", None], 
                    "solver": ["saga", "liblinear", "lbfgs"], 
                    "max_iter": [10000],
                }
            },
            "sgd": {
                "model": SGDClassifier(),
                "params": {
                    "alpha": [
                        0.001, 
                        0.01, 
                        0.1, 
                        1,
                        10,
                        100,
                        1000
                    ],
                    "loss": ["log_loss", "hinge", "modified_huber", "perceptron"],
                    "penalty": ["l1", "l2", "elasticnet", None],
                    "learning_rate": ["optimal", "invscaling", "adaptive"],
                    "max_iter": [10000],
                }
            },
            "svm": {
                "model": SVC(),
                "params": {
                    "C": [3, 6, 9, 20],
                    "gamma": ["scale", "auto"],
                    "degree": [2, 3],
                    "kernel": ["sigmoid", "poly", "rbf"],
                }
            },
            "knn": {
                "model": KNeighborsClassifier(),
                "params": {
                    "n_neighbors": [4, 5, 6, 10],
                    "weights": ["uniform", "distance"], 
                    "algorithm": ["auto", "kd_tree", "ball_tree", "brute"], 
                    "leaf_size": [1, 2],
                    "p": [1, 2],
                }
            },
            "rforest": {
                "model": RandomForestClassifier(),
                "params": {
                    "n_estimators": [8, 12, 20, 30], 
                    "criterion": ["gini", "entropy"],
                    "max_depth": [None, 2, 10], 
                    "min_samples_split": [20, 25, 30],
                }
            }
        }


    @property
    def scores(self):
        return self._scores
    
    
    def split_data(self):
        self._X_train, self._X_test, self._y_train, self._y_test = train_test_split(self._X, self._y, test_size=0.3, random_state=1123)
        self._X_val, self._X_test, self._y_val, self._y_test = train_test_split(self._X_test, self._y_test, test_size=0.5, random_state=1123)
        return self
    

    def _standardiser(self):
        scaler = StandardScaler()
        self._X_train = scaler.fit_transform(self._X_train)
        self._X_val = scaler.transform(self._X_val)
        self._X_test = scaler.transform(self._X_test)
        return self
    

    def _normaliser(self):
        scaler = MinMaxScaler()
        self._X_train = scaler.fit_transform(self._X_train)
        self._X_val = scaler.transform(self._X_val)
        self._X_test = scaler.transform(self._X_test)
        return self


    def scale_data(self):
        self._standardiser()
        self._normaliser()
        return self


    def tuning(self):
        for key, values in self._param_grids.items():
            print(f"\nTraining {key} ...")
            model = values["model"]
            params = values["params"]
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=params,
                cv=5,
                scoring="recall",
                verbose=2, # remove before final run
                n_jobs=-1
            )
            grid_search.fit(self._X_train, self._y_train)
            self._scores[key] = {
                "best_params": grid_search.best_params_,
                "train_score": grid_search.score(self._X_train, self._y_train),
                "val_score": grid_search.score(self._X_val, self._y_val)
            }
        return self


    def ensemble(self, X, y):
        pass


    def evaluation(self, X, y):
        pass


def main():
    """Run it all."""
    pass


if __name__ == "__main__":
    temp = main()