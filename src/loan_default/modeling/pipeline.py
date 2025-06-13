
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

import numpy as np


def pipeline(X_train, X_test, y_train, y_test):

    num_preprocessor = Pipeline([
        ('nan', SimpleImputer(missing_values = np.nan, strategy="mean")),
        ('num', StandardScaler()),
    ])

    cat_preprocessor = Pipeline([
        ('nan', SimpleImputer(missing_values = np.nan, strategy="most_frequent")),
        ('ohe', OneHotEncoder(handle_unknown='ignore')),
    ])

    preprocessor = ColumnTransformer([
        ('cat', cat_preprocessor, [0, 4, 16]),
        ('num', num_preprocessor, [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]),
    ])


    pipe = make_pipeline(preprocessor, LogisticRegression())

    pipe.fit(X_train.values.reshape(-1, X_train.shape[1]), y_train)

    print(f"Accuracy score: {accuracy_score(pipe.predict(X_test.values.reshape(-1, X_train.shape[1])), y_test):.2f}")

    return pipe