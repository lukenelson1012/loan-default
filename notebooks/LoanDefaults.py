
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

from loan_default.etl.extract import read_data
from loan_default.modelling.pipeline import pipeline
from loan_default.modelling.evaluation import report


df = read_data()

df.drop(index=1, inplace=True)
df.reset_index()

df.drop(columns=["member_id", "id", "zip_code", "next_pymnt_d"], inplace=True)

# Data Cleaning

df["revol_util"] = df["revol_util"].str.rstrip("%")
df.loc[df["revol_util"] == "�0.00%�"] = 0
df.drop(index=2, inplace=True)
df["revol_util"] = df["revol_util"].astype(float)
df.rename(columns={"revol_util":"revol_util_%"}, inplace=True)

int_month_split = df["term"].str.rstrip(" months")
df["term"] = int_month_split
df["term"].astype("float")
df.rename(columns={"term":"term_months"}, inplace=True)

df["emp_length"] = df["emp_length"].str.rstrip(" years")
df["emp_length"] = df["emp_length"].str.rstrip(" year")
df["emp_length"] = df["emp_length"].str.rstrip("+")
df.loc[df["emp_length"] == "< 1", "emp_length"] = 0
df["emp_length"].value_counts()
df["emp_length"].astype("float")

df["loan_status"] = df["loan_status"].str.lstrip("Does not meet the credit policy. Status:")
df.loc[df["loan_status"] == "fault", "loan_status"] = "Default"

int_loc = df.loc[df["purpose"]==0, "purpose"].index
df.drop(int_loc, inplace=True)
df["purpose"].astype("str")

# Train_test_split

y = df["repay_fail"]
X = df[["purpose", "loan_amnt", "term_months", "emp_length", "home_ownership", 
        "delinq_2yrs", "total_pymnt", "int_rate", "funded_amnt", "installment", 
        "funded_amnt", "annual_inc", "dti", "open_acc", "revol_util_%", "total_acc",
        "last_pymnt_amnt", "verification_status"]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

final_pipeline = pipeline(X_train, X_test, y_train, y_test)
report(final_pipeline, X_test, y_test)