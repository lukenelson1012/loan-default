
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from loan_default.etl.extract import read_data
from loan_default.modeling.pipeline import pipeline
from loan_default.modeling.evaluation import report


def drop_unused_data(df):
        df.drop(index=1, inplace=True)
        df.reset_index()
        df.drop(columns=["member_id", "id", "zip_code", "next_pymnt_d"], inplace=True)

def revol_util_preprocessing(df):
        df["revol_util"] = df["revol_util"].str.rstrip("%")
        df.loc[df["revol_util"] == "�0.00%�"] = 0
        df.drop(index=2, inplace=True)
        df["revol_util"] = df["revol_util"].astype(float)
        df.rename(columns={"revol_util":"revol_util_%"}, inplace=True)

def term_preprocessing(df):
        int_month_split = df["term"].str.rstrip(" months")
        df["term"] = int_month_split
        df["term"].astype("float")
        df.rename(columns={"term":"term_months"}, inplace=True)

def emp_length_preprocessing(df):
        df["emp_length"] = df["emp_length"].str.rstrip(" years")
        df["emp_length"] = df["emp_length"].str.rstrip(" year")
        df["emp_length"] = df["emp_length"].str.rstrip("+")
        df.loc[df["emp_length"] == "< 1", "emp_length"] = 0
        df["emp_length"].value_counts()
        df["emp_length"].astype("float")

def loan_status_preprocessing(df):
        df["loan_status"] = df["loan_status"].str.lstrip("Does not meet the credit policy. Status:")
        df.loc[df["loan_status"] == "fault", "loan_status"] = "Default"

def purpose_preprocessing(df):
        int_loc = df.loc[df["purpose"]==0, "purpose"].index
        df.drop(int_loc, inplace=True)
        df["purpose"].astype("str")

def main():
        # Load data; see extract.py
        df = read_data()

        # Pre-processing steps
        drop_unused_data(df)
        revol_util_preprocessing(df)
        term_preprocessing(df)
        emp_length_preprocessing(df)
        loan_status_preprocessing(df)
        purpose_preprocessing(df)

        # Train-test split
        y = df["repay_fail"]
        X = df[["purpose", "loan_amnt", "term_months", "emp_length", "home_ownership", 
                "delinq_2yrs", "total_pymnt", "int_rate", "funded_amnt", "installment", 
                "funded_amnt", "annual_inc", "dti", "open_acc", "revol_util_%", "total_acc",
                "last_pymnt_amnt", "verification_status"]]
        

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

        # Pipeline and cross validation
        final_pipeline = pipeline(X_train, X_test, y_train, y_test)
        report(final_pipeline, X_test, y_test)


main()