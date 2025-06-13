# Loan Default Prediction

Goal: Predict whether someone will default on their loans based on demographic, employment, and loan information
Result: The model is able to predict with 96% accuracy whether someone will default or not on their loans
* weighted average precision - 97%
* weighted average recall - 96%

Potential changes:
* Add more usable variables into the pipeline
* Use the string column variable names in pipeline.py instead of integer indices for simpler reading
* Adjust model appropriately to maximize recall for the "yes" classification (for less risk when later classifying loans)

The data for this project was downloaded from Kaggle. You can use the same dataset by going to the link below:
https://www.kaggle.com/datasets/joebeachcapital/loan-default

For any extra information about this project, please refer to the pyproject.toml file in the main branch.
