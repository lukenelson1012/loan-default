
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline


def report(pipeline: Pipeline, X_test, y_test):
    print("Classsification report:")
    print(classification_report(pipeline.predict(X_test.values.reshape(-1, 18)), y_test))