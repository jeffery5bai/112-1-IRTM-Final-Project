import pandas as pd

from sklearn.naive_bayes import ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, recall_score, precision_score, ConfusionMatrixDisplay, fbeta_score, RocCurveDisplay
import matplotlib.pyplot as plt

def result_report(model, test_x, test_y, pred_y):
    result_accuracy = accuracy_score(test_y, pred_y)
    result_recall = recall_score(test_y, pred_y)
    result_precision = precision_score(test_y, pred_y)
    result_f1score = f1_score(test_y, pred_y)
    result_f5score = fbeta_score(test_y, pred_y, beta=0.5)
    result_f3score = fbeta_score(test_y, pred_y, beta=0.3)
    result_cm = confusion_matrix(test_y, pred_y)

    print(f"Accuracy: {round(result_accuracy, 2)}")
    print(f"Recall: {round(result_recall, 2)}")
    print(f"Precision: {round(result_precision, 2)}")

    print(f"F1-Score: {round(result_f1score, 2)}")
    print(f"F0.5-Score: {round(result_f5score, 2)}")
    print(f"F0.3-Score: {round(result_f3score, 2)}")
    # print(f"Confusion Matrix:\n {result_cm}")

    disp = ConfusionMatrixDisplay(confusion_matrix=result_cm, display_labels=model.classes_)
    disp.plot()
    plt.show()

    RocCurveDisplay.from_estimator(model, test_x, test_y)
    plt.show()