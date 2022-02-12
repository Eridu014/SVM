#import packages
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

#import csv
ccdata = np.loadtxt("creditcarddata.csv",  delimiter = ",", skiprows = 1)

#establish domain and range
DomainVal = ccdata[:, 1: 24]
RangeVal = ccdata[:, 24: 25]

#split the dataset into training and testing subets - 25% for testing
X_train, X_test, y_train, y_test = train_test_split(DomainVal, RangeVal, test_size = 0.25)

#turn to 1D array
y_train = y_train.flatten()
y_test = y_test.flatten()

#svm classifier
clf_svm = SVC(gamma = 0.001, C = 1.0)

#fit method to train the CLF
clf_svm.fit(X_train, y_train)

#predicted result on untrained(test) data
y_prediction = clf_svm.predict(X_test)

#percent of predictions that match the labelled data
percent_correct = roc_auc_score(y_test, y_prediction)

#conditional statement for output
if (percent_correct >= 0.85):
    print("SVM - Credit Card Prediction" + "\n" + "Accuracy is: " + str(round(percent_correct * 100, 2)) + "%" + "\n" + "We should invest in the company")
else:
    print("SVM - Credit Card Prediction" + "\n" + "Accuracy is: " + str(round(percent_correct * 100, 2)) + "%" + "\n" + "We should not invest in the company")

