#import packages
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

#import csv
ccdata = np.loadtxt("creditcarddata.csv",  delimiter = ",", skiprows = 1)

#establish domain and range
DomainVal = ccdata[:, 1: 24]
RangeVal = ccdata[:, 24: 25]
 
# Create training and test split
X_train, X_test, y_train, y_test = train_test_split(DomainVal, RangeVal, test_size=0.25)

#turn to 1D array
y_train = y_train.flatten()
y_test = y_test.flatten()

#Scaling it improves processing speed, reducing runtime
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# Instantiate the Support Vector Classifier (SVC)
svc = SVC(gamma = 0.001, C = 1.0)
 
# Fit the model
svc.fit(X_train_std, y_train)

# Make the predictions
y_prediction = svc.predict(X_test_std)

#percent of predictions that match the labelled data
percent_correct = roc_auc_score(y_test, y_prediction)

#conditional statement for output
if (percent_correct >= 0.85):
    print("SVM - Credit Card Prediction" + "\n" + "Accuracy is: " + str(round(percent_correct * 100, 2)) + "%" + "\n" + "We should invest in the company")
else:
    print("SVM - Credit Card Prediction" + "\n" + "Accuracy is: " + str(round(percent_correct * 100, 2)) + "%" + "\n" + "We should not invest in the company")

