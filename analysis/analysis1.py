#This script runs an SVM on the income data.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.learning_curve import learning_curve
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import train_test_split

# PREPARE DATA:
file ='/Users/ilya/metis/week4/metis_project_3/analysis/clean_data.csv'
data = pd.read_csv(file, header = 0)
data_dummied = pd.get_dummies(data)
data_dummied = data_dummied.drop(data_dummied.columns[-2], axis = 1)

data_train, data_test = train_test_split(data_dummied)

x_train = data_train.ix[:,:-1]
y_train = data_train[data_train.columns[-1]]
x_test = data_test.ix[:,:-1]
y_test = data_test[data_test.columns[-1]]

# RUN MODEL:
svm = SVC()
svm.fit(x_train, y_train)


# EVALUATE MODEL: