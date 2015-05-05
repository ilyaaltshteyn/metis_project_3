# This script runs an SVM on the income data. It varies the SVM hyperparameters 
# and tests their various combinations using cross validation.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.learning_curve import learning_curve
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV

# PREPARE DATA:
file ='/Users/ilya/metis/week4/metis_project_3/analysis/clean_data.csv'
data = pd.read_csv(file, header = 0)

# Cut out 1/10th of data for faster cross-validation:
data = data.ix[:3999]

del data['marital_status']
del data['relationship']

data_dummied = pd.get_dummies(data)
data_dummied = data_dummied.drop(data_dummied.columns[-2], axis = 1)

x = data_dummied.ix[:,:-1]
y = data_dummied[data_dummied.columns[-1]]

# Select several features to use. Can't use all of them bc many are categorical
# and have many categories.

# Run and Cross-validate model:
# Stategy: cross-validate lots of combinations of hyperparameters using
# exhaustive grid search. Hyperparameter combos are stored in param_grid. The
# measure that cross-validate spits out is accuracy score, because that is the
# most relevant measure for identifying who has which income.

param_grid = [
  {'C': [.001, .01, .1, 1, 10, 100, 1000], 'kernel': ['linear']},
  {'C': [.01, .1, 1, 10, 100], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
 ]

svm1 = SVC()
svm1_accuracy = GridSearchCV(SVC(), param_grid, cv = 5, scoring = accuracy)

#Sklearn has a grid search function
svm1 = SVC(C = 100)
svm1 = np.mean(cross_val_score(svm1, x, y, cv = 5,
    scoring = "accuracy", verbose = 10))

svm2 = SVC(C = 10)
svm2_scores = np.mean(cross_val_score(svm2, x, y, cv = 5,
    scoring = "accuracy", verbose = 10))

svm3 = SVC(C = 1)
svm3_scores = np.mean(cross_val_score(svm3, x, y, cv = 5,
    scoring = "accuracy", verbose = 10))

svm4 = SVC(C = .1)
svm4_scores = np.mean(cross_val_score(svm3, x, y, cv = 5,
    scoring = "accuracy", verbose = 10))

svm5 = SVC(C = .01)
svm5_scores = np.mean(cross_val_score(svm3, x, y, cv = 5,
    scoring = "accuracy", verbose = 10))

svm6 = SVC(C = .001)
svm6_scores = np.mean(cross_val_score(svm3, x, y, cv = 5,
    scoring = "accuracy", verbose = 10))


# Strategy pt2-- Vary kernel and C statistic parameters to find optimal hyperparameters.
# Return cross-validated scores. The kernel is for transforming your data into
# other dimensions. The 




