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

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .25)
# Select several features to use. Can't use all of them bc many are categorical
# and have many categories.

# Run and Cross-validate model:
# Stategy: cross-validate lots of combinations of hyperparameters using
# exhaustive grid search. Hyperparameter combos are stored in param_grid. The
# measure that cross-validate spits out is accuracy score, because that is the
# most relevant measure for identifying who has which income.

param_grid = [
  {'C': [.001, .01, .1, 1, 10], 'kernel': ['linear']},
  {'C': [.01, .1, 1, 10], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
 ]

svm1 = SVC()
svm1_grid = GridSearchCV(SVC(), param_grid, cv = 3, scoring = 'accuracy',
    verbose = 10)
svm1_grid.fit(x_train, y_train)
print svm1_grid.best_params_





