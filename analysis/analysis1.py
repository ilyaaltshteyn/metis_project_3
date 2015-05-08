# This script runs a bunch of classifiers on the income data. It varies the 
# classifiers' hyperparameters and tests their various combinations using cross 
# validation. Then it compares their predictions by plotting ROC curves.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style = 'white')
from sklearn.preprocessing import scale
from sklearn.metrics import accuracy_score
from sklearn.learning_curve import learning_curve
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFECV # Does feature selection w/cross-val

#                           ***PREPARE DATA***
file ='/Users/ilya/metis/week4/metis_project_3/analysis/clean_data.csv'
data = pd.read_csv(file, header = 0)

# Standardize continuous features so they're on equal scales:
numerical_columns = [x for x in data.columns if data[x].dtype == 'int64'][:-1]
for column in numerical_columns:
    if data[column].name == 'sex':
      continue
    data[column] = data[column].astype(float)
    data[column] = scale(data[column])

# Cut out 1/10th of data for faster cross-validation:
data = data.ix[:3999]

# Feature selection and adjustments:
# Drop features that are unlikely to have an effect on the outcome.
del data['marital_status']
del data['relationship']

data_dummied = pd.get_dummies(data)
data_dummied = data_dummied.drop(data_dummied.columns[-2], axis = 1)

x = data_dummied.ix[:,:-1]
y = data_dummied[data_dummied.columns[-1]]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .25)

# Do feature selection with recursive elimination of worst-performing features
# along with cross-validation. Do this for each of the models, to figure out
# the optimal combo of features for that model.

logistic = LogisticRegression(verbose = 10)
logistic_selector = RFECV(logistic, step = 1, cv = 3)
logistic_selector = logistic_selector.fit(x, y)
print logistic_selector.n_features_
print logistic_selector.support_
print logistic_selector.ranking_
print logistic_selector.grid_scores_
print logistic_selector.estimator_

def plot_feature_count_vs_crossvalscore(grid):
  x = range(1,len(grid) + 1)
  y = grid
  plt.plot(x, y)
  plt.title('Feature count vs cross validation score')
  plt.xlabel('Feature count')
  plt.ylabel('Cross validation score')
  plt.show()

plot_feature_count_vs_crossvalscore(logistic_selector.grid_scores_)



#                       ***Run and Cross-validate models***
# Stategy: cross-validate lots of combinations of hyperparameters using
# exhaustive grid search. Hyperparameter combos are stored in param_grid. The
# measure that cross-validate spits out is accuracy score, because that is the
# most relevant measure for identifying who has which income.

# Models and their hyperparams to vary:
# SVM - C, kernel, gamma (only for rbf kernel)
# KNN - n_neighbors, weight = ['uniform', 'distance'], leaf_size, p = [1,2]
# Decision tree - criterion = ['gini', 'entropy'], max_depth = [2,4,6,8,10,12], min_samples_split = [2,10,50], min_samples_leaf = [1,5,10]
# Random forest - n_estimators = [3,9,27], criterion = ['gini', 'entropy'], max_depth = [2,4,6,8], min_samples_split = [2,10,50], min_samples_leaf = [1,5,10], 
# Naive Bayes - 
# Logistic regression -

#### SVM
param_grid = [
  {'C': [.001, .01, .1, 1, 10], 'kernel': ['linear']},
  {'C': [.01, .1, 1, 10], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
 ]

svm_grid = GridSearchCV(SVC(), param_grid, cv = 3, scoring = 'accuracy', 
                        verbose = 10)
svm_grid.fit(x_train, y_train)
print svm_grid.best_params_
print svm_grid.best_score_

# The best SVM has C = .1, kernel = linear


# !!!!!! PLOT PREDICTIONS LATER!!!



#### KNN
param_grid = [{'n_neighbors': [1, 10, 100], 'weights' : ['distance', 'uniform'],
               'leaf_size' : [10, 100, 1000], 'p' : [1,2]}]

knn_grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv = 3,
                         scoring = 'accuracy', verbose = 10)
knn_grid.fit(x_train, y_train)
print knn_grid.best_params_
print knn_grid.best_score_

# The best KNN has n_neighbors = 100, weights = 'uniform', leaf_size = 100, p = 2





#### Decision tree
param_grid = [{
    'criterion' : ['gini', 'entropy'], 'max_depth' : [2,4,6,8,10,12], 
    'min_samples_split' : [2,10,50], 'min_samples_leaf' : [1,5,10]
}]

tree_grid = GridSearchCV(DecisionTreeClassifier(), param_grid, cv = 3,
                         scoring = 'accuracy', verbose = 10)
tree_grid.fit(x_train, y_train)
print tree_grid.best_params_
print tree_grid.best_score_

# The best decision tree has min_samples_split = 50, criterion = 'entropy', max_depth = 4, min_samples_leaf = 5








#### 







