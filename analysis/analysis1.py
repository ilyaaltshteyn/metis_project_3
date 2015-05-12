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
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import RFECV # Does feature selection w/cross-val
from sklearn.ensemble import ExtraTreesClassifier # Does feature selection w/trees
from sklearn.ensemble import RandomForestClassifier
import pylab as pl
from sklearn import svm, datasets
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, auc

#                          ***PREPARE DATA***
file ='/Users/ilya/metis/week4/metis_project_3/analysis/clean_data.csv'
data_all = pd.read_csv(file, header = 0)

# Standardize continuous features so they're on equal scales:
numerical_columns = [x for x in data_all.columns if data_all[x].dtype == 'int64'][:-1]
for column in numerical_columns:
    if data_all[column].name == 'sex':
      continue
    data_all[column] = data_all[column].astype(float)
    data_all[column] = scale(data_all[column])

# Cut out 1/10th of data for faster cross-validation:
data = data_all.ix[:3999]

#                         ***FEATURE SELECTION***
# Will do this separately for classifiers that return a coef_ weight and ones
# that don't, and won't do it at all for trees/forests because varying their
# depth in the grid search that I do later is equivalent to tossing features.

# Drop features that are unlikely to have an effect on the outcome.
data_dummied = pd.get_dummies(data)
data_dummied = data_dummied.drop(data_dummied.columns[-2], axis = 1)

x = data_dummied.ix[:,:-1]
y = data_dummied[data_dummied.columns[-1]]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .25)

# First, work with models that return a coef_ weight for each feature.
# Select features using recursive feature elimination (cross-validate model 
# with each feature taken out, only keep features that make the cross-val score 
# stronger):

def plot_feature_count_vs_crossvalscore(grid, name, features):
  x = range(1,len(grid) + 1)
  y = grid
  plt.plot(x, y, color = 'black', alpha = .6)
  plt.title('Feature count vs cross validation score for %s.\n\
            %s features will be left in.' % (name,features), size = 16)
  plt.xlabel('Feature count', size = 14)
  plt.ylabel('Cross validation score', size = 14)
  plt.xticks(size = 12)
  plt.yticks(size = 12)
  sns.despine()
  plt.show()
  
def cut_irrelevant_features(dataframe, support):
  for index, col in enumerate(dataframe.columns):
    if support[index] == False:
      dataframe = dataframe.drop(col, axis = 1)
  return dataframe

def cross_val_feature_drop(model, x=x_train, y=y_train, step = 1, cv = 5):
  """Takes a model class, step size, cross-val folds, x_train (features) matrix and
  y_train (outcome var) matrix and returns the optimal number of features, an array
  of true/false values that represent whether or not each column in x should
  be included in the model, and a grid of cross-validated model scores, the
  same length as the max number of features (columns of x), when each feature
  is included in the model."""

  mod = model
  mod_selector = RFECV(model, step=step, cv=cv)
  mod_selector = mod_selector.fit(x, y)
  nfeatures = mod_selector.n_features_
  support = mod_selector.support_
  grid = mod_selector.grid_scores_
  return nfeatures, support, grid

# Apply the functions defined above to models that return coef_ weights:

# LOGISTIC:
features, support_logistic, grid = cross_val_feature_drop( # I defined this function!
  model = LogisticRegression(verbose = 10))

plot_feature_count_vs_crossvalscore(grid, 'logistic regression classifier', features) # I defined this function!

# Redefine feature matrix to make it only include the best features:
x_logistic = cut_irrelevant_features(x, support_logistic)
x_logistic_train, x_logistic_test, y_logistic_train, y_logistic_test = \
  train_test_split(x_logistic, y, test_size = .25)

print "Logistic model will use %r features" % features

# SVM:
features, support_svm, grid = cross_val_feature_drop(SVC(kernel = 'linear', verbose = 10))

plot_feature_count_vs_crossvalscore(grid, 'SVM classifier with linear kernel', features) # I defined this function!

x_svm = cut_irrelevant_features(x, support_svm)
x_svm_train, x_svm_test, y_svm_train, y_svm_test = \
  train_test_split(x_svm, y, test_size = .25)


print "SVM will use %r features" % features

# Second, work with models that don't return a coef_ weight for each feature.
# Here, you can't eliminate features based on their weights. So let's use tree-based
# feature selection. Basically, you're figuring out how much each feature adds
# to a random forest and then figuring out feature importances to the tree and
# dropping unimportant features.

print x.shape
clf = ExtraTreesClassifier(n_estimators = 500)
selector = clf.fit(x_train,y_train)
x_tree_selected = selector.transform(x)
x_tree_selected_train, x_tree_selected_test, y_tree_selected_train, y_tree_selected_test = \
  train_test_split(x_tree_selected, y, test_size = .25)

print clf.feature_importances_
plt.bar(range(len(clf.feature_importances_)),clf.feature_importances_, 
        color = 'black', alpha = .6)
plt.title('Feature importances in a random forest used for feature \n\
          selection. %s features will be left in.' % x_tree_selected.shape[1],
          size = 16)
sns.despine()
plt.xlabel('Each bar is a feature', size = 14)
plt.ylabel('Feature importance: more is better', size = 14)
plt.yticks(size = 12)
plt.xticks(size = 12)
plt.show()
print "The tree selection leaves in %r features" % x_tree_selected.shape[1]


#                       ***Tune model hyperparameters***
# Stategy: cross-validate lots of combinations of hyperparameters using
# exhaustive grid search. Hyperparameter combos are stored in param_grid. The
# measure that cross-validate spits out is accuracy score, because that is the
# most relevant measure for identifying who has which income.

# Models and their hyperparams to vary:
# Logistic regression - penalty = ['l1', 'l2'], C = [.01, .1, 1, 10, 100], solver = [‘newton-cg’, ‘lbfgs’, ‘liblinear’]
# SVM - C, kernel, gamma (only for rbf kernel)
# KNN - n_neighbors, weight = ['uniform', 'distance'], leaf_size, p = [1,2]
# Decision tree - criterion = ['gini', 'entropy'], max_depth = [2,4,6,8,10,12], min_samples_split = [2,10,50], min_samples_leaf = [1,5,10]
# Random forest - n_estimators = [3,9,27], criterion = ['gini', 'entropy'], max_depth = [2,4,6,8], min_samples_split = [2,10,50], min_samples_leaf = [1,5,10], 
# Naive Bayes - nothing to vary.

#### Logistic regression
param_grid = [{'penalty' : ['l1', 'l2'], 'C' : [.01, .1, 1, 10, 100], 
  'solver' : ['newton-cg', 'lbfgs', 'liblinear']}]

logistic_grid = GridSearchCV(LogisticRegression(), param_grid, cv = 3,
                        scoring = 'accuracy', verbose = 10)

logistic_grid.fit(x_logistic_train, y_logistic_train)
print logistic_grid.best_params_
print logistic_grid.best_score_

# The best logistic regression has penalty = l1, C = 1, solver = liblinear.
# Its score is .824

clf_logistic = LogisticRegression(penalty = 'l1', C = 1, solver = 'liblinear')

#### SVM
param_grid = [
  {'C': [.001, .01, .1, 1, 10], 'kernel': ['linear']}]

svm_grid = GridSearchCV(SVC(), param_grid, cv = 3, scoring = 'accuracy', 
                        verbose = 10)
svm_grid.fit(x_svm_train, y_svm_train)
print svm_grid.best_params_
print svm_grid.best_score_

# The best SVM has C = .1, kernel = linear. Its score is .825
clf_svm = SVC(C = .1, kernel = 'linear', probability = True)


####SVM with RBF kernel (use tree-selected data):
param_grid = [{'C': [.01, .1, 1, 10], 'gamma': [0.001, 0.0001], 
              'kernel': ['rbf']}]

svm_rbf_grid = GridSearchCV(SVC(), param_grid, cv = 3, scoring = 'accuracy', 
                        verbose = 10)
svm_rbf_grid.fit(x_tree_selected_train, y_tree_selected_train)
print svm_rbf_grid.best_params_
print svm_rbf_grid.best_score_

# The best SVM with rbf kernel has C = 10, gamma = .001. Its score is .810
clf_svm_rbf = SVC(C = .1, kernel = 'rbf', gamma = .001, probability = True)


#### KNN
param_grid = [{'n_neighbors': [1, 10, 100], 'weights' : ['distance', 'uniform'],
               'leaf_size' : [10, 100, 1000], 'p' : [1,2]}]

knn_grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv = 3,
                         scoring = 'accuracy', verbose = 10)
knn_grid.fit(x_tree_selected_train, y_tree_selected_train)
print knn_grid.best_params_
print knn_grid.best_score_

# The best KNN has n_neighbors = 100, weights = 'uniform', leaf_size = 10, p = 2
# Its score is .811
clf_knn = KNeighborsClassifier(n_neighbors = 100, weights = 'uniform',
                           leaf_size = 10, p = 2)


#### Decision tree
param_grid = [{
    'criterion' : ['gini', 'entropy'], 'max_depth' : [2,4,6,8,10,12], 
    'min_samples_split' : [2,10,50], 'min_samples_leaf' : [1,5,10]
}]

tree_grid = GridSearchCV(DecisionTreeClassifier(), param_grid, cv = 3,
                         scoring = 'accuracy', verbose = 10)
tree_grid.fit(x_tree_selected_train, y_tree_selected_train)
print tree_grid.best_params_
print tree_grid.best_score_

# The best decision tree has min_samples_split = 50, criterion = 'gini', 
# max_depth = 6, min_samples_leaf = 5. Its score is .818
clf_tree = DecisionTreeClassifier(min_samples_split = 50, criterion = 'gini',
                              max_depth = 6, min_samples_leaf = 5)


#### Random forest
param_grid = [{'n_estimators' : [3,9,27], 'criterion' : ['gini', 'entropy'], 
  'max_depth' : [2,4,6,8], 'min_samples_split' : [2,10,50], 'min_samples_leaf' : [1,5,10]}]

forest_grid = GridSearchCV(RandomForestClassifier(), param_grid, cv = 3,
                           scoring = 'accuracy', verbose = 10)
forest_grid.fit(x_tree_selected_train, y_tree_selected_train)
print forest_grid.best_params_
print forest_grid.best_score_

# The best random forest has n_estimators = 27, criterion = 'entropy', max_depth = 8, 
# min_samples_split = 50, min_samples_leaf = 1. Its score is .827.
clf_forest = RandomForestClassifier(n_estimators = 27, criterion = 'entropy',
                                max_depth = 8, min_samples_split = 50,
                                min_samples_leaf =1)


#                   ***Compare models against each other***
# Strategy: first, plot ROC curves of all models on a single plot to compare
# the models. Also print their accuracies to compare. If there's time, plot 
# heatmaps of where in the data the models do well.

# Plot ROC curves. Remember the data for each model is the features selected
# data subset for that specific model.

def roc_plotter(classifier, name, x_train, y_train, x_test, y_test):

    # Run classifier
    probas_ = classifier.fit(x_train, y_train).predict_proba(x_test)

    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
    roc_auc = auc(fpr, tpr)
    print "Area under the ROC curve : %f" % roc_auc

    # Plot ROC curve
    with sns.color_palette('hls', 8):
      plt.plot(fpr, tpr, label='AUC = %0.2f for %s' % (roc_auc, name), alpha = .85)
      plt.plot([0, 1], [0, 1], 'k--', color = 'black', alpha = .05)
      plt.xlim([0.0, 1.0])
      plt.ylim([0.0, 1.0])

roc_plotter(clf_logistic, 'logistic', x_logistic_train, y_logistic_train, 
            x_logistic_test, y_logistic_test)
roc_plotter(clf_forest, 'forest', x_tree_selected_train, y_tree_selected_train, 
            x_tree_selected_test, y_tree_selected_test)
roc_plotter(clf_logistic, 'logistic w/tree selected features', x_tree_selected_train, y_tree_selected_train, 
            x_tree_selected_test, y_tree_selected_test)
roc_plotter(clf_knn, 'knn', x_tree_selected_train, y_tree_selected_train, 
            x_tree_selected_test, y_tree_selected_test)
roc_plotter(clf_svm, 'linear svm', x_svm_train, y_svm_train, x_svm_test, y_svm_test)
roc_plotter(clf_svm, 'linear svm w/tree selected features', x_tree_selected_train, y_tree_selected_train, 
            x_tree_selected_test, y_tree_selected_test)
roc_plotter(clf_svm_rbf, 'svm_rbf', x_tree_selected_train, y_tree_selected_train, 
            x_tree_selected_test, y_tree_selected_test)
roc_plotter(clf_tree, 'tree', x_tree_selected_train, y_tree_selected_train, 
            x_tree_selected_test, y_tree_selected_test)
plt.xlabel('False Positive Rate', size = 14)
plt.ylabel('True Positive Rate', size = 14)
plt.title('Receiver operating characteristic example', size = 15)
plt.xticks(size = 12)
plt.yticks(size = 12)
plt.legend(title = 'Area under ROC curve\nfor each classifier',
           fontsize = 12, loc= 4)
plt.show()

# Now use the ENTIRE dataset and calculate cross_val_score for each classifier.
# I'm going to use the trained models to predict the rest of the dataset and
# obtain an accuracy score for each.
data2 = data_all[4000:]
data2_dummied = pd.get_dummies(data2)
data2_dummied = data2_dummied.drop(data2_dummied.columns[-2], axis = 1)

x2 = data2_dummied.ix[:,:-1]
y2 = data2_dummied[data2_dummied.columns[-1]]

# First, build a function to pre-process the data so it's the right shape:
def preprocess(classifier, supported_features = None, x = x2):
    if supported_features != None:
      return cut_irrelevant_features(x, supported_features)
    else:
      return selector.transform(x)

# Need to re-fit the classifiers that used different feature sets above bc
# they're still fitted with one of the feature sets. After I do that, I preprocess
# the data and compute accuracy scores.
clf_logistic.fit(x_logistic_train, y_logistic_train)
x2_logistic = preprocess(clf_logistic, supported_features = support_logistic)
acc_logistic = accuracy_score(y2, clf_logistic.predict(x2_logistic))

clf_logistic.fit(x_tree_selected_train, y_tree_selected_train)
x2_logistic_tree_selected = preprocess(clf_logistic)
acc_logistic_tree_processed = accuracy_score(y2, clf_logistic.predict(x2_logistic_tree_selected))

clf_svm.fit(x_svm_train, y_svm_train)
x2_svm = preprocess(clf_svm, supported_features = support_svm)
acc_svm = accuracy_score(y2, clf_svm.predict(x2_svm))

clf_svm.fit(x_tree_selected_train, y_tree_selected_train)
x2_svm_tree_selected = preprocess(clf_svm)
acc_svm_tree_processed = accuracy_score(y2, clf_svm.predict(x2_svm_tree_selected))

x2_svm_rbf = preprocess(clf_svm_rbf)
acc_svm_rbf = accuracy_score(y2, clf_svm_rbf.predict(x2_svm_rbf))

x2_knn = preprocess(clf_knn)
acc_knn = accuracy_score(y2, clf_knn.predict(x2_knn))

x2_tree = preprocess(clf_tree)
acc_tree = accuracy_score(y2, clf_tree.predict(x2_tree))

x2_forest = preprocess(clf_forest)
acc_forest = accuracy_score(y2, clf_forest.predict(x2_forest))


print "Accuracy scores of each model against the validation data:\n\
  logistic: %f \n\
  logistic w/tree selected data: %f \n\
  svm w/linear kernel: %f \n\
  svm w/linear kernel and tree selected data: %f \n\
  svm w/rbf kernel: %f \n\
  knn: %f \n\
  decision tree: %f \n\
  random forest: %f" % (acc_logistic, acc_logistic_tree_processed, acc_svm, 
                        acc_svm_rbf, acc_knn, acc_svm_tree_processed,
                        acc_tree, acc_forest)



