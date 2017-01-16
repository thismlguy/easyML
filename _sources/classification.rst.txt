Classification Models
=====================

This module helps you perform the classification task using a variety of models from scikit-learn implementation. This module provides an easy to use interface with following advantages:

- Define scoring metrics at time of initialization, which gets computed at time of modelfit
- A single line of code allows you to:
  + Fit the model 
  + Compute pre-defined scoring metrics on both test and train data 
  + Make predictions on data with unknown labels
  + Print a summary of the model with pre-defined metrics and feature importance charts
- Perform easy grid search
- Perform recursive feature elimination

Supported Algorithms
--------------------

The module supports a bunch of scikit-learn's modules as below:

.. toctree::
   :maxdepth: 3
   
   classification/logistic_regression
   classification/decision_tree
   classification/random_forest
   classification/extra_trees
   classification/adaboost
   classification/gbm
   classification/linear_svm

Note: The support for XGBoost is coming up next. Stay tuned!!
