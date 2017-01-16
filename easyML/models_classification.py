#####################################################################
##### IMPORT STANDARD MODULES
#####################################################################

#Python 3 support:
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import pydot
import os
from scipy.stats.mstats import chisquare, mode
    
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier 
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics, model_selection
from sklearn.feature_selection import RFE, RFECV
from abc import ABCMeta, abstractmethod
# from StringIO import StringIO
# import xgboost as xgb
# from xgboost.sklearn import XGBClassifier

from .genericmodelclass import GenericModelClass
from .data import DataBlock


#####################################################################
##### GENERIC MODEL CLASS
#####################################################################

class base_classification(GenericModelClass):
    """ A base class which defines the generic classification functions 
    and variable definitions.

    Parameters
    ----------
    alg : object
        An sklearn-style estimator

    data_block : object
        An object of easyML's DataBlock class. You should first create an 
        object of that class and then pass it as a parameter.

    predictors : list of strings, default []
        A list of columns which are to be used as predictors (also called 
        independent variables or features).
        The default value is an empty list because these need not always be
        defined at the time of class initialization. The set_predictors 
        method can be used later but before creating any predictive model.

    cv_folds : int, default 5
        The number of folds to be created while performing CV.
        This parameter can be adjusted later by passing using the 
        set_parameters method

    scoring_metric : str, default 'accuracy'
        The scoring metric to be used for evaluating the model across the
        different functions available. The available options are 
        - 'accuracy'
        - 'auc'
        - 'log_loss'
        - 'f1'
        - 'average_precision'
    
    additional_display_metrics : list of string, default []
        A list of additional display metrics to be shown for the test and
        train dataframes in data_block. Note:
        - These will be just shown for user reference and not actually used 
        for model evaluation
        - The same available options as scoring_metric apply
    """

    #Define as a meta class to disable direct instances
    __metaclass__ = ABCMeta

    # Map possible inputs to functions in sklean.metrics. 
    # Each value of the dictionary is a tuple of 3:
    # (function, multi-class support, requires-probabilities)
        # function: the sklearn metrics function
        # multi-class support: if True, function allows multi-class support
        # requires-probabilities: if True, the function requires 
        # probabilities to be passed as arguments
    metrics_map = {
        'accuracy':(metrics.accuracy_score,True,False),
        'auc':(metrics.roc_auc_score,False,True),
        'log_loss':(metrics.log_loss,True,True),
        'f1':(metrics.f1_score,True,False),
        'average_precision':(metrics.average_precision_score,False,True)
    }

    def __init__(
            self, alg, data_block, predictors=[],cv_folds=5,
            scoring_metric='accuracy',additional_display_metrics=[]
        ):

        GenericModelClass.__init__(
            self, alg=alg, data_block=data_block, predictors=predictors,
            cv_folds=cv_folds,scoring_metric=scoring_metric,
            additional_display_metrics=additional_display_metrics)

        #Run input datatype checks:
        self.check_datatype(data_block,'data_block',DataBlock)
        self.subset_check(predictors)
        self.check_datatype(cv_folds,'cv_folds',int)
        self.check_datatype(scoring_metric,'scoring_metric',basestring)
        self.check_datatype(
            additional_display_metrics,'additional_display_metrics',list)

        #Store predicted probabilities in a dictionary with keys as the
        # name of the dataset (train/test/predict) and values as the actual
        # predictions.
        self.predictions_probabilities = {}  

        #Boolean to store whether the estimator chosen allows probability
        # predictions
        self.probabilities_available = True

        #Define number of classes in target.
        self.num_target_class = len(
            self.datablock.train[self.datablock.target].unique())

        #A Series object to store generic classification model outcomes. 
        self.classification_output=pd.Series(
            index = ['ModelID','CVScore_mean','CVScore_std','AUC',
                    'ActualScore (manual entry)','CVMethod','Predictors']
                )

        #Get the dictionary of available dataframes
        self.dp = self.datablock.data_present()

        #Check all the entered metrics. Note that this check has to be
        #placed after declaration of num_target_class attribute
        for metric in [scoring_metric]+additional_display_metrics:
            self.check_metric(metric,self.num_target_class)

    @classmethod
    def check_metric(cls,metric,num_target_class):
        if metric not in cls.metrics_map:
            raise self.InvalidInput("The input '%s' is not a valid scoring metric for this module"%metric)

        if num_target_class>2:
            if not cls.metrics_map[metric][1]:
                raise self.InvalidInput("The %s metric does not support multi-class classification case"%metric)

    
    def fit_model(
        self, performCV=True, printResults=True,
        printTopN=None, printConfusionMatrix=True,
        printModelParameters=True):
        
        """An advanced model fit function which fits the model on the 
        training data and performs cross-validation. It prints a model
        report containing the following:
        - The parameters being used to fit the model
        - Confusion matrix for the train and test data
        - Scoring metrics for the train and test data
        - CV mean and std scores for scoring metric
        - Additional scoring metrics on train and test data, if specified

        Note that you can decide which details are to be printed using method
        arguments.
        
        Parameters
        ----------
        performCV : bool, default True
            if True, the model performs cross-validation using the number of
            folds as the cv_folds parameter of the model

        printResults : bool, default True
            if True, prints the report of the model. This should be kept as 
            True unless the module being used in a background script

        printTopN : int, default None
            The number of top scored features to be displayed in the feature 
            importance or coefficient plot of the model. If None, all the 
            features will be displayed by default. Note:
            - For algorithms supporting real coefficient, the features will 
            be sorted by their magnitudes (absolute values).
            - For algorithms supporting positive feature importance scores, 
            features are sorted on the score itself.

            This will be ignored is printResults is False.

        printConfusionMatrix : bool, default True
            if True, the confusion matrix for the train and test dataframes 
            are printed, otherwise they are ommitted.
            This will be ignored is printResults is False.

        print
        
        printModelParameters : bool, default True
            if True, the parameters being used to the run the model are 
            printed. It helps in validating the parameters and also makes
            jupyter notebooks more informative if used
        """

        self.check_datatype(performCV,'performCV',bool)
        self.check_datatype(printResults,'printResults',bool)
        self.check_datatype(printConfusionMatrix,'printConfusionMatrix',bool)
        self.check_datatype(printModelParameters,'printModelParameters',bool)
        if printTopN:
            self.check_datatype(printTopN,'printTopN',int)
          
        self.alg.fit(
            self.datablock.train[self.predictors], 
            self.datablock.train[self.datablock.target])

        #Get algo_specific_values
        self.algo_specific_fit(printTopN)
          
        #Get predictions:
        for key,data in self.dp.items():
            self.predictions_class[key] = self.alg.predict(
                                                data[self.predictors])

        if self.probabilities_available:
            for key,data in self.dp.items():
                self.predictions_probabilities[key] = self.alg.predict_proba(
                                                    data[self.predictors])

        self.calc_model_characteristics(performCV)
        if printResults:
            self.printReport(printConfusionMatrix, printModelParameters)


    def calc_model_characteristics(self, performCV=True):
        # Determine key metrics to analyze the classification model. These 
        # are stored in the classification_output series object belonginf to 
        # this class.
        for metric in [self.scoring_metric]+self.additional_display_metrics:
            #Determine for both test and train, except predict:
            for key,data in self.dp.items():
                if key!='predict':  
                    name = '%s_%s'%(metric,key)
                    #Case where probabilities to be passed as arguments
                    if base_classification.metrics_map[metric][2]:
                        self.classification_output[name] = \
                            base_classification.metrics_map[metric][0](
                                data[self.datablock.target],
                                self.predictions_probabilities[key])
                    #case where class predictions to be passed  as arguments
                    else:                                                   
                        self.classification_output[name] = \
                            base_classification.metrics_map[metric][0](
                                data[self.datablock.target],
                                self.predictions_class[key])

                #Determine confusion matrix:
                name = 'ConfusionMatrix_%s'%key
                self.classification_output[name] = pd.crosstab(
                        data[self.datablock.target], 
                        self.predictions_class[key]
                    ).to_string()

        if performCV:
            cv_score = self.KFold_CrossValidation(
                        scoring_metric=self.scoring_metric)
        else:
            cv_score = {
                'mean_error': 0.0, 
                'std_error': 0.0
            }

        self.classification_output['CVMethod'] = \
                                        'KFold - ' + str(self.cv_folds)
        self.classification_output['CVScore_mean'] = cv_score['mean_error']
        self.classification_output['CVScore_std'] = cv_score['std_error']
        self.classification_output['Predictors'] = str(self.predictors)

    
    def printReport(self, printConfusionMatrix, printModelParameters):
        # Print the metric determined in the previous function.

        print("\nModel Report")
        #Outpute the parameters used for modeling
        if printModelParameters:
            print('\nModel being built with the following parameters:')
            print(self.alg.get_params())

        if printConfusionMatrix:
            for key,data in self.dp.items():
                if key!='predict':
                    print("\nConfusion Matrix for %s data:"%key)
                    print(pd.crosstab(
                            data[self.datablock.target], 
                            self.predictions_class[key])
                    )
            print('Note: rows - actual; col - predicted')
        
        print("\nScoring Metric:")
        for key,data in self.dp.items():
            if key!='predict':
                name = '%s_%s'%(self.scoring_metric,key)
                print("\t%s (%s): %s" % 
                    (
                    self.scoring_metric,
                    key,
                    "{0:.3%}".format(self.classification_output[name])
                    )
                )

        print("\nCV Score for Scoring Metric (%s):"%self.scoring_metric)
        print("\tMean - %f | Std - %f" % (
            self.classification_output['CVScore_mean'],
            self.classification_output['CVScore_std'])
        )

        if self.additional_display_metrics:
            print("\nAdditional Scoring Metrics:")
            for metric in self.additional_display_metrics:
                for key,data in self.dp.items():
                    if key!='predict':
                        name = '%s_%s'%(metric,key)
                        print("\t%s (%s): %s" % (
                            metric,
                            key,
                            "{0:.3%}".format(
                                    self.classification_output[name])
                            )
                        )
        
    def plot_feature_importance(self, printTopN):
        num_print = len(self.feature_imp)
        if printTopN is not None:
            num_print = min(printTopN,len(self.feature_imp))
        self.feature_imp.iloc[:num_print].plot(
                kind='bar', title='Feature Importances')
        plt.ylabel('Feature Importance Score')
        plt.show(block=False)

    def plot_abs_coefficients(self,coeff,printTopN):
        num_print = len(coeff)
        if printTopN is not None:
            num_print = min(printTopN,num_print)

        coeff_abs_sorted = sorted(
                abs(coeff).index,
                key=lambda x: abs(coeff_abs[x]),
                reverse=True
            )
        
        coeff[coeff_abs_sorted].iloc[:num_print,].plot(
                    kind='bar', 
                    title='Feature Coefficients (Sorted by Magnitude)'
                )
        plt.ylabel('Magnitute of Coefficients')
        plt.show(block=False)

    
    def submission_proba(
        self, IDcol, proba_colnames,filename="Submission.csv"):
        """ 

        """
        submission = pd.DataFrame({ 
            x: self.datablock.predict[x] for x in list(IDcol)
            })
        
        if len(list(proba_colnames))>1:
            for i in range(len(proba_colnames)):
                submission[proba_colnames[i]] = self.test_pred_prob[:,i]
        else: 
            submission[list(proba_colnames)[0]] = self.test_pred_prob[:,1]
        submission.to_csv(filename, index=False)

    
    def set_parameters(self, param=None, cv_folds=None, set_default=False):
        """ Set the parameters of the model. Only the parameters to be
        updated are required to be passed.

        Parameters
        __________
        param : dict, default None
            A dictionary of key,value pairs where the keys are the parameters
            to be updated and values as the new value of those parameters.
            If None, no update performed
            Ignored if set_default iss True.
        
        cv_folds : int, default None
            Pass the number of CV folds to be used in the model.
            If None, no update performed.

        set_default : bool, default True
            if True, the model will be set to default parameters as defined 
            in model definition by scikit-learn. Note that this will not 
            affect the cv_folds parameter.
        """

        #Check input
        self.check_datatype(param,'param',dict)
        self.check_datatype(set_default,'set_default',bool)

        if param:
            if set(param.keys()).issubset(
                    set(base_classification.default_parameters.keys())
                    ):
                raise self.InvalidInput("""The parameters passed should be a 
                    subset of the model parameters""")

        if set_default:
            param = self.default_parameters
            
        self.alg.set_params(**param)
        self.model_output.update(pd.Series(param))

        if cv_folds:
            self.cv_folds = cv_folds

    def export_model_base(self, IDcol, mstr):
        self.create_ensemble_dir()
        filename = os.path.join(os.getcwd(),'ensemble/%s_models.csv'%mstr)
        comb_series = self.classification_output.append(
                                        self.model_output, 
                                        verify_integrity=True)

        if os.path.exists(filename):
            models = pd.read_csv(filename)
            mID = int(max(models['ModelID'])+1)
        else:
            mID = 1
            models = pd.DataFrame(columns=comb_series.index)
            
        comb_series['ModelID'] = mID
        models = models.append(comb_series, ignore_index=True)
        
        models.to_csv(filename, index=False, float_format="%.5f")
        model_filename = os.path.join(
                                os.getcwd(),
                                'ensemble/%s_%s.csv'%(mstr,str(mID))
                                )
        self.submission(IDcol, model_filename)

    @abstractmethod
    def algo_specific_fit(self,printTopN):
          #Run algo-specific commands
          pass

    @abstractmethod
    def export_model(self,IDcol):
          #Export models
          pass


#####################################################################
##### LOGISTIC REGRESSION
#####################################################################

class logistic_regression(base_classification):
    """ Create a Logistic Regression model using implementation from 
    scikit-learn.

    Parameters
    ----------
    data_block : object of type easyML.DataBlock
        An object of easyML's DataBlock class. You should first create an 
        object of that class and then pass it as a parameter.

    predictors : list of strings, default []
        A list of columns which are to be used as predictors (also called 
        independent variables or features).
        The default value is an empty list because these need not always be
        defined at the time of class initialization. The set_predictors 
        method can be used later but before creating any predictive model.

    cv_folds : int, default 5
        The number of folds to be created while performing CV.
        This parameter can be adjusted later by passing using the 
        set_parameters method

    scoring_metric : str, default 'accuracy'
        The scoring metric to be used for evaluating the model across the
        different functions available. The available options are 
        - 'accuracy'
        - 'auc'
        - 'log_loss'
        - 'f1'
        - 'average_precision'
    
    additional_display_metrics : list of string, default []
        A list of additional display metrics to be shown for the test and
        train dataframes in data_block. Note:
        - These will be just shown for user reference and not actually used 
        for model evaluation
        - The same available options as scoring_metric apply
    """

    default_parameters = {
            'C':1.0, 
            'tol':0.0001, 
            'solver':'liblinear',
            'multi_class':'ovr',
            'class_weight':'balanced'
        }

    def __init__(
        self,data_block, predictors=[],cv_folds=10,
        scoring_metric='accuracy',additional_display_metrics=[]):

        base_classification.__init__(
            self, alg=LogisticRegression(), data_block=data_block, 
            predictors=predictors,cv_folds=cv_folds,
            scoring_metric=scoring_metric, 
            additional_display_metrics=additional_display_metrics
            )

        self.model_output=pd.Series(self.default_parameters)
        self.model_output['Coefficients'] = "-"

        #Set parameters to default values:
        self.set_parameters(set_default=True)

    def algo_specific_fit(self, printTopN):

        if self.num_target_class==2:
            coeff = pd.Series(
                np.concatenate(
                    (self.alg.intercept_,
                    self.alg.coef_[0])), 
                index=["Intercept"]+self.predictors
            )
            self.plot_abs_coefficients(coeff,printTopN)
        else:
            cols=['coef_class_%d'%i for i in range(0,self.num_target_class)]
            coeff = pd.DataFrame(
                            self.alg.coef_.T, 
                            columns=cols,
                            index=self.predictors
                            )
            print('\nCoefficients:')
            print(coeff)

        self.model_output['Coefficients'] = coeff.to_string()
    
    
    def export_model(self, IDcol):
        #Export the model into the model file as well as create a submission 
        #with model index. This will be used for creating an ensemble.
        self.export_model_base(IDcol,'logistic_reg')


#####################################################################
##### DECISION TREE
#####################################################################

class decision_tree(base_classification):

    """ Create a Decision Tree model using implementation from 
    scikit-learn.

    Parameters
    ----------
    data_block : object of type easyML.DataBlock
        An object of easyML's DataBlock class. You should first create an 
        object of that class and then pass it as a parameter.

    predictors : list of strings, default []
        A list of columns which are to be used as predictors (also called 
        independent variables or features).
        The default value is an empty list because these need not always be
        defined at the time of class initialization. The set_predictors 
        method can be used later but before creating any predictive model.

    cv_folds : int, default 5
        The number of folds to be created while performing CV.
        This parameter can be adjusted later by passing using the 
        set_parameters method

    scoring_metric : str, default 'accuracy'
        The scoring metric to be used for evaluating the model across the
        different functions available. The available options are 
        - 'accuracy'
        - 'auc'
        - 'log_loss'
        - 'f1'
        - 'average_precision'
    
    additional_display_metrics : list of string, default []
        A list of additional display metrics to be shown for the test and
        train dataframes in data_block. Note:
        - These will be just shown for user reference and not actually used 
        for model evaluation
        - The same available options as scoring_metric apply
    """
    default_parameters = {
        'criterion':'gini', 
        'max_depth':None, 
        'min_samples_split':2, 
        'min_samples_leaf':1, 
        'max_features':None, 
        'random_state':None, 
        'max_leaf_nodes':None, 
        'class_weight':'balanced'
        }

    def __init__(
        self,data_block, predictors=[],cv_folds=10,
        scoring_metric='accuracy',additional_display_metrics=[]):

        base_classification.__init__(
            self, alg=DecisionTreeClassifier(), data_block=data_block, 
            predictors=predictors,cv_folds=cv_folds,
            scoring_metric=scoring_metric, 
            additional_display_metrics=additional_display_metrics
            )

        self.model_output = pd.Series(self.default_parameters)
        self.model_output['Feature_Importance'] = "-"

        #Set parameters to default values:
        self.set_parameters(set_default=True)
    
    def algo_specific_fit(self, printTopN):
        # print Feature Importance Scores table
        self.feature_imp = pd.Series(
                            self.alg.feature_importances_, 
                            index=self.predictors
                        ).sort_values(ascending=False)
        
        self.plot_feature_importance(printTopN)
        self.model_output['Feature_Importance'] = \
                                    self.feature_imp.to_string()

    def export_model(self, IDcol):
        #Export the model into the model file as well as create a submission 
        #with model index. This will be used for creating an ensemble.
        self.export_model_base(IDcol,'decision_tree')

    ## UNDER DEVELOPMENT CODE FOR PRINTING TREES
    # def get_tree(self):
    #     return self.alg.tree_
    # Print the tree in visual format
    # Inputs:
    #     export_pdf - if True, a pdf will be exported with the 
    #     filename as specified in pdf_name argument
    #     pdf_name - name of the pdf file if export_pdf is True
    # def printTree(self, export_pdf=True, file_name="Decision_Tree.pdf"):
    #     dot_data = StringIO() 
    #     export_graphviz(
    #             self.alg, out_file=dot_data, feature_names=self.predictors,
    #             filled=True, rounded=True, special_characters=True)

    #     export_graphviz(
    #         self.alg, out_file='data.dot', feature_names=self.predictors,  
    #         filled=True, rounded=True, special_characters=True
    #         ) 
    #     graph = pydot.graph_from_dot_data(dot_data.getvalue())
        
    #     if export_pdf:
    #         graph.write_pdf(file_name)

    #     return graph

#####################################################################
##### RANDOM FOREST
#####################################################################

class random_forest(base_classification):
    """ Create a Random Forest model using implementation from 
    scikit-learn.

    Parameters
    ----------
    data_block : object of type easyML.DataBlock
        An object of easyML's DataBlock class. You should first create an 
        object of that class and then pass it as a parameter.

    predictors : list of strings, default []
        A list of columns which are to be used as predictors (also called 
        independent variables or features).
        The default value is an empty list because these need not always be
        defined at the time of class initialization. The set_predictors 
        method can be used later but before creating any predictive model.

    cv_folds : int, default 5
        The number of folds to be created while performing CV.
        This parameter can be adjusted later by passing using the 
        set_parameters method

    scoring_metric : str, default 'accuracy'
        The scoring metric to be used for evaluating the model across the
        different functions available. The available options are 
        - 'accuracy'
        - 'auc'
        - 'log_loss'
        - 'f1'
        - 'average_precision'
    
    additional_display_metrics : list of string, default []
        A list of additional display metrics to be shown for the test and
        train dataframes in data_block. Note:
        - These will be just shown for user reference and not actually used 
        for model evaluation
        - The same available options as scoring_metric apply
    """

    default_parameters = {
        'n_estimators':10, 
        'criterion':'gini', 
        'max_depth':None, 
        'min_samples_split':2, 
        'min_samples_leaf':1, 
        'max_features':'auto', 
        'max_leaf_nodes':None,
        'oob_score':False, 
        'random_state':None, 
        'class_weight':'balanced', 
        'n_jobs':1 
    }
        
    def __init__(
        self,data_block, predictors=[],cv_folds=10,
        scoring_metric='accuracy',additional_display_metrics=[]):

        base_classification.__init__(
            self, alg=RandomForestClassifier(), data_block=data_block, 
            predictors=predictors,cv_folds=cv_folds,
            scoring_metric=scoring_metric, 
            additional_display_metrics=additional_display_metrics
            )

        self.model_output = pd.Series(self.default_parameters)
        self.model_output['Feature_Importance'] = "-"
        self.model_output['OOB_Score'] = "-"

        #Set parameters to default values:
        self.set_parameters(set_default=True)


    def algo_specific_fit(self, printTopN):
        # print Feature Importance Scores table
        self.feature_imp = pd.Series(
            self.alg.feature_importances_, 
            index=self.predictors
            ).sort_values(ascending=False)

        self.plot_feature_importance(printTopN)

        self.model_output['Feature_Importance'] = \
                                self.feature_imp.to_string()

        if self.model_output['oob_score']:
            print('OOB Score : %f' % self.alg.oob_score_)
            self.model_output['OOB_Score'] = self.alg.oob_score_

    def export_model(self, IDcol):
        #Export the model into the model file as well as create a submission 
        #with model index. This will be used for creating an ensemble.
        self.export_model_base(IDcol,'random_forest')

#####################################################################
##### EXTRA TREES FOREST
#####################################################################

class extra_trees(base_classification):
    """ Create an Extra Trees Forest model using implementation from 
    scikit-learn.

    Parameters
    ----------
    data_block : object of type easyML.DataBlock
        An object of easyML's DataBlock class. You should first create an 
        object of that class and then pass it as a parameter.

    predictors : list of strings, default []
        A list of columns which are to be used as predictors (also called 
        independent variables or features).
        The default value is an empty list because these need not always be
        defined at the time of class initialization. The set_predictors 
        method can be used later but before creating any predictive model.

    cv_folds : int, default 5
        The number of folds to be created while performing CV.
        This parameter can be adjusted later by passing using the 
        set_parameters method

    scoring_metric : str, default 'accuracy'
        The scoring metric to be used for evaluating the model across the
        different functions available. The available options are 
        - 'accuracy'
        - 'auc'
        - 'log_loss'
        - 'f1'
        - 'average_precision'
    
    additional_display_metrics : list of string, default []
        A list of additional display metrics to be shown for the test and
        train dataframes in data_block. Note:
        - These will be just shown for user reference and not actually used 
        for model evaluation
        - The same available options as scoring_metric apply
    """

    default_parameters = {
        'n_estimators':10, 
        'criterion':'gini', 
        'max_depth':None,
        'min_samples_split':2,
        'min_samples_leaf':1, 
        'max_features':'auto', 
        'max_leaf_nodes':None,
        'oob_score':False, 
        'random_state':None, 
        'class_weight':'balanced', 
        'n_jobs':1 
    }

    def __init__(
        self,data_block, predictors=[],cv_folds=10,
        scoring_metric='accuracy',additional_display_metrics=[]):

        base_classification.__init__(
            self, alg=ExtraTreesClassifier(), data_block=data_block, 
            predictors=predictors,cv_folds=cv_folds,
            scoring_metric=scoring_metric, 
            additional_display_metrics=additional_display_metrics)

        self.model_output = pd.Series(self.default_parameters)
        self.model_output['Feature_Importance'] = "-"
        self.model_output['OOB_Score'] = "-"

        #Set parameters to default values:
        self.set_parameters(set_default=True)


    def algo_specific_fit(self, printTopN):
        # print Feature Importance Scores table
        self.feature_imp = pd.Series(
                                self.alg.feature_importances_, 
                                index=self.predictors
                            ).sort_values(ascending=False)

        self.plot_feature_importance(printTopN)

        self.model_output['Feature_Importance'] = \
                                self.feature_imp.to_string()

        if self.model_output['oob_score']:
            print('OOB Score : %f' % self.alg.oob_score_)
            self.model_output['OOB_Score'] = self.alg.oob_score_

    def export_model(self, IDcol):
        #Export the model into the model file as well as create a submission 
        #with model index. This will be used for creating an ensemble.
        self.export_model_base(IDcol,'extra_trees')

#####################################################################
##### ADABOOST CLASSIFICATION
#####################################################################

class adaboost(base_classification):
    """ Create an AdaBoost model using implementation from 
    scikit-learn.

    Parameters
    ----------
    data_block : object of type easyML.DataBlock
        An object of easyML's DataBlock class. You should first create an 
        object of that class and then pass it as a parameter.

    predictors : list of strings, default []
        A list of columns which are to be used as predictors (also called 
        independent variables or features).
        The default value is an empty list because these need not always be
        defined at the time of class initialization. The set_predictors 
        method can be used later but before creating any predictive model.

    cv_folds : int, default 5
        The number of folds to be created while performing CV.
        This parameter can be adjusted later by passing using the 
        set_parameters method

    scoring_metric : str, default 'accuracy'
        The scoring metric to be used for evaluating the model across the
        different functions available. The available options are 
        - 'accuracy'
        - 'auc'
        - 'log_loss'
        - 'f1'
        - 'average_precision'
    
    additional_display_metrics : list of string, default []
        A list of additional display metrics to be shown for the test and
        train dataframes in data_block. Note:
        - These will be just shown for user reference and not actually used 
        for model evaluation
        - The same available options as scoring_metric apply
    """

    default_parameters = { 
        'n_estimators':50, 
        'learning_rate':1.0 
    }

    def __init__(
        self,data_block, predictors=[],cv_folds=10,
        scoring_metric='accuracy',additional_display_metrics=[]):

        base_classification.__init__(
            self, alg=AdaBoostClassifier(), data_block=data_block, 
            predictors=predictors,cv_folds=cv_folds,
            scoring_metric=scoring_metric, 
            additional_display_metrics=additional_display_metrics
            )

        self.model_output = pd.Series(self.default_parameters)
        self.model_output['Feature_Importance'] = "-"

        #Set parameters to default values:
        self.set_parameters(set_default=True)

    def algo_specific_fit(self, printTopN):
        # print Feature Importance Scores table
        self.feature_imp = pd.Series(
                        self.alg.feature_importances_, 
                        index=self.predictors
                        ).sort_values(ascending=False)

        self.plot_feature_importance(printTopN)

        self.model_output['Feature_Importance'] = \
                                        self.feature_imp.to_string()

        plt.xlabel("AdaBoost Estimator")
        plt.ylabel("Estimator Error")
        plt.plot(
            range(1, int(self.model_output['n_estimators'])+1), 
            self.alg.estimator_errors_
            )
        plt.plot(
            range(1, int(self.model_output['n_estimators'])+1), 
            self.alg.estimator_weights_
            )
        plt.legend(
            ['estimator_errors','estimator_weights'], 
            loc='upper left'
            )
        plt.show(block=False)


    def export_model(self, IDcol):
        #Export the model into the model file as well as create a submission 
        #with model index. This will be used for creating an ensemble.
        self.export_model_base(IDcol,'adaboost')

#####################################################################
##### GRADIENT BOOSTING MACHINE
#####################################################################

class gradient_boosting_machine(base_classification):
    """ Create a GBM (Gradient Boosting Machine) model using implementation 
    from scikit-learn.

    Parameters
    ----------
    data_block : object of type easyML.DataBlock
        An object of easyML's DataBlock class. You should first create an 
        object of that class and then pass it as a parameter.

    predictors : list of strings, default []
        A list of columns which are to be used as predictors (also called 
        independent variables or features).
        The default value is an empty list because these need not always be
        defined at the time of class initialization. The set_predictors 
        method can be used later but before creating any predictive model.

    cv_folds : int, default 5
        The number of folds to be created while performing CV.
        This parameter can be adjusted later by passing using the 
        set_parameters method

    scoring_metric : str, default 'accuracy'
        The scoring metric to be used for evaluating the model across the
        different functions available. The available options are 
        - 'accuracy'
        - 'auc'
        - 'log_loss'
        - 'f1'
        - 'average_precision'
    
    additional_display_metrics : list of string, default []
        A list of additional display metrics to be shown for the test and
        train dataframes in data_block. Note:
        - These will be just shown for user reference and not actually used 
        for model evaluation
        - The same available options as scoring_metric apply
    """

    default_parameters = {
        'loss':'deviance', 
        'learning_rate':0.1, 
        'n_estimators':100, 
        'subsample':1.0, 
        'min_samples_split':2, 
        'min_samples_leaf':1,
        'max_depth':3, 'init':None, 
        'random_state':None, 
        'max_features':None, 
        'verbose':0, 
        'max_leaf_nodes':None, 
        'warm_start':False, 
        'presort':'auto'
    }

    def __init__(
        self, data_block, predictors=[],cv_folds=10,
        scoring_metric='accuracy',additional_display_metrics=[]):

        base_classification.__init__(
            self, alg=GradientBoostingClassifier(), data_block=data_block, 
            predictors=predictors,cv_folds=cv_folds,
            scoring_metric=scoring_metric, 
            additional_display_metrics=additional_display_metrics
            )
        
        self.model_output = pd.Series(self.default_parameters)
        self.model_output['Feature_Importance'] = "-"
        
        #Set parameters to default values:
        self.set_parameters(set_default=True)

    def algo_specific_fit(self, printTopN):
        # print Feature Importance Scores table
        self.feature_imp = pd.Series(
                            self.alg.feature_importances_, 
                            index=self.predictors
                            ).sort_values(ascending=False)

        self.plot_feature_importance(printTopN)

        self.model_output['Feature_Importance'] = \
                                        self.feature_imp.to_string()

        #Plot OOB estimates if subsample <1:
        if self.model_output['subsample']<1:
            plt.xlabel("GBM Iteration")
            plt.ylabel("Score")
            plt.plot(
                range(1, self.model_output['n_estimators']+1), 
                self.alg.oob_improvement_
                )
            plt.legend(['oob_improvement_','train_score_'], loc='upper left')
            plt.show(block=False)

        
    def export_model(self, IDcol):
        #Export the model into the model file as well as create a submission 
        #with model index. This will be used for creating an ensemble.
        self.export_model_base(IDcol,'gbm')


#####################################################################
##### Support Vector Classifier
#####################################################################

class linear_svm(base_classification):
    """ Create a Linear Support Vector Machine model using implementation 
    from scikit-learn.

    Parameters
    ----------
    data_block : object of type easyML.DataBlock
        An object of easyML's DataBlock class. You should first create an 
        object of that class and then pass it as a parameter.

    predictors : list of strings, default []
        A list of columns which are to be used as predictors (also called 
        independent variables or features).
        The default value is an empty list because these need not always be
        defined at the time of class initialization. The set_predictors 
        method can be used later but before creating any predictive model.

    cv_folds : int, default 5
        The number of folds to be created while performing CV.
        This parameter can be adjusted later by passing using the 
        set_parameters method

    scoring_metric : str, default 'accuracy'
        The scoring metric to be used for evaluating the model across the
        different functions available. The available options are 
        - 'accuracy'
        - 'auc'
        - 'log_loss'
        - 'f1'
        - 'average_precision'
    
    additional_display_metrics : list of string, default []
        A list of additional display metrics to be shown for the test and
        train dataframes in data_block. Note:
        - These will be just shown for user reference and not actually used 
        for model evaluation
        - The same available options as scoring_metric apply
    """

    default_parameters = {
        'C':1.0, 
        'kernel':'linear',  #modified not default
        'degree':3, 
        'gamma':'auto', 
        'coef0':0.0, 
        'shrinking':True, 
        'probability':False, 
        'tol':0.001, 
        'cache_size':200, 
        'class_weight':None, 
        'verbose':False, 
        'max_iter':-1, 
        'decision_function_shape':None, 
        'random_state':None
    }

    def __init__(
        self,data_block, predictors=[],cv_folds=10,
        scoring_metric='accuracy',additional_display_metrics=[]):

        base_classification.__init__(
            self, alg=SVC(), data_block=data_block, predictors=predictors,
            cv_folds=cv_folds,scoring_metric=scoring_metric, 
            additional_display_metrics=additional_display_metrics
            )
        
        self.model_output=pd.Series(self.default_parameters)
        self.model_output['Coefficients'] = "-"

        #Set parameters to default values:
        self.set_parameters(set_default=True)

        #Check if probabilities enables:
        if not self.alg.get_params()['probability']:
            self.probabilities_available = False  
        
    
    def algo_specific_fit(self, printTopN):

        if self.num_target_class==2:
            coeff = pd.Series(
                np.concatenate((self.alg.intercept_,self.alg.coef_[0])),
                index=["Intercept"]+self.predictors
                )

            #print the chart of importances
            self.plot_abs_coefficients(coeff, printTopN)
        else:
            cols=['coef_class_%d'%i for i in range(0,self.num_target_class)]
            coeff = pd.DataFrame(
                            self.alg.coef_.T, 
                            columns=cols,
                            index=self.predictors
                            )
            print('\nCoefficients:')
            print(coeff)

        self.model_output['Coefficients'] = coeff.to_string()
    
    def export_model(self, IDcol):
        #Export the model into the model file as well as create a submission 
        #with model index. This will be used for creating an ensemble.
        self.export_model_base(IDcol,'linear_svm')


#####################################################################
##### XGBOOST ALGORITHM (UNDER DEVELOPMENT)
#####################################################################

"""
#Define the class similar to the overall classification class
class XGBoost(base_classification):
    def __init__(self,data_block, predictors, cv_folds=5,scoring_metric_skl='accuracy', scoring_metric_xgb='error'):
        
        base_classification.__init__(self, alg=XGBClassifier(), data_block=data_block, predictors=predictors,cv_folds=cv_folds,scoring_metric=scoring_metric_skl)
        
        #Define default parameters on your own:
        self.default_parameters = { 
                                 'max_depth':3, 'learning_rate':0.1,
                                 'n_estimators':100, 'silent':True,
                                 'objective':"binary:logistic",
                                 'nthread':1, 'gamma':0, 'min_child_weight':1,
                                 'max_delta_step':0, 'subsample':1, 'colsample_bytree':1, 'colsample_bylevel':1,
                                 'reg_alpha':0, 'reg_lambda':1, 'scale_pos_weight':1,
                                 'base_score':0.5, 'seed':0, 'missing':None
                            }
        self.model_output = pd.Series(self.default_parameters)

        #create DMatrix with nan as missing by default. If later this is changed then the matrix are re-calculated. If not set,will give error is nan present in data
        self.xgtrain = xgb.DMatrix(self.datablock.train[self.predictors].values, label=self.datablock.train[self.datablock.target].values, missing=np.nan)
        self.xgtest = xgb.DMatrix(self.datablock.predict[self.predictors].values, missing=np.nan)
        self.num_class = 2
        self.n_estimators = 10
        self.eval_metric = 'error'

        self.train_predictions = []
        self.train_pred_prob = []
        self.test_predictions = []
        self.test_pred_prob = []
        self.num_target_class = len(data_train[target].unique())

        #define scoring metric:
        self.scoring_metric_skl = scoring_metric_skl
        # if scoring_metric_xgb=='f1':
        #    self.scoring_metric_xgb = self.xg_f1
        # else:
        self.scoring_metric_xgb = scoring_metric_xgb

        #Define a Series object to store generic classification model outcomes; 
        self.classification_output=pd.Series(index=['ModelID','Accuracy','CVScore_mean','CVScore_std','SpecifiedMetric',
                                             'ActualScore (manual entry)','CVMethod','ConfusionMatrix','Predictors'])

        #feature importance (g_scores)
        self.feature_imp = None
        self.model_output['Feature_Importance'] = "-"

        #Set parameters to default values:
        # self.set_parameters(set_default=True)

    #Define custom f1 score metric:
    def xg_f1(self,y,t):
        t = t.get_label()
        y_bin = [1. if y_cont > 0.5 else 0. for y_cont in y] # binaryzing your output
        return 'f1',metrics.f1_score(t,y_bin)

    # Set the parameters of the model. 
    # Note: 
    #    > only the parameters to be updated are required to be passed
    #    > if set_default is True, the passed parameters are ignored and default parameters are set which are defined in   scikit learn module
    def set_parameters(self, param=None, set_default=False):        
        if set_default:
            param = self.default_parameters
            
        self.alg.set_params(**param)
        self.model_output.update(pd.Series(param))

        if 'missing' in param:
            #update DMatrix with missing:
            self.xgtrain = xgb.DMatrix(self.datablock.train[self.predictors].values, label=self.datablock.train[self.datablock.target].values, missing=param['missing'])
            self.xgtest = xgb.DMatrix(self.datablock.predict[self.predictors].values, missing=param['missing'])

        if 'num_class' in param:
            self.num_class = param['num_class']

        if 'cv_folds' in param:
            self.cv_folds = param['cv_folds']

    # def set_feature_importance(self):
        
    #    fs = self.alg.booster().get_fscore()
    #    ftimp = pd.DataFrame({
    #            'feature': fs.keys(),
    #            'importance_Score': fs.values()
    #        })
    #    ftimp['predictor'] = ftimp['feature'].apply(lambda x: self.predictors[int(x[1:])])
    #    self.feature_imp = pd.Series(ftimp['importance_Score'].values, index=ftimp['predictor'].values)

    #Fit the model using predictors and parameters specified before.
    # Inputs:
    #    printCV - if True, CV is performed
    def modelfit(self, performCV=True, useTrainCV=False, TrainCVFolds=5, early_stopping_rounds=20, show_progress=True, printTopN='all'):

        if useTrainCV:
            xgb_param = self.alg.get_xgb_params()
            if self.num_class>2:
                xgb_param['num_class']=self.num_class
            if self.scoring_metric_xgb=='f1':
                cvresult = xgb.cv(xgb_param,self.xgtrain, num_boost_round=self.alg.get_params()['n_estimators'], nfold=self.cv_folds,
                 metrics=['auc'],feval=self.xg_f1,early_stopping_rounds=early_stopping_rounds, show_progress=show_progress) 
            else:  
                cvresult = xgb.cv(xgb_param,self.xgtrain, num_boost_round=self.alg.get_params()['n_estimators'], nfold=self.cv_folds,
                metrics=self.scoring_metric_xgb, early_stopping_rounds=early_stopping_rounds, show_progress=show_progress)
            self.alg.set_params(n_estimators=cvresult.shape[0])

        print(self.alg.get_params())
        obj = self.alg.fit(self.datablock.train[self.predictors], self.datablock.train[self.datablock.target], eval_metric=self.eval_metric)
        
        #Print feature importance
        # self.set_feature_importance()
        self.feature_imp = pd.Series(self.alg.booster().get_fscore()).sort_values(ascending=False)
        num_print = len(self.feature_imp)
        if printTopN is not None:
            if printTopN != 'all':
                num_print = min(printTopN,len(self.feature_imp))
            self.feature_imp.iloc[:num_print].plot(kind='bar', title='Feature Importances')
            plt.ylabel('Feature Importance Score')
            plt.show(block=False)

        self.model_output['Feature_Importance'] = self.feature_imp.to_string()

        #Get train predictions:
        self.train_predictions = self.alg.predict(self.datablock.train[self.predictors])
        self.train_pred_prob = self.alg.predict_proba(self.datablock.train[self.predictors])

        #Get test predictions:
        self.test_predictions = self.alg.predict(self.datablock.predict[self.predictors])
        self.test_pred_prob = self.alg.predict_proba(self.datablock.predict[self.predictors])

        self.calc_model_characteristics(performCV)
        self.printReport()

    
    #Export the model into the model file as well as create a submission with model index. This will be used for creating an ensemble.
    def export_model(self, IDcol):
        self.create_ensemble_dir()
        filename = os.path.join(os.getcwd(),'ensemble/xgboost_models.csv')
        comb_series = self.classification_output.append(self.model_output, verify_integrity=True)

        if os.path.exists(filename):
            models = pd.read_csv(filename)
            mID = int(max(models['ModelID'])+1)
        else:
            mID = 1
            models = pd.DataFrame(columns=comb_series.index)
            
        comb_series['ModelID'] = mID
        models = models.append(comb_series, ignore_index=True)
        
        models.to_csv(filename, index=False, float_format="%.5f")
        model_filename = os.path.join(os.getcwd(),'ensemble/xgboost_'+str(mID)+'.csv')
        self.submission(IDcol, model_filename)
"""

#####################################################################
##### ENSEMBLE (UNDER DEVELOPMENT)
#####################################################################
"""
#Class for creating an ensemble model using the exported files from previous classes
class Ensemble_Classification(object):
    #initialize the object with target variable
    def __init__(self, target, IDcol):
        self.datablock.target = target
        self.data = None
        self.relationMatrix_chi2 = None
        self.relationMatrix_diff = None
        self.IDcol = IDcol

    #create the ensemble data
    # Inputs:
    #     models - dictionary with key as the model name and values as list containing the model numbers to be ensebled
    # Note: all the models in the list specified should be present in the ensemble folder. Please cross-check once 
    def create_ensemble_data(self, models):
        self.data = None
        for key, value in models.items():
            # print key,value
            for i in value:
                fname = key + '_' + str(i)
                fpath = os.path.join(os.getcwd(), 'ensemble', fname+'.csv')
                tempdata = pd.read_csv(fpath)
                tempdata = tempdata.rename(columns = {self.datablock.target: fname})
                if self.data is None:
                    self.data = tempdata
                else:
                    self.data = self.data.merge(tempdata,on=self.data.columns[0])

    #get the data being used for ensemble
    def get_ensemble_data(self):
        return self.data
    
    #Check chisq test between different model outputs to check which combination of ensemble will generate better results. Note: Models with high correlation should not be combined together.
    def chisq_independence(self, col1, col2, verbose = False):
        contingencyTable = pd.crosstab(col1,col2,margins=True)

        if len(col1)/((contingencyTable.shape[0] - 1) * (contingencyTable.shape[1] - 1)) <= 5:
            return "TMC"

        expected = contingencyTable.copy()
        total = contingencyTable.loc["All","All"]
        # print contingencyTable.index
        # print contingencyTable.columns
        for m in contingencyTable.index:
            for n in contingencyTable.columns:
                expected.loc[m,n] = contingencyTable.loc[m,"All"]*contingencyTable.loc["All",n]/float(total)
        
        if verbose:
            print('\n\nAnalysis of models: %s and %s' % (col1.name, col2.name))
            print('Contingency Table:')
            print(contingencyTable)
            # print '\nExpected Frequency Table:'
            # print expected
        observed_frq = contingencyTable.iloc[:-1,:-1].values.ravel()
        expected_frq = expected.iloc[:-1,:-1].values.ravel()

        numless1 = len(expected_frq[expected_frq<1])
        perless5 = len(expected_frq[expected_frq<5])/len(expected_frq)

        #Adjustment in DOF so use the 1D chisquare to matrix shaped data; -1 in row n col because of All row and column
        matrixadj = (contingencyTable.shape[0] - 1) + (contingencyTable.shape[1] - 1) - 2
        # print matrixadj
        pval = np.round(chisquare(observed_frq, expected_frq,ddof=matrixadj)[1],3)

        if numless1>0 or perless5>=0.2:
            return str(pval)+"*"
        else: 
            return pval

    #Create the relational matrix between models
    def check_ch2(self, verbose=False):
        col = self.data.columns[1:]
        self.relationMatrix_chi2 = pd.DataFrame(index=col,columns=col)

        for i in range(len(col)):
            for j in range(i, len(col)):
                if i==j:
                    self.relationMatrix_chi2.loc[col[i],col[j]] = 1
                else:
                    pval = self.chisq_independence(self.data.iloc[:,i+1],self.data.iloc[:,j+1], verbose=verbose)
                    self.relationMatrix_chi2.loc[col[j],col[i]] = pval
                    self.relationMatrix_chi2.loc[col[i],col[j]] = pval

        print('\n\n Relational Matrix (based on Chi-square test):')
        print(self.relationMatrix_chi2)

    def check_diff(self):
        col = self.data.columns[1:]
        self.relationMatrix_diff = pd.DataFrame(index=col,columns=col)
        nrow = self.data.shape[0]
        for i in range(len(col)):
            for j in range(i, len(col)):
                if i==j:
                    self.relationMatrix_diff.loc[col[i],col[j]] = '-'
                else:
                    # print col[i],col[j]
                    pval = "{0:.2%}".format(sum( np.abs(self.data.iloc[:,i+1]-self.data.iloc[:,j+1]) )/float(nrow))
                    self.relationMatrix_diff.loc[col[j],col[i]] = pval
                    self.relationMatrix_diff.loc[col[i],col[j]] = pval

        print('\n\n Relational Matrix (based on perc difference):')
        print(self.relationMatrix_diff)


    #Generate submission for the ensembled model by combining the mentioned models.
    # Inputs:
    #     models_to_use - list with model names to use; if None- all models will be used
    #     filename - the filename of the final submission
    #     Note: the models should be odd in nucmber to allow a clear winner in terms of mode otherwise the first element will be chosen 
    def submission(self, models_to_use=None, filename="Submission_ensemble.csv"):

        #if models_to_use is None then use all, else filter:
        if models_to_use is None:
            data_ens = self.data
        else:
            data_ens = self.data[models_to_use]

        def mode_ens(x):
            return int(mode(x).mode[0])

        ensemble_output = data_ens.apply(mode_ens,axis=1)
        submission = pd.DataFrame({
                self.IDcol: self.data.iloc[:,0],
                self.datablock.target: ensemble_output
            })
        submission.to_csv(filename, index=False)
"""