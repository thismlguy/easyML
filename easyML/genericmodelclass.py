#####################################################################
##### IMPORT STANDARD MODULES
#####################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, GridSearchCV
from sklearn import metrics, model_selection
from sklearn.feature_selection import RFE, RFECV
from abc import ABCMeta, abstractmethod

from .base_class import BaseClass

#####################################################################
##### GENERIC MODEL CLASS
#####################################################################

class GenericModelClass(BaseClass):
    """ This class contains the generic classification functionms and variable definitions applicable across all models 
    """

    #Define as a meta class to disable direct instances
    __metaclass__ = ABCMeta

    #Empty dictionary for default parameters to be updated in inhereted 
    #class:
    default_parameters = {}

    def __init__(
        self, alg, data_block, predictors=[],cv_folds=5,
        scoring_metric='accuracy',additional_display_metrics=[]):

        self.alg = alg
        self.datablock = data_block
        self.cv_folds = cv_folds
        self.predictors = predictors
        self.predictions_class = {}

        #define scoring metric:
        self.scoring_metric = scoring_metric
        self.additional_display_metrics=additional_display_metrics

        #grid-search objects:
        self.gridsearch_class = None
        self.gridsearch_result = None

        #not to be used for all but most
        self.feature_imp = None
    
    #Modify and get predictors for the model:
    def set_predictors(self, predictors):
        """ Set the columns to be used predictors (also called independent 
        variables or features) of the model.
        
        Parameters
        ----------
        predictors : list of strings
            A list of columns which are to be used as predictors (also 
            called independent variables or features).
        """
        self.predictors=predictors


    def set_scoring_metric(scoring_metric, additional_display_metrics=[]):
        """ A method to update the metric to be used for scoring the 
        models and some additional metric just for reference if required.
        
        Parameters
        ----------
        scoring_metric : str
            This is same as in model class.

        additional_display_metrics : list of str
            This is same as in model class.
            If nothing specified, then the original values will not be
            modified
        """
        self.scoring_metric = scoring_metric
        if additional_display_metrics:
            self.additional_display_metrics = additional_display_metrics


    def KFold_CrossValidation(self, scoring_metric):
        # Generate cross validation folds for the training dataset. 

        error = model_selection.cross_val_score(
                estimator=self.alg, 
                X=self.datablock.train[self.predictors].values, 
                y=self.datablock.train[self.datablock.target].values, 
                cv=self.cv_folds, scoring=scoring_metric, n_jobs=-1
                ) 
            
        return {
            'mean_error': np.mean(error),
            'std_error': np.std(error),
            'all_error': error 
            }

    
    def recursive_feature_elimination(self, nfeat=None, step=1, inplace=False):

        """A method to implement recursive feature elimination on the model.
        Note that CV is not performed in this function. The method will 
        continue to eliminate some features (specified by step parameter)
        at each step until the specified number of features are reached.

        Parameters
        __________
        nfeat : int or None, default=None
            The num of top features to select. If None, half of the features 
            are selected.

        step : int or float, default=1
            If int, then step corresponds to the number of features to remove
            at each iteration. 
            If float and within (0.0, 1.0), then step corresponds to the 
            percentage (rounded down) of features to remove at each 
            iteration.
            If float and greater than one, then integral part will be
            considered as an integer input
            
        inplace : bool, default=False
            If True, the predictors of the class are modified to those 
            selected by the RFE procedure.

        Returns
        _______
        selected : A series object containing the selected features as 
        index and their rank in selection as values
        """
        rfe = RFE(self.alg, n_features_to_select=nfeat, step=step)
        
        rfe.fit(
                self.datablock.train[self.predictors], 
                self.datablock.train[self.datablock.target]
                )
        
        ranks = pd.Series(rfe.ranking_, index=self.predictors)
        
        selected = ranks.loc[rfe.support_]

        if inplace:
            self.set_predictors(selected.index.tolist())
        
        return selected

    
    def recursive_feature_elimination_cv(self, step=1, inplace=False):
        """A method to implement recursive feature elimination on the model 
        with cross-validation(CV). At each step, features are ranked as per 
        the algorithm used and lowest ranked features are removed,
        as specified by the step argument. At each step, the CV score is 
        determined using the scoring metric specified in the model. The set 
        of features with highest cross validation scores is then chosen. 

        Parameters
        __________
        step : int or float, default=1
            If int, then step corresponds to the number of features to remove
            at each iteration. 
            If float and within (0.0, 1.0), then step corresponds to the 
            percentage (rounded down) of features to remove at each 
            iteration.
            If float and greater than one, then integral part will be
            considered as an integer input
            
        inplace : bool, default=False
            If True, the predictors of the class are modified to those 
            selected by the RFECV procedure.

        Returns
        _______
        selected : pandas series
            A series object containing the selected features as 
            index and their rank in selection as values
        """
        rfecv = RFECV(
                self.alg, step=step,cv=self.cv_folds,
                scoring=self.scoring_metric,n_jobs=-1
                )
        
        rfecv.fit(
                self.datablock.train[self.predictors], 
                self.datablock.train[self.datablock.target]
                )

        if step>1:
            min_nfeat = (len(self.predictors) 
                        - step*(len(rfecv.grid_scores_)-1))

            plt.xlabel("Number of features selected")
            plt.ylabel("Cross validation score")
            plt.plot(
                    range(min_nfeat, len(self.predictors)+1, step), 
                    rfecv.grid_scores_
                    )
            plt.show(block=False)

        ranks = pd.Series(rfecv.ranking_, index=self.predictors)
        selected = ranks.loc[rfecv.support_]

        if inplace:
            self.set_predictors(selected.index.tolist())
        return ranks

    
    def grid_search(
        self, param_grid, n_jobs=1,iid=True, cv=5, getResults=False):
        """A method to perform GridSearch with cross-validation (CV) to find
        the optimum parameters for the model.

        Parameters
        __________
        param_grid : dict 
            A dictionary with parameters names (string) as keys and lists of
            parameter settings to try as values.
            Example: for a decision tree, {'max_depth':[3,4,5],
            'min_samples_split':[30,50,100]} will check all the 9 possible 
            combinations of parameters 

        n_jobs : int, default=1
            The number of jobs to run in parallel

        iid : bool, default=True
            If True, the data is assumed to be identically distributed across
            the folds, and the loss minimized is the total loss per sample, 
            and not the mean loss across the folds.

        cv : int, default=5
            The number of cross-validation folds to generate for testing each
            set of parameters

        getResults : bool, default=False
            If True, a pandas dataframe containing the results of the grid 
            search will be returned

        Returns
        _______
        gridsearch_results : pandas dataframe
            The pandas dataframe containing the grid-search results.
        """
        self.gridsearch_class = GridSearchCV(
                self.alg, param_grid=param_grid, scoring=self.scoring_metric,
                n_jobs=n_jobs, iid=iid, cv=cv)

        self.gridsearch_class.fit(
                self.datablock.train[self.predictors], 
                self.datablock.train[self.datablock.target]
                )

        self.gridsearch_result = pd.DataFrame(
                                self.gridsearch_class.cv_results_)

        print('Grid Search Results:')
        print(self.gridsearch_result[
            ['params','mean_test_score','std_test_score',
            'mean_train_score','std_train_score']
            ])
        print('\nBest Parameters: ', self.gridsearch_class.best_params_)
        print('\nBest Score: ', self.gridsearch_class.best_score_)
        
        if getResults:
            return self.gridsearch_result

    
    def submission(self, IDcol, filename="submission.csv"):
        """This method creates a submission file (mostly used for online 
        hackathons) with the prediction classes.

        Parameters
        __________
        IDcol : str or list of str
            The column names to be present in the exported file apart from
            the prediction columns. They generally represent the unique IDs
            of each observation or row.

        filename : str, default="submission.csv"
            The name of the file, along with the path, which will contain
            the submission results. The exported file will have the columns
            mentioned in the IDcol parameter along with column representing
            the predicted class.
        """

        if not isinstance(IDcol,list):
            IDcol = [IDcol]
        submission = pd.DataFrame(
            { x: self.datablock.test[x] for x in IDcol }
            )
        #The output is typecasted to integer specifically so that the csv
        # file doesn't read the format wrong
        submission[self.datablock.target] = \
                                self.predictions['predict'].astype(int)
        submission.to_csv(filename, index=False)


    def create_ensemble_dir(self):
        #checks whether the ensemble directory exists and creates one if 
        #it doesn't
        ensdir = os.path.join(os.getcwd(), 'ensemble')
        if not os.path.isdir(ensdir):
            os.mkdir(ensdir)

    # Define abstract classes which the classes which inheret this will implement:
    @abstractmethod
    def set_parameters(self):
        pass 

    @abstractmethod
    def fit_model(self):
        pass 

    @abstractmethod
    def export_model(self):
        pass 