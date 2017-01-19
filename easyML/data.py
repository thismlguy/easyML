#####################################################################
##### IMPORT STANDARD MODULES
#####################################################################

#Python 3 support:
from __future__ import absolute_import, division, print_function, unicode_literals

import pandas as pd
import numpy as np
import collections     #to maintain dp as an ordered dictionary
from .base_class import BaseClass

#####################################################################
##### DEFINE DATA_BLOCK CLASS
#####################################################################

class DataBlock(BaseClass):
    """ Define the data block which contains the entire set of data. 
    This is a unique data structure which will be passed as input to 
    the other classes of this module. This helps in packaging all the
    data together so that every action is performed on all the dataframes
    together, thus reducing the coding overheads significantly.

    Note: The function will CREATE COPIES of the dataset by default. If the
    data size is too big, then you can either pass file-paths from where to 
    read data or set the makeCopy parameter to False.

    Parameters
    __________
    train : pandas dataframe or str
        The training data, which is used to train the model.
        This data should have both the predictors, i.e. independent data
        and the target, i.e. dependent data.
        The data can be passed as a pandas dataframe or the filepath to the
        .csv file which stores the data. Only .csv extension is supported 
        for now.
        This data has to be provided.

    test : pandas dataframe or str or None
        The testing data.
        This data should have both the predictors, i.e. independent data
        and the target, i.e. dependent data.
        This is typically used to evaluate models. This is generally used if
        a lot of data is available and it is feasible to split it into train
        or test. Otherwise, cross-validation is performed on training data
        and there is no explicit test data.
        The data can be passed as a pandas dataframe or the filepath to the
        .csv file which stores the data. Only .csv extension is supported 
        for now.
        Pass None to the argument if data is not available.
    
    predict : pandas or str dataframe
        The data with unknown labels or targets, which are to be predicted. 
        This is generally available in online hackathons where it is called
        as a test set because true labels are available with the organizers 
        which are used to evaluate the model. But for the purpose of this 
        module, any data without known targets should be passed into predict
        dataframe because we will make these predictions and export the 
        results.
        The data can be passed as a pandas dataframe or the filepath to the
        .csv file which stores the data. Only .csv extension is supported 
        for now.
        Pass None to the argument if data is not available.

        Note that a target column with all None values will be added to this
        dataframe by the module to allow integration with other datasets. If 
        such a column already exists, then its values will be made None.
    
    target : str 
        The name of the variable representing the values to be predicted

    ID : str or list of str or None, default None
        The name of the variavle(s) which individually/together form a 
        unique entity for each observation. 
        This can be left empty at the time of initialization but will inhibit
        the functioning of certain methods in the module related to 
        exporting data.

    makeCopy : bool, default=True
        if True, then copies of the dataframes passed into the class 
        initialization will be created. Otherwise, the function will point 
        to the same variables as in your environment and they will get 
        modified by the functions of easyML module.

    Attributes
    __________
    train : pandas dataframe 
        Same as passed values. Provided so that separate copies of data are 
        not requried.

    test : pandas dataframe 
        Same as passed values. Provided so that separate copies of data are 
        not requried.

    predict : pandas dataframe 
        Same as passed values. Provided so that separate copies of data are 
        not requried.

    """

    def __init__(
            self, train, test, predict, target, ID=None, makeCopy=True):

        self.check_datatype2(train,'train',(str,pd.DataFrame))
        self.train = pd.DataFrame(train,copy=True)
        
        if test is not None:
            self.check_datatype2(test,'test',(str,pd.DataFrame))
            self.test = pd.DataFrame(test,copy=True)

        if predict is not None:
            self.check_datatype2(predict,'predict',(str,pd.DataFrame))
            self.predict = pd.DataFrame(predict,copy=True)

        self.target = target
        self.ID = ID
    
        #check data and initialize columns:
        self.check_data()

        #get columns:
        self.update_column_list()

    def update_column_list(self):
        self.columns = list(self.train.columns)
        self.numeric_columns = list(self.train._get_numeric_data().columns)
        self.other_columns = list(set(self.columns).difference(set(self.numeric_columns)))        

    class InvalidDataBlock(Exception):
        #""" Raise Exception when the data passed to the datablock is not valid """
        pass

    def check_data(self):
        #Check if train and test have same columns:
        if self.test is not None:
            if set(self.train.columns) != set(self.test.columns):
                raise self.InvalidDataBlock("The train and test dataframe should have the equal number of columns with the same names")

        #Check if predict has a subset of columns from train and everything 
        #except target:
        if self.predict is not None:
            #Make target column in predict to None or add it not there
            self.predict[self.target] = None
            if set(self.train.columns) != set(self.predict.columns):
                raise self.InvalidDataBlock("The predict dataframe should contain same columns as train with/without the target column")


    def data_present(self):
        """ Get a dictionary with key as the name of the data (train/test/
        predict) and value has the actual data. 
        Only the dataframe available are returned
        
        Returns
        __________
        dictionary : dict
            A dictionary with key as the name of the data (train/test/
            predict) and value has the actual data
        """
        dfs = collections.OrderedDict({'train':self.train})
        if self.test is not None:
            dfs['test']=self.test
        if self.predict is not None:
            dfs['predict']=self.predict
        return dfs

    def combined_data(self, addSource=False):
        """ Get a dataframe with all dataframes combined into one.
        Only the dataframe available are returned.
        
        Parameters
        __________
        addSource : bool, default=False
            if True, a column names 'source' will be added containing the
            name of the data (train/test/predict) to which the particular
            observation belongs.
            Note: if a column named source already exists, it will be 
            overwritten.

        Returns
        __________
        df : pandas dataframe
            A combined dataframe containg observation from all dataframes 
            in the datablock.
            Note that the values of the target variable for predict dataframe
            will be None
        """
        new_df = pd.concat(self.data_present(),axis=0,ignore_index=True)

        if addSource:
            source = []
            for key,data in self.data_present():
                source.append([key]*data.shape[0])
            new_df['source'] = source
            
        return new_df

