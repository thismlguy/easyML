#####################################################################
##### IMPORT STANDARD MODULES
#####################################################################

#Python 3 support:
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mode
import warnings

from .base_class import BaseClass
from .data import DataBlock

#####################################################################
##### DEFINE PREPROCESS CLASS
#####################################################################

class FeatureEngineering(BaseClass):
    """ A feature engineering module which helps you create new features
    into the data using the easy-to-use built-in modules. The features
    are created on all 3 of the dataframes in datablock simultaneously.

    Parameters
    ----------
    data_block : object of type DataBlock
        An object of easyML's DataBlock class. You should first create an 
        object of that class and then pass it as a parameter.

    Attributes
    ----------
    datablock : object of type DataBlock
        The pre-processed data containing all the modifications made 
        using the module's methods
    """
    def __init__(self, data_block):

        #check if data element is of class datablock
        self.check_datatype(data_block,'data_block',DataBlock)

        self.datablock = data_block
        self.target = self.datablock.target        #though redundant but will make code mode readable

        #get tuple of available data
        self.dp = self.datablock.data_present().values()

    
    def create_bins(
        self,column, cut_points,labels=None, inplace=True, suffix='_bin'):
        """ Used to create a categorical feature from a numeric feature by 
        binning specific ranges of numbers together. 

        Parameters
        __________
        column : str 
            This represents the column on which the function is to be applied

        cut_points : list of num 
            This represents a list of numbers at which the data has to be 
            cut. For example, an input [5,10] will divide the column into 3 
            categories: [(-inf,5],(5,10],(10,inf)] 
        
        labels : list of str or None, default=None
            User-defined labels for the resulting category in increasing 
            order. The length of the list should be 1 more than the length 
            of cuts_points. 
            If None, then default labels are 0, 1, ..., n-1. Note that an 
            object type feature is formed if custom labels provided, 
            otherwise the returned columns is of type numeric by default.

        inplace : bool, default=True 
            If True, then the original column in the data will be modified.
            If False, then a new column will be created with a suffix as 
            specified by suffix parameter.

        suffix : str, default='_bin'
            If inplace argument is False, then this is the suffix applied to
            the column name to creat a new column. 
            Note that if such a column already exists, it will be overwritten
        """

        #check:
        self.check_datatype(column,'column',basestring)
        self.check_datatype(cut_points,'cut_points',list)       
        self.check_datatype(inplace,'inplace',bool)
        self.check_datatype(suffix,'suffix',basestring) 

        if labels:
            self.check_datatype(labels,'labels',list) 
            if len(labels) != len(cut_points)+1:
                raise self.InvalidInput("Length of labels (%d) should be 1 more than the length of cut_points (%d)"%(len(labels),len(cut_points)))

        self.subset_check(column)
        
        #Define tag for converting to numeric:
        change_to_num = False

        #Define min and max values:
        min_val = min([data[column].min() for data in self.dp])
        max_val = max([data[column].max() for data in self.dp])

        #create list by adding min and max to cut_points
        break_points = [min_val] + cut_points + [max_val]

        #if no labels provided, use default labels 0 ... (n-1)
        if not labels:
            labels = range(len(cut_points)+1)
            change_to_num=True

        if inplace:
                new_var = column
        else:
            new_var = "".join([column,suffix])
        
        #Bin all datasets:
        for data in self.dp:
            data[new_var] = pd.cut(
                        data[column], bins=break_points,
                        labels=labels, include_lowest=True
                        )
            if change_to_num:
                data[new_var] = data[new_var].apply(lambda x: int(x))

    def combine_categories(
        self,column, combine_dict, inplace=True, suffix='_combined'):
        """ Used to comine categories of a categorical feature. 
        Note that only the categories specified in the combine_dict will be 
        changed and rest all will remain as it is.

        Parameters
        __________
        column : str 
            This represents the column on which the function is to be applied

        combine_dict : dict
            This represents the replacement dictionary of the form 
            {'new_categories':[list of old categories]} where the dictionary 
            keys represents the new categories to be created and the values 
            represent the old categories which are to be combined to get the 
            new one. 
            For example, {'d':['a','b','c']} will combine all 'a','b','c' to 
            give just 1 category 'd', which might be a pre-existing category 
            or a new one.

        inplace : bool, default=True 
            If True, then the original column in the data will be modified.
            If False, then a new column will be created with a suffix as 
            specified by suffix parameter.

        suffix : str, default='_combined'
            If inplace argument is False, then this is the suffix applied to
            the column name to creat a new column. 
            Note that if such a column already exists, it will be overwritten
        """

        #check:
        self.check_datatype(column,'column',basestring)
        self.check_datatype(combine_dict,'combine_dict',dict)     
        self.check_datatype(inplace,'inplace',bool)
        self.check_datatype(suffix,'suffix',basestring) 

        self.subset_check(column)

        #Check each value is a list:
        self.check_list_type(
                combine_dict.values(),list,
                "Each value of the combine_dict should be a list")

        replace_dict={}
        #dictionary comprehension to form a reverse dictionary with keys as 
        #the old value and value as new value.
        [replace_dict.update({val:key}) 
            for key,values in combine_dict.items() 
                for val in values]

        if inplace:
                new_var = column
        else:
            new_var = "".join([column,suffix])
            for data in self.dp:
                data[new_var] = data[column]

        for data in self.dp:
            data[new_var].replace(replace_dict,inplace=True)

