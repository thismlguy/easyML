#####################################################################
##### IMPORT STANDARD MODULES
#####################################################################

#Python 3 support:
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals

# %matplotlib inline    #only valid for iPython notebook
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

class PreProcess(BaseClass):
    """ A preprocessing modules which helps you preprocess you data
    using the easy-to-use built-in modules. The preprocessing is 
    done on all 3 of the dataframes in datablock simultaneously.

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
        #though redundant but will make code mode readable
        self.target = self.datablock.target

        #get tuple of available data
        self.dp = self.datablock.data_present().values()

    def check_missing(
        self, subset=None, printResult=True, returnSeries=False):
        """ Checks the missing values in the all dataframes. 
        The target column in the predict variable will not be 
        checked here because it is assumed to have all missing values.

        Parameters
        __________
        subset : list of str or None, default=None
            A list specifying the subset of columns in which the missing
            values are to be checked.
            If None, all columns checked

        printResult : bool, default=True 
            if True, the result will be printed

        returnSeries : bool, default=False 
            if True, the function will return a tuple of 3 objects each one 
            being a pandas Series with index as the column name and value as 
            the number of missing values in the column

        Returns
        _______
        missing_values : tuple with 3 elements
            Returns a tuple of 3 pandas series objects pertaining to the 
            train, test and predict dataframes. Each series has index as the
            name of the columns having missing values and values as the
            number of missing values in the column.
            Returns (None,None,None) if no missing values found.
            Nothing is returned is returnSeries argument is False
        """

        #Check if subset is actually a subset of the parameters or not:
        if subset:
            self.subset_check(subset)
        else:
            subset = self.datablock.columns

        check_dict = {}
        for key,data in self.dp:
            miss_val = data[subset].apply(lambda x: sum(x.isnull()))
            if key=='predict':
                #Remove target index if present:
                if self.target in miss_val:
                    miss_val.drop(self.target,inplace=True)
            check_dict[key] = miss_val

        if printResult:
            for df,miss in check_dict.items():
                if sum(miss)==0:
                    print('\nNo missing value found in %s dataframe'%df)
                else:
                    print('''\nTotal %d missing values found in %s dataframe 
                        in following columns:'''%(sum(miss),df))
                    print(pd.DataFrame(
                                miss[miss>0],
                                columns=['Num Missing Values'])
                    )

        if returnSeries:
            return (miss_train, miss_test, miss_predict)


    def imputation(
            self, column, metric=None, groupby=None, constant=None, 
            inplace=True, suffix='_imputed'):
        """ Used to performs imputation on a column. The imputation is 
        performed on all dataframes together. For instance, if a median imputation is performed, then median of the train dataframe is 
        determined and the same value used to impute the test and predict
        dataframes as well.

        Parameters
        __________
        column : str
            The name of the column to be imputed.

        metric : str or None, default=None 
            The metric to be used for imputation. Possible options:
            - mean: impute with column mean (only for numeric columns)
            - median: impute with column median; default for numeric data 
            (only for numeric columns) 
            - mode: impute with column model default for non-numeric data 
            (only for non-numeric columns)
        
        groupby : str or list of str or None, default=None 
            The list of columns by which the metric is to be grouped for 
            imputation. Note that these columns should not have any misssing 
            values.

        constant : no constraint, default=None 
            To be used if a constant user-defined value is to be used for 
            imputation. This is ignored if metric is not None.

        inplace : bool, default=True 
            If True, then the original column in the data will be imputed. 
            If False, then a new column will be created with a suffix as 
            specified by suffix parameter. This can be used to test 
            different imputation metrics.

        suffix : str, default='_imputed'
            If inplace argument is False, then this is the suffix applied to
            the column name to creat a new column. 
            Note that if such a column already exists, it will be overwritten
        """

        #Perform checks:
        self.check_datatype(column,'column',basestring)
        if metric:
            self.check_value_in_range(
                metric,
                ['mean','mediam','mode'],
                'The metric can only be "mean","median" or "mode", found %s'%metric
                )
        self.check_datatype(inplace,'inplace',bool)
        self.check_datatype(suffix,'suffix',basestring) 

        self.subset_check(column)

        if groupby:
            for col in groupby:
                self.subset_check(col)
                if sum([sum(x[col].isnull()) for x in self.dp]):
                    raise ValueError('The groupby column %s contains missing values. Please impute that first.'%col)

        if inplace:
            new_var = column
        else:
            new_var = "".join([column,suffix])
            for data in self.dp:
                data[new_var] = data[column]
            self.datablock.update_column_list()

        # If constant value passed, then exit the function with that 
        # imputation
        if constant:
            warnings.warn('Missing values being imputed with the constant value passed in metrics argument')
            for data in self.dp:
                data[new_var].fillna(metric,inplace=True)
            return

        #Define a function to impute by groups if such selected
        def fill_grps(impute_grps):
            for data in self.dp:
                for i, row in data.loc[
                                    data[column].isnull(),
                                    [column]+groupby
                                ].iterrows():
                    x = tuple(row.loc[groupby])
                    data.loc[i,new_var] = impute_grps.loc[x]
        
        #Case1: continuous column
        if column in self.datablock.numeric_columns:
            if metric is not None:
                if metric not in ['mean','median']:
                    raise ValueError('metric can only be mean or median for numeric column')
            else:
                metric = 'median'

            #check constant input:
            #Impute by mean:
            if metric == "mean":
                if groupby is not None:
                    #impute groups to be determined on basis of train data
                    #only not any other data
                    impute_grps = self.datablock.train.pivot_table(
                        values=column, index=groupby, aggfunc=np.mean, 
                        fill_value=self.datablock.train[column].median()
                        )
                    fill_grps(impute_grps)
                else:
                    impute_val =self.datablock.train[column].mean()
                    for data in self.dp:
                        data[new_var].fillna(impute_val,inplace=True)
            
            #Impute by median:
            elif metric == "median":
                if groupby is not None:
                    #impute groups to be determined on basis of train data
                    #only not any other data
                    impute_grps = self.datablock.train.pivot_table(
                        values=column, index=groupby, aggfunc=np.median, 
                        fill_value=self.datablock.train[column].median()
                        )
                    fill_grps(impute_grps)
                else:
                    impute_val =self.datablock.train[column].median()
                    for data in self.dp:
                        data[new_var].fillna(impute_val,inplace=True)

        #Case2: Categorical variable:
        if column in self.datablock.other_columns:

            if metric is not None:
                if metric not in ['mode']:
                    raise ValueError('metric can only be mode for non-numeric column')
            else:
                metric = 'mode'

            #Define the custom functino to determine the mode using scipy's 
            #mode function
            def cust_mode(x):
                return mode(x).mode[0]

            #Impute by mode:
            if metric == "mode":
                if groupby is not None:
                    #impute groups to be determined on basis of train data
                    #only not any other data
                    impute_grps = self.datablock.train.pivot_table(
                        values=column, index=groupby, aggfunc=cust_mode, 
                        fill_value=cust_mode(self.datablock.train[column])
                        )
                    fill_grps(impute_grps)
                else:
                    impute_val = cust_mode(self.datablock.train[column])
                    for data in self.dp:
                        data[new_var].fillna(impute_val,inplace=True)

    def scale(self,subset,scale_range=(0,1),inplace=True,suffix='_scaled'):
        """ Used to scale the data within a fixed range of values. 

        Parameters
        __________
        subset : str or list of str
            This represents the columns to be scaled. 2 options:
            - str: a single column to be scaled
            - list of str: list of multiple columns to be scaled

        scale_range : tuple or dictionary, default=(0,1)
            This represents the range to which the data is to be scaled. 
            2 options:
            - tuple (min,mex): fixed range for all columns mentioned in subset
            - dictionary : a dictionary with keys as columns mentioned in 
            subset list and values as the range to which that column is to 
            be scaled. Note that this works only if subset is entered as a 
            list of strings

        inplace : bool, default=True 
            If True, the dataframes will be modified and columns scaled. 
            If False, new columns will be created with suffix as specified 
            in suffix parameter
        
        suffix : str, default='_scaled'
            If inplace argument is False, then this is the suffix applied to
            the column name to creat a new column. 
            Note that if such a column already exists, it will be overwritten
        """

        #check:

        self.check_datatype2(subset,'subset',(basestring,list))
        self.check_datatype2(scale_range,'scale_range',(tuple,dict))
        self.check_datatype(inplace,'inplace',bool)
        self.check_datatype(suffix,'suffix',basestring) #basestring works for both python2 and python3

        self.subset_check(subset)

        if isinstance(subset,str):
            subset = [subset]
        
        #Iterate over all columns and scale them:
        for column in subset:
            if isinstance(scale_range,tuple):
                r = scale_range
            else:
                if column not in scale_range:
                    raise KeyError("%s not found in the dictionary range"%column)
                r = scale_range[column]

            #check each tuple of size 2:
            if len(r)!=2:
                raise InvalidInput("range should contain tuples of fixed size 2. tuple of size %d found"%len(r))

            #check second element always greater than the first
            if r[0]>=r[1]:
                raise InvalidInput("each range tuple should be of form (min,max) where min<max")

            #
            if inplace:
                new_var = column
            else:
                new_var = "".join([column,suffix])
                for data in self.dp:
                    data[new_var] = data[column]
            
            #Get min and max values:
            min_val = min([data[column].min() for data in self.dp])
            max_val = max([data[column].max() for data in self.dp])

            for data in self.dp:
                data[new_var] = data[column] - min_val
                data[new_var] = data[new_var] / (max_val - min_val)
                data[new_var] = data[new_var]*(r[1]-r[0]) + r[0]

        #Update the list of columns if inplace False
        if not inplace:
            self.datablock.update_column_list()

    def normalize(self,subset,norm='l2',inplace=True,suffix='_norm'):
        """ Used to normalize the data using an l1 or l2 normalization. 

        Parameters
        __________
        subset : list of str 
            This represents the columns to be normalized. Input should be a 
            list of columns.
        
        norm : str ('l1' or 'l2'), default='l2'
            This specifies the type or normalization- l1 or l2

        inplace : bool, default=True 
            If True, the dataframes will be modified and columns normalized. 
            If False, new columns will be created with suffix as specified 
            in suffix parameter
        
        suffix : str, default='_norm'
            If inplace argument is False, then this is the suffix applied to
            the column name to creat a new column. 
            Note that if such a column already exists, it will be overwritten
        """

        #check:
        self.check_datatype(subset,'subset',list)
        self.check_datatype(norm,'norm',basestring)     #basestring works for both python2 and python3
        self.check_datatype(inplace,'inplace',bool)
        self.check_datatype(suffix,'suffix',basestring) #basestring works for both python2 and python3

        if norm not in ['l1','l2']:
            raise self.InvalidInput("norm can only take values 'l1' or 'l2', found %s"%norm)
        self.subset_check(subset)

        #Iterate over all columns and scale them:
        for column in subset:

            #Check if column contains a missing value
            if sum(sum(data[column].isnull()) for data in self.dp)>0:
                raise self.InvalidInput("The %s column contains missing values, please impute first!"%column)

            if inplace:
                new_var = column
            else:
                new_var = "".join([column,suffix])
                for data in self.dp:
                    data[new_var] = data[column]
            
            #Get min and max values:
            if norm=='l1':
                divisor = sum([sum([abs(x) for x in data[column]]) for data in self.dp])
            else:
                divisor = np.sqrt(sum([sum([x**2 for x in data[column]]) for data in self.dp]))

            for data in self.dp:
                data[new_var] = data[new_var] / divisor

        #Update the list of columns if inplace False
        if not inplace:
            self.datablock.update_column_list()

    def apply(
        self, column, func, rowwise=True, inplace=True,
        suffix='_modified', combine_data=False):
        """ Used to apply any function on the data to allow flexibility in 
        preprocessing part by incorporating functions not directly included 
        in this package. Note that your function should be able to handle 
        missing values in data if they exist. 

        Parameters
        __________
        column : str 
            This represents the column on which the function is to be applied

        func : pre-defined function 
            A function which can be of 2 types:
            1. Takes a single value as input and returns a single value. 
            This function will be applied to each observation independently 
            and applicable if rowwise=True
            2. Takes a list of numbers as input and returns a list of same 
            size. This will be applied to entire column at same time and 
            applicable if rowwise=False

        rowwise : bool, default=True 
            if True, a function of type1 should be passed else function of 
            type2

        inplace : bool, default=True 
            If True, the dataframes will be modified. 
            If False, new columns will be created with suffix as specified 
            in suffix parameter
        
        suffix : str, default='_modified'
            If inplace argument is False, then this is the suffix applied to
            the column name to creat a new column. 
            Note that if such a column already exists, it will be overwritten

        combine_data : bool, default=False
            Works only in case of rowwise=False. If yes, the type 2 function 
            will be applied on the combined dataset together, ie. a vector 
            with observations of all the 3 datasets concatenated will be 
            passed to it and it should return a vector with numbers in the
            same order so that they can be mapped back.
        """

        #check:
        self.check_datatype(column,'column',basestring)
        if not callable(func):
            raise self.InvalidInput("The func parameter should be a callable function")
        self.check_datatype(rowwise,'rowwise',bool)
        self.check_datatype(inplace,'inplace',bool)
        self.check_datatype(suffix,'suffix',basestring) 

        self.subset_check(column)

        if inplace:
            new_var = column
        else:
            new_var = "".join([column,suffix])
            for data in self.dp:
                data[new_var] = data[column]
        
        
        #function to apply func as per rowwise:
        def applyfunc(df,col,rowwise):
            if rowwise:
                return df[col].apply(func)
            else:
                return func(list(df[col].values))

        if combine_data:
            #get tuple of available data
            result = applyfunc(self.datablock.combined_data(),column,rowwise)
            print(result)
            ind=0
            for data in self.dp:
                data[new_var] = result[ind:ind+data.shape[0]]
                ind=data.shape[0]
        else: 
            for data in self.dp:
                data[new_var] = applyfunc(data,column,rowwise)

        #Update the list of columns if inplace False
        if not inplace:
            self.datablock.update_column_list()

