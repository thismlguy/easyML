#####################################################################
##### IMPORT STANDARD MODULES
#####################################################################

from abc import ABCMeta, abstractmethod

#####################################################################
##### DEFINE BASE CLASSES WITH CHECKS AND EXCEPTIONS
#####################################################################

class BaseClass(object):
    
    #Define this as a meta class so that direct objects of this are not possible
    __metaclass__ = ABCMeta

    class InvalidInput(Exception):
        # Error to be raised when the subset parameter is not a subset of 
        #the train columns 
        pass

    def subset_check(self,subset):
        if not isinstance(subset,list):
            subset = [subset]
        if not set(subset).issubset(set(self.datablock.columns)):
                raise self.InvalidInput("The subset parameter is not a subset of the dataframes in datablock")

    def check_datatype(self,var,varname,types):
        #check whether var belongs to the type 
        if not (isinstance(var,types)):
            raise self.InvalidInput("%s should be %s, found %s"%(
                                        varname, types,type(var))
                                )

    def check_datatype2(self,var,varname,types):
        #check whether var belongs to one of 2 types in types
        if not (isinstance(var,types[0]) | isinstance(var,types[1])):
            raise self.InvalidInput("%s can either be %s or %s, found %s"%(
                                    varname, types[0],types[1],type(var))
                                )

    def check_list_type(self,check_list,types,message):
        #check whether each element of list if in given types
        for i in check_list:
            if not isinstance(i,types):
                raise self.InvalidInput(message)

    def check_value_in_range(self,value,range,message):
        if value not in range:
            raise self.InvalidInput(message)