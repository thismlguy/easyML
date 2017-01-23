#####################################################################
##### IMPORT STANDARD MODULES
#####################################################################

from __future__ import print_function

from ..data import DataBlock

from ..preprocess import PreProcess

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from random import sample

#####################################################################
##### TESTS FOR DATABLOCK
#####################################################################

def test_datablock(datablock):
	assert datablock.train.shape == (150, 5)
	assert datablock.test.shape == (150, 5)
	assert datablock.predict.shape == (150, 5)

#####################################################################
##### TESTS FOR PREPROCESS
#####################################################################

def test_check_missing_no_missing(datablock):
	pp = PreProcess(datablock)
	result = pp.check_missing(printResult=False,returnResult=True)
	for df,miss in result.items():
		print(df,miss)
		assert miss.sum()==0

def test_check_missing_missing_induced(datablock):
	df = pd.DataFrame(datablock.train,copy=True)
	pp = PreProcess(DataBlock(df,df,df,'target'))
	num_miss=25
	for data in pp.datablock.data_present().values():
		data.iloc[sample(range(150),num_miss),0] = np.nan
	result = pp.check_missing(printResult=False,returnResult=True)
	for key,miss in result.items():
		assert miss.sum()==num_miss

