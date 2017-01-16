#####################################################################
##### IMPORT STANDARD MODULES
#####################################################################

from ..data import DataBlock

import pandas as pd
from sklearn.datasets import load_iris
import pytest

#####################################################################
##### DEFINE FIXTURES
#####################################################################

@pytest.fixture(scope='module')
def datablock():
	X,y = load_iris(return_X_y=True)
	df = pd.DataFrame(X,columns=['var%d'%i for i in range(4)])
	df['target'] = y

	#make 1 variable categorical
	df['var3'] = df['var3'].apply(lambda x: int(x)).astype(object)

	#make outcome binary
	df['target'].iloc[df['target']==2]=1
	return DataBlock(df,df,df,'target')
