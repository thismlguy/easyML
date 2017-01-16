Pre-Processing Module
=====================

The preprocess module is designed to help you perform some basic checks like missing values. It also helps you clearn your data for analysis by providing easy-to-use functions for imputation, scaling data and normalizing the data. It also has a generic apply function to help you 

Upcoming Features
-----------------

- Check near-zero variance
- Check duplicates
- Check data types
- Check multi-colinraity
- Split data train-test
- Sense check on data
- Text processing

Class Structure
---------------

 .. autoclass:: easyML.preprocess.PreProcess
   :members: check_missing, imputation, scale, normalize, apply
..   :member_order: bysource