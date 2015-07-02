from RaschModel import LearnRaschModel
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

"""
For this example to work, download the smallest movielens data:
http://files.grouplens.org/datasets/movielens/ml-100k.zip
"""

# setup initial params
thresh = 4

# read training and testing data
df_train = pd.read_table("data/ml-100K/u1.base", header=-1)
df_test = pd.read_table("data/ml-100K/u1.test", header=-1)

# binarize ratings for the example to work
df_train[2] = (df_train[2] > thresh) * 1
df_test[2] = (df_test[2] > thresh) * 1

# initialize class for learning rasch model using default params
lrm = LearnRaschModel(verbose=True)

# fix model on training data
lrm.fit(df_train, user_id=0, item_id=1, response=2, inplace=True)

print "Time taken: " + str(lrm.time_taken)

pr = lrm.predict(df_test, user_id=0, item_id=1)

print "AUC on test data: " + str(roc_auc_score(df_test[2], pr))
