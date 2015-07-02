from RaschModel import RaschModel
from RaschModel import LearnRaschModel
import numpy as np
import pandas as pd

"""
N = 10
Q = 100
a = np.random.randn(N,1)
b = np.random.randn(Q,1)
rasch = RaschModel(a, b)
Y = rasch.sample()
a_est, b_est, num_iter = LearnRaschModel(solver='newton').fit(Y)
"""

df = pd.read_table("./data/ml-100k/u.data", header=-1)
df[2] = (df[2] > 3) * 1

lrm = LearnRaschModel(solver='gradient', max_iter_outer=10, gamma=0.1, max_iter_inner=10, verbose=True)
lrm.fit(df, user_id=0, item_id=1, response=2, inplace=True)


