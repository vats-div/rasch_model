from RaschModel import RaschModel
from RaschModel import LearnRaschModel
import numpy as np

N = 100
Q = 10
a = np.random.randn(N,1)
b = np.random.randn(Q,1)
rasch = RaschModel(a, b)
Y = rasch.sample()

a_est, b_est, num_iter = LearnRaschModel(solver='newton').fit(Y)

print(a)
print(a_est)
