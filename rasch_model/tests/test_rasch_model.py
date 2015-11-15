import sys
sys.path.append('../rasch_model')

from rasch_model.RaschModel import RaschModel
from rasch_model.RaschModel import LearnRaschModel

from nose.tools import assert_equal
from nose.tools import assert_true

import numpy as np


def test_raschmodel_init():
    # Test RaschModel
    rm = RaschModel([0.3, 0.4], [0.2, -0.3, 0.2])

    # test length of the vectors
    assert_equal(rm.N, 2)
    assert_equal(rm.Q, 3)

    # test if samples are 0 or 1
    sm = rm.sample().flatten()
    assert_equal(np.all((sm == 0) | (sm == 1)),
                 True)


def test_learn_rasch_model():
    # Test Rasch Model Learning for a simple example

    a = [-0.5, 0.5]
    b = np.random.randn(500, 1)
    rm = RaschModel(a, b)
    Y = RaschModel(a, b).sample()

    lrm = LearnRaschModel(verbose=False, max_iter_inner=2, model='rasch')
    lrm.fit(Y)
    a_est = lrm.get_user()
    print a_est
    assert_true((a_est[0] < 0) & (a_est[1] > 0))
    assert_true(np.abs(np.sum(a_est)) < 0.00001)
