"""
Rasch Model
"""

# Author: Divyanshu Vats <vats.div@gmail.com>

import numpy as np
import pandas as pd


class RaschModel:
    """ Rasch Model for modeling a binary matrix

    Main Assumption:
        P(y_{ij}) = exp(y_{ij}(a_i + b_j) / (1 + exp(a_i + b_j)),
    where y_{ij} in {0,1}

    Paramters
    ---------
    a : user level real valued Nx1 array
    b : item level real valued Qx1 array

    Attributes
    ----------
    N : number of users
    Q : number of items
    M : matrix formed using a and b that gets passed to the logit function
    """

    def __init__(self, a, b):
        self.N = len(a)
        self.Q = len(b)
        self.a = np.reshape(a, (len(a), 1))
        self.b = np.reshape(b, (len(b), 1))

    def get_user(self):
        return self.a

    def get_item(self):
        return self.b

    def sample(self):
        """ Sample a QxN matrix from the Rasch Model
        Note: not memory efficient if Q or N are large
        Mainly used for testing
        Returns
        -------
        N x Q binary matrix
        """
        M = (self.a.dot(np.ones((1, self.Q))) +
             np.ones((self.N, 1)).dot(self.b.T))

        def logit(x): return 1 / (1+np.exp(-x))
        return (logit(M) > np.random.rand(self.N, self.Q)) * 1


class LearnRaschModel:
    """Alternating descent methods for learning Rasch Model parameters

    Parameters
    ----------
    max_iter_inner : int (default : 100)
        Maximum number of iterations in the inner computations when solving a
        logistic regression problem

    max_iter_outer : int (default : 10)
        Maximum number of iterations in the outer computations of alternating
        between the computations of a and b

    gamma : float (default : 1.0)
        shrinkage parameter for gradient descent

    tol_inner : float (default : 1e-5)
        Stopping criteria in the inner computations

    tol_outer : float (default : 1e-5)
        Stopping criteria in the inner computations

    mu : float (default : 0.0)
        The standard Rasch model parameter learning problem is ill-posed.
        Thus, we need to impose some condition for the uniqueness of the
        solution.  In the computations, we assume that mean(a) = mu, where a
        are the user level parameters

    solver : {'gradient', 'newton'}
        Algorithm to use for optimization

    verbose: boolean (default: True)
        if True, then prints iterations
    """

    def __init__(self, max_iter_inner=100, max_iter_outer=10, gamma=1.0,
                 tol_inner=1e-5, tol_outer=1e-5, mu=0.0,
                 solver='gradient', verbose=True):
        self.max_iter_inner = max_iter_inner
        self.max_iter_outer = max_iter_outer
        self.gamma = gamma
        self.tol_inner = tol_inner
        self.tol_outer = tol_outer
        self.mu = mu
        self.verbose = verbose
        self.solver = solver

    def fit(self, data=None, user_id=None, item_id=None,
            response=None, sum_user=None, sum_item=None):
        """ Fit the model Rasch model given data

        Parameters
        ----------
        data : Either:
                 1) ndarray, shape N x Q, where N is the number
                    of users and Q is the number of items.
                 2) pandas DataFrame
        user_id : if data is a DataFrame, then column name of user id
        item_id : if data is a DataFrame, then column name of item id
        response : if data is a DataFrame, then column with response

        Returns
        -------
        RaschModel object
        """

        # compute sufficient statistics
        if ((sum_user is None) and (sum_item is None)):
            sum_user, sum_item = _get_item_user_sums(data, user_id,
                                                     item_id, response)

        a_est, b_est, n_iter = _learn_rasch(np.array(sum_item.values()),
                                            np.array(sum_user.values()),
                                            self.max_iter_inner,
                                            self.max_iter_outer, self.gamma,
                                            self.tol_inner, self.tol_outer,
                                            self.mu, self.verbose, self.solver)

        self.rm = RaschModel(a_est, b_est)
        self.num_iter = n_iter

        a_est = dict(zip(sum_user.keys(), a_est))
        b_est = dict(zip(sum_item.keys(), b_est))

        return a_est, b_est

    def get_user(self):
        """ Return user level parameters
        """
        return self.rm.get_user()

    def get_item(self):
        """ Return item level parameters
        """
        return self.rm.get_item()


def _get_item_user_sums(data, user_id, item_id, response):

    if isinstance(data, np.ndarray):
        sum_item = dict(pd.Series(np.nansum(data, axis=0)))
        sum_user = dict(pd.Series(np.nansum(data, axis=1)))

    if isinstance(data, pd.DataFrame):
        sum_item = dict(data[[user_id, item_id, response]].
                        groupby(item_id)[response].
                        sum())
        sum_user = dict(data[[user_id, item_id, response]].
                        groupby(user_id)[response].
                        sum())

    return sum_user, sum_item


def _rasch_alternating(mu, b, dim, gamma, max_iter, tol, solver):
    """
    Logistic regression when fixing one of the Rasch model parameters
    """
    a_old = 0
    a_new = 0
    gamma = gamma / dim

    def gradient(a):
        return (mu - np.nansum(1 / (1+np.exp(-(a + b)))))

    def hessian(a):
        tmp = -np.nansum(1 / ((1 + np.exp(-(a + b))) * ((1 + np.exp(a + b)))))
        return(tmp)

    tolerance = tol + 1
    i = 0

    if (solver == 'gradient'):
        while ((tolerance > tol) & (i < max_iter)):
            a_new = a_old + gamma * gradient(a_old)
            tolerance = np.abs(a_new - a_old) / np.abs(a_new + 1e-10)
            a_old = a_new
            i = i + 1

    if (solver == 'newton'):
        while ((tolerance > tol) & (i < max_iter)):
            a_new = a_old - gradient(a_old) / hessian(a_old)
            tolerance = np.abs(a_new - a_old) / np.abs(a_new + 1e-10)
            a_old = a_new
            i = i + 1

    return a_new


def _run_alt(sum_item, sum_user, N, Q, b_est,
             gamma, max_iter_inner, tol, solver):
    """
    Run the alternating minimization code
    """

    a_est = np.array([_rasch_alternating(sum_user[k], b_est, Q, gamma,
                                         max_iter_inner, tol, solver)
                      for k in range(0, N)])

    # makes the problem well-posed
    # TODO: make this a user defined input with some
    #       other possible initial conditions
    a_est = a_est - np.nanmean(a_est[~np.isinf(a_est)])
    b_est = np.array([_rasch_alternating(sum_item[k], a_est, N, gamma,
                                         max_iter_inner, tol, solver)
                      for k in range(0, Q)])

    return a_est, b_est


def _learn_rasch(sum_item, sum_user, max_iter_inner, max_iter_outer, gamma,
                 tol_inner, tol_outer, mu, verbose, solver):
    """
    Main function for computing the Rasch model parameters
    """

    sum_item = sum_item.astype(float)
    sum_user = sum_user.astype(float)
    N = len(sum_user)
    Q = len(sum_item)
    b_init = np.zeros(((1, Q)))
    b_init = (sum_item - np.nanmean(sum_item)) / np.nanstd(sum_item)
    a_old, b_old = _run_alt(sum_item, sum_user, N, Q, b_init, gamma,
                            max_iter_inner, tol_inner, solver)
    tolerance = 1 + tol_outer

    def _compute_tolerance(a, b):
        ind = (~np.isinf(a) & ~np.isinf(b))
        return np.linalg.norm(a[ind] - b[ind]) / np.linalg.norm(a[ind] + 1e-10)

    i = 1
    while ((tolerance > tol_outer) & (i < max_iter_outer)):
        if (verbose):
            print("Iteration: " + str(i))
        a_new, b_new = _run_alt(sum_item, sum_user, N, Q, b_old, gamma,
                                max_iter_inner, tol_inner, solver)
        tolerance = _compute_tolerance(a_new, a_old) +\
            _compute_tolerance(b_new, b_old)
        b_old = b_new
        a_old = a_new
        i = i + 1

    return a_new, b_new, i
