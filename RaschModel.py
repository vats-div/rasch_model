"""
Rasch Model
"""

# Author: Divyanshu Vats <vats.div@gmail.com>

import numpy as np


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
        self.a = a
        self.b = b
        self.N = len(a)
        self.Q = len(b)
        self.M = a.dot(np.ones((1, self.Q))) + np.ones((self.N, 1)).dot(b.T)

    def sample(self):
        """ Sample a QxN matrix from the Rasch Model
        Returns
        -------
        N x Q binary matrix
        """
        def logit(x): return 1 / (1+np.exp(-x))
        return (logit(self.M) > np.random.rand(self.N, self.Q)) * 1


class LearnRaschModel:
    """Alternating descent method for learning Rasch Model parameters

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
    verbose: boolean (default: True)
        if True, then prints iterations
    """

    def __init__(self, max_iter_inner=100, max_iter_outer=10, gamma=1.0,
                 tol_inner=1e-5, tol_outer=1e-5, mu=0.0, verbose=True):
        self.max_iter_inner = max_iter_inner
        self.max_iter_outer = max_iter_outer
        self.gamma = gamma
        self.tol_inner = tol_inner
        self.tol_outer = tol_outer
        self.mu = mu
        self.verbose = verbose

    def fit(self, Y):
        """ Fit the model Rasch model given training data Y

        Parameters
        ----------
        Y : numpy array of shape N x Q

        Returns
        -------
        RaschModel object
        """
        a_est, b_est, n_iter = _learn_rasch(Y, self.max_iter_inner,
                                            self.max_iter_outer, self.gamma,
                                            self.tol_inner, self.tol_outer,
                                            self.mu, self.verbose)
        return a_est, b_est, n_iter


def _rasch_alternating(Y, b, k, gamma, max_iter, tol):
    """
    Logistic regression when fixing one of the Rasch modep parameters
    """
    a_old = 0
    a_new = 0
    mu = np.sum(Y[k, :])
    gamma = gamma / np.shape(Y)[1]

    def gradient(a): return (mu - np.sum(1 / (1+np.exp(-(a + b)))))
    tolerance = tol + 1
    i = 0
    while ((tolerance > tol) & (i < max_iter)):
        a_new = a_old + gamma * gradient(a_old)
        tolerance = np.abs(a_new - a_old) / np.abs(a_new + 1e-5)
        a_old = a_new
        i = i + 1
    return a_new


def _run_alt(Y, N, Q, b_est, gamma, max_iter_inner, tol):
    """
    Run the alternating minimization code
    """
    a_est = np.array([_rasch_alternating(Y, b_est, k, gamma,
                                         max_iter_inner, tol)
                      for k in range(0, N)])
    a_est = a_est - np.mean(a_est)
    b_est = np.array([_rasch_alternating(Y.T, a_est, k, gamma,
                                         max_iter_inner, tol)
                      for k in range(0, Q)])
    return a_est, b_est


def _learn_rasch(Y, max_iter_inner, max_iter_outer, gamma, tol_inner,
                 tol_outer, mu, verbose):
    """
    Main function for computing the Rasch model parameters
    """

    N, Q = np.shape(Y)
    b_init = np.zeros((1, Q))
    a_old, b_old = _run_alt(Y, N, Q, b_init, gamma, max_iter_inner, tol_inner)
    tolerance = 1 + tol_outer

    def compute_tol(x, y): return np.linalg.norm(x-y) / np.linalg.norm(x)
    i = 1
    if (verbose):
        print("Iteration: " + str(i))
    while ((tolerance > tol_outer) & (i < max_iter_outer)):
        a_new, b_new = _run_alt(Y, N, Q, b_old, gamma, max_iter_inner,
                                tol_inner)
        tolerance = compute_tol(a_new, a_old) + compute_tol(b_new, b_old)
        b_old = b_new
        a_old = a_new
        i = i + 1
        if (verbose):
            print("Iteration: " + str(i))

    return a_new, b_new, i
