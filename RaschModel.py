"""
Rasch Model
"""

# Author: Divyanshu Vats <vats.div@gmail.com>

import numpy as np
import pandas as pd
from operator import itemgetter
import time

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

    tol : float (default : 1e-5)
        Stopping criteria in the inner computations

    mu : float (default : 0.0)
        The standard Rasch model parameter learning problem is ill-posed.
        Thus, we need to impose some condition for the uniqueness of the
        solution.  In the computations, we assume that mean(a) = mu, where a
        are the user level parameters

    verbose: boolean (default: True)
        if True, then prints iterations
    """

    def __init__(self, max_iter_inner=5, max_iter_outer=30, 
                 gamma=1.0, tol=1e-5, mu=0.0,
                 solver='gradient', verbose=False):
        self.max_iter_inner = max_iter_inner
        self.max_iter_outer = max_iter_outer
        self.gamma = gamma
        self.tol = tol
        self.mu = mu
        self.verbose = verbose
        self.solver = solver

    def fit(self, data=None, user_id=None, item_id=None, response=None, 
            sum_user=None, sum_item=None, inplace=False):
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
        inplace : If True, then returns the parameters

        Returns
        -------
        RaschModel object
        """

        time_begin = time.time()
        
        # compute sufficient statistics if not available
        if ((sum_user is None) and (sum_item is None)):
            self.data = data
            self.user_id = user_id
            self.item_id = item_id
            self.response = response
            sum_user, sum_item = _get_item_user_sums(data, user_id,
                                                     item_id, response)
        self.N, self.Q = (len(sum_user), len(sum_item))

        a_est, b_est, n_iter = self._learn_rasch(np.array(sum_user.values()),
                                                 np.array(sum_item.values()))

        time_end = time.time()
        
        # save model parameters
        self.time_taken = time_end - time_begin        
        self.a_est = dict(zip(sum_user.keys(), a_est))
        self.b_est = dict(zip(sum_item.keys(), b_est))         
        self.num_iter = n_iter
        
        if (inplace == False):
            return self.a_est, self.b_est
    
    def recommend(self, user_key, k=5):
        
        # compute probability of each item
        tmp = dict(zip(self.b_est.keys(),
                       1 / (1+np.exp(-(self.a_est[user_key] + 
                                       self.b_est.values())))))

        return sorted(tmp.items(), key=itemgetter(1), reverse=True)[:k]

    def _learn_rasch(self, sum_user, sum_item):
        """
        Main function for computing the Rasch model parameters
        """
        sum_user = sum_user.astype(float)    
        sum_item = sum_item.astype(float)
        
        def norm_vector(x): return (x - np.nanmean(x)) / np.nanstd(x)
        
        a_old, b_old = norm_vector(sum_user), norm_vector(sum_item)
            
        for i in range(self.max_iter_outer):
            a_new, b_new = self._run_alt(sum_item, sum_user, a_old, b_old)
            tol = _c_tol(a_new, a_old) + _c_tol(b_new, b_old)
            if (tol < self.tol): break
            b_old = b_new
            a_old = a_new
            if (self.verbose):
                print("Iteration: " + str(i) + ", tolerance: " + str(tol))

        return a_new, b_new, i

    def _run_alt(self, sum_item, sum_user, a_est, b_est):
        """
        Run the alternating minimization code
        """

        a_est = np.array([self._rasch_alternating(sum_user[k], a_est[k], b_est)
                          for k in range(0, self.N)])

        # makes the problem well-posed
        a_est = a_est - np.nanmean(a_est[~np.isinf(a_est)]) +  self.mu
        b_est = np.array([self._rasch_alternating(sum_item[k], b_est[k], a_est)
                          for k in range(0, self.Q)])
    
        return a_est, b_est

    def _rasch_alternating(self, sum_y, a_est, b):
        """
        Logistic regression when fixing one of the Rasch model parameters
        """

        grad_sum = 0

        def gradient(a):
            return (sum_y - np.nansum(1 / (1 + np.exp(-(a + b)))))
        
        for i in range(self.max_iter_inner):
            grad_temp = gradient(a_est)
            grad_sum += np.square(grad_temp)
            a_est = a_est + self.gamma * gradient(a_est) / np.sqrt(grad_sum)

        return a_est
     
    def likelihood(self):
        
        # \sum_{i,j} y_{i,j} (a_i + b_j) - log(1 + exp(a_i + b_j))
        ind_u = self.data[self.user_id]
        ind_i = self.data[self.item_id]
        def compute_sum(k): 
            return self.a_est[ind_u[k]] + self.b_est[ind_i[k]]

        c = [compute_sum(k) for k in range(len(self.data))]
        return np.sum(c * self.data[self.response] - np.log(1 + np.exp(c)))

    def get_user(self):
        """ Return user level parameters
        """
        return self.a_est.values()

    def get_item(self):
        """ Return item level parameters
        """
        return self.b_est.values()
    

def _c_tol(a, b):
    ind = (~np.isinf(a) & ~np.isinf(b))
    return np.linalg.norm(a[ind] - b[ind]) / np.linalg.norm(a[ind] + 1e-20)


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


