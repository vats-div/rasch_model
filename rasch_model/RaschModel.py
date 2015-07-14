"""
Rasch Model
"""

# Author: Divyanshu Vats <vats.div@gmail.com>

import numpy as np
import pandas as pd
from operator import itemgetter
import argparse
import time
from sklearn.metrics import roc_auc_score


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

        return (_logit(M) > np.random.rand(self.N, self.Q)) * 1


class LearnRaschModel:
    """Alternating descent methods for learning Rasch Model parameters

    Parameters
    ----------
    max_iter_inner : int (default : 1)
        Maximum number of iterations in the inner computations when solving a
        logistic regression problem

    max_iter : int (default : 30)
        Maximum number of iterations in the outer computations of alternating
        between the computations of a and b

    alpha : float (default : 0.0)
        regularization parameter

    gamma : float (default : 1.0)
        shrinkage parameter for gradient descent

    tol : float (default : 1e-5)
        Stopping criteria in the inner computations

    mu : float (default : 0.0)
        The standard Rasch model parameter learning problem is ill-posed.
        Thus, we need to impose some condition for the uniqueness of the
        solution.  In the computations, we assume that mean(a) = mu, where a
        are the user level parameters

    seed : int (default : 1.0)

    verbose: boolean (default: True)
        if True, then prints iterations

    init: string (default: random)
          initialization to use, either random or zeros

    model: str (default: rasch)
           type of model to fit.  either 'rasch' or '2PL'
    """

    def __init__(self, max_iter_inner=1, max_iter=30, alpha=0.0,
                 gamma=1.0, tol=1e-5, mu=0.0, seed=1,
                 verbose=False, init='random', model='rasch'):
        self.max_iter_inner = max_iter_inner
        self.max_iter = max_iter
        self.gamma = gamma
        self.tol = tol
        self.mu = mu
        self.verbose = verbose
        self.seed = seed
        self.alpha = alpha
        self.init = init
        self.model = model

    def fit(self, data=None, user_id=0, item_id=1, response=2,
            sum_user=None, sum_item=None):
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

        time_begin = time.time()

        # save dataframe
        self.data = _convert_to_dataframe(data)

        # compute sufficient statistics if not already available
        if ((sum_user is None) and (sum_item is None)):
            self.user_id = user_id
            self.item_id = item_id
            self.response = response
            sum_user, sum_item = self._get_item_user_sums(self.data)

        self.N, self.Q = (len(sum_user), len(sum_item))

        self._save_params()

        # get user and item indices and store as
        # dataframe of observed indices
        # self.obser has columns 'index_user' and 'index_item'
        self.obser = self._get_observed_indices(sum_user, sum_item)

        a_est, b_est, s_est = self._learn_rasch(np.array(sum_user.values()),
                                                np.array(sum_item.values()))
        time_end = time.time()

        # save model parameters as a dictionary
        self.time_taken = time_end - time_begin
        self.a_est = dict(zip(sum_user.keys(), a_est))
        self.b_est = dict(zip(sum_item.keys(), b_est))

        # if model is 2PL, then save s_est
        if self.model is '2PL':
            self.s_est = dict(zip(sum_item.keys(), s_est))

    def _save_params(self):
        """ save model parameters
        """

        # initialize user and item gradients
        self.grad_user = np.zeros((self.N, 1))
        self.grad_item = np.zeros((self.Q, 1))

        if self.model is '2PL':
            self.grad_slope = np.zeros((self.Q, 1))

    def recommend(self, user_key, k=5):

        # compute probability of each item
        tmp = dict(zip(self.b_est.keys(),
                       _logit(self.a_est[user_key] + self.b_est.values())))

        return sorted(tmp.items(), key=itemgetter(1), reverse=True)[:k]

    def _learn_rasch(self, sum_user, sum_item):
        """
        Main function for computing the Rasch model parameters
        """

        # ensure right dimensions, i.e., column vectors
        sum_user = _fc(sum_user.astype(float))
        sum_item = _fc(sum_item.astype(float))

        # set random seed
        np.random.seed(self.seed)

        # initialization
        if self.init is 'random':
            a_old = np.random.normal(size=(self.N, 1))
            b_old = np.random.normal(size=(self.Q, 1))
        else:
            a_old = np.zeros((self.N, 1))
            b_old = np.zeros((self.Q, 1))

        # initialize s_old to None
        if self.model is '2PL':
            s_old = np.ones((self.Q, 1))
        else:
            s_old = None

        # don't really need s_old since s is initialized to all ones
        logL_old = self.likelihood(a_old, b_old)

        for i in range(self.max_iter):
            a_new, b_new, s_new = self._run_alt(sum_item, sum_user,
                                                a_old, b_old, s_old)

            # need to update sum_user and sum_item for 2PL
            if self.model is '2PL':
                sum_user, sum_item = self._update_item_user_sums(s_new)

            logL_new = self.likelihood(a_new, b_new, s_new)
            if (self.verbose):
                print("Iteration: " + str(i) + ", logL: " + str(logL_new))
            if ((logL_new - logL_old) < 0):
                break
            b_old = b_new
            a_old = a_new
            s_old = s_new
            logL_old = logL_new

        # save total number of iterations
        self.num_iter = i

        if self.model is '2PL':
            s_old = np.reshape(s_old, self.Q)

        return np.reshape(a_old, self.N), np.reshape(b_old, self.Q), s_old

    def _run_alt(self, sum_item, sum_user, a_est, b_est, s_est=None):
        """
        Run the alternating minimization code
        """

        a_est = self._rasch_alternating(sum_user, a_est, b_est, s_est, True)
        # makes the problem well-posed
        a_est = a_est - np.nanmean(a_est[~np.isinf(a_est)]) + self.mu

        b_est = self._rasch_alternating(sum_item, b_est, a_est, s_est, False)

        if self.model is '2PL':
            s_est = self._compute_slope(s_est, a_est, b_est)
            s_est = np.float(self.Q) * s_est / np.linalg.norm(s_est, 1)

        return a_est, b_est, s_est

    def _rasch_alternating(self, sum_y, a_est, b, s_est, user):
        """
        Logistic regression when fixing one of the Rasch model parameters
        """
        if user:
            grad_sum = self.grad_user
            ind1 = 'index_user'
            ind2 = 'index_item'
        else:
            grad_sum = self.grad_item
            ind1 = 'index_item'
            ind2 = 'index_user'

        grad_func = self._exact_gradient

        for i in range(self.max_iter_inner):
            grad_temp = grad_func(sum_y, a_est, b, s_est, ind1, ind2)
            grad_sum += np.square(grad_temp)
            a_est = a_est + self.gamma * grad_temp / np.sqrt(grad_sum)

        if user:
            self.grad_user = grad_sum
        else:
            self.grad_item = grad_sum

        return a_est

    def _exact_gradient(self, sum_y, a, b, s, ind1, ind2):
        """ Exact gradient, but may be memory intensive for large data """

        if s is not None:
            s_temp = s[self.obser['index_item']]
        else:
            s_temp = 1.0

        self.obser['val'] = _logit(a[self.obser[ind1]] +
                                   b[self.obser[ind2]]) * s_temp

        tmp = self.obser.groupby(ind1)['val'].sum().values

        return (sum_y - _fc(tmp)) - self.alpha * 2 * a

    def _compute_slope(self, s_est, a_est, b_est):
	""" Compute the slope parameter per item
        """

        def _slope_grad():
            """
            \sum_{i} [y_{ij}(a_i+b_j) - logit(s_j(a_i+b_j)) * (a_i+b_j)]
            """
            sum_ab = (a_est[self.obser['index_user']] +
                      b_est[self.obser['index_item']])
            sum_abs = s_est[self.obser['index_item']] * sum_ab
            self.obser['val'] = _fc(self.data[self.response].values) *\
                                sum_ab - _logit(sum_abs) * sum_ab
            tmp = self.obser.groupby('index_item')['val'].sum().values
            return _fc(tmp)

        grad_temp = _slope_grad()
        self.grad_slope = self.grad_slope + np.square(grad_temp)

        return (s_est +
                1.0 * self.gamma * grad_temp / np.sqrt(self.grad_slope))

    def likelihood(self, a=None, b=None, s=None):

        # \sum_{i,j} y_{i,j} s_j(a_i + b_j) - log(1 + exp(s_j(a_i + b_j)))
        if ((a is None) and (b is None) and (s is None)):
            a = np.array(self.a_est.values())
            b = np.array(self.b_est.values())

            if self.model is '2PL':
                s = np.array(self.a_est.values())

        c = a[self.obser['index_user']] + b[self.obser['index_item']]

        if (self.model is '2PL') and (s is not None):
            c = s[self.obser['index_item']] * c

        first_term = np.nansum(c[self.data[self.response].values > 0])
        second_term = np.nansum(np.log(1 + np.exp(c)))

        return (first_term - second_term -
                self.alpha * np.sum(a*a) - self.alpha * np.sum(b*b))

    def predict(self, data, user_id, item_id):

        pr = np.zeros((len(data), 1))
        a_est = self.a_est
        b_est = self.b_est
        mean_a = np.mean(a_est.values())
        mean_b = np.mean(b_est.values())

        for i in range(len(data)):
            if data.ix[i][user_id] in a_est:
                a_temp = a_est[data.ix[i][user_id]]
            else:
                a_temp = mean_a
            if data.ix[i][item_id] in b_est:
                b_temp = b_est[data.ix[i][item_id]]
            else:
                b_temp = mean_b

            pr[i] = _logit(a_temp + b_temp)

        return pr

    def get_user(self):
        """ Return user level parameters
        """
        return self.a_est.values()

    def get_item(self):
        """ Return item level parameters
        """
        return self.b_est.values()

    def _get_observed_indices(self, sum_user, sum_item):
        """ Return a dataframe with columns index_user, index_item, and 'response'
        """

        user_index_df = pd.DataFrame(sum_user.keys()).reset_index().\
                        rename(columns={'index': 'index_user', 0: 'val_user'})
        item_index_df = pd.DataFrame(sum_item.keys()).reset_index().\
                        rename(columns={'index': 'index_item', 0: 'val_item'})

        return pd.merge(item_index_df,
                        self.data[[self.user_id, self.item_id, self.response]],
                        left_on='val_item', right_on=self.item_id).\
                  merge(user_index_df, right_on='val_user',
                        left_on=self.user_id)[['index_user', 'index_item']]

    def _get_item_user_sums(self, data):

        sum_item = dict(data[[self.user_id, self.item_id, self.response]].
                        groupby(self.item_id)[self.response].
                        sum())
        sum_user = dict(data[[self.user_id, self.item_id, self.response]].
                        groupby(self.user_id)[self.response].
                        sum())

        return sum_user, sum_item

    def _update_item_user_sums(self, s):

        tmp_data = self.data.copy()
        tmp_data[self.response] = (tmp_data[self.response].values *
                                   s[self.obser['index_item']].flatten())
        sum_user, sum_item = self._get_item_user_sums(tmp_data)

        return _fc(sum_user.values()), _fc(sum_item.values())


def _c_tol(a, b):
    ind = (~np.isinf(a) & ~np.isinf(b))
    return np.linalg.norm(a[ind] - b[ind]) / np.linalg.norm(a[ind] + 1e-20)


def _logit(a):
    return (1.0 / (1.0 + np.exp(-a)))


def _fc(x):
    """ force a column vector """
    return np.expand_dims(x, 1)


def _convert_to_dataframe(data):
    """ Convert np.array to data frame """

    if isinstance(data, np.ndarray):
        u_ind, i_ind = np.meshgrid(range(data.shape[0]),
                                   range(data.shape[1]))
        u_ind = u_ind.flatten()
        i_ind = i_ind.flatten()
        df_new = pd.DataFrame()
        df_new[0] = u_ind
        df_new[1] = i_ind
        df_new[2] = data[u_ind, i_ind]
    else:
        df_new = data

    return df_new


def main(ns):
    file_train = ns.train
    file_test = ns.test
    train_data = pd.read_table(file_train, header=ns.header, sep=ns.sep)

    if ns.user_id is None:
        ns.user_id = 0
        ns.item_id = 1
        ns.response = 2

    train_data[ns.response] = (train_data[ns.response] > 0) * 1
    test_data = pd.read_table(file_test, header=ns.header, sep=ns.sep)

    lrm = LearnRaschModel(max_iter=ns.max_iter, verbose=True,
                          gamma=ns.gamma, alpha=ns.alpha)
    lrm.fit(train_data, user_id=ns.user_id, item_id=ns.item_id,
            response=ns.response)

    pr = lrm.predict(test_data, user_id=ns.user_id, item_id=ns.item_id)
    print roc_auc_score(test_data[ns.response], pr)

    test_data["irt.urlbuy"] = pr
    print "Saving CSV file to " + ns.output
    test_data.to_csv(ns.output, index=False)

if __name__ == '__main__':

    lrm = LearnRaschModel()
    parser = argparse.ArgumentParser()
    parser.add_argument('--train',
                        help="path to training data",
                        required=True)
    parser.add_argument('--test',
                        help="path to testing data",
                        required=True),
    parser.add_argument('--user_id',
                        help="name of the column that contains user id",
                        required=False),
    parser.add_argument('--item_id',
                        help="name of the column that contains item id",
                        required=False),
    parser.add_argument('--response',
                        help="name of the column that contains the response",
                        required=False),
    parser.add_argument('--output',
                        help="name of file to save the user, item\
                        probabilities",
                        required=True)
    parser.add_argument('--header',
                        help="row number in data that corresponds to header",
                        nargs='?', default=0, type=int)
    parser.add_argument('--sep',
                        help="separator in train/test data that distringuishes\
                        columns",
                        nargs='?', default=',')
    parser.add_argument('--gamma',
                        help="step size for gradient descent\
                        columns",
                        nargs='?', default=1.0, type=float)
    parser.add_argument('--max_iter',
                        help="maximum number of iterations\
                        columns",
                        nargs='?', default=30, type=int)
    parser.add_argument('--alpha',
                        help="regularization parameter\
                        columns",
                        nargs='?', default=0.0, type=float)
    ns = parser.parse_args()
    main(ns)
