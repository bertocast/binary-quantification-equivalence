import cvxpy
import quadprog
import numpy as np
import math
import numbers
from sklearn.utils import check_X_y
from sklearn.utils.validation import check_array, check_consistent_length


############
#  Functions for solving optimization problems with different loss functions
############
def solve_l1(train_distrib, test_distrib, n_classes, solver='ECOS'):
    """ Solves AC, PAC, PDF and Friedman optimization problems for L1 loss function

        min   |train_distrib * prevalences - test_distrib|
        s.t.  prevalences_i >=0
              sum prevalences_i = 1

        Parameters
        ----------
        train_distrib : array, shape depends on the optimization problem
            Represents the distribution of each class in the training set
            PDF: shape (n_bins * n_classes, n_classes)
            AC, PAC, Friedman: shape (n_classes, n_classes)

        test_distrib : array, shape depends on the optimization problem
            Represents the distribution of the testing set
            PDF: shape shape (n_bins * n_classes, 1)
            AC, PAC, Friedman: shape (n_classes, 1)

        n_classes : int
            Number of classes

        solver : str, (default='ECOS')
            The solver used to solve the optimization problem. The following solvers have been tested:
            'ECOS', 'ECOS_BB', 'CVXOPT', 'GLPK', 'GLPK_MI', 'SCS' and 'OSQP', but it seems that 'CVXOPT' does not
            work

        Returns
        -------
        prevalences : array, shape=(n_classes, )
           Vector containing the predicted prevalence for each class
    """
    prevalences = cvxpy.Variable(n_classes)
    objective = cvxpy.Minimize(cvxpy.norm(np.squeeze(test_distrib) - train_distrib * prevalences, 1))

    contraints = [cvxpy.sum(prevalences) == 1, prevalences >= 0]

    prob = cvxpy.Problem(objective, contraints)
    prob.solve(solver=solver)
    return np.array(prevalences[0:n_classes].value).squeeze()


def solve_l2cvx(train_distrib, test_distrib, n_classes, solver='ECOS'):
    prevalences = cvxpy.Variable(n_classes)
    objective = cvxpy.Minimize(cvxpy.sum_squares(train_distrib * prevalences - test_distrib))

    contraints = [cvxpy.sum(prevalences) == 1, prevalences >= 0]

    prob = cvxpy.Problem(objective, contraints)
    prob.solve(solver=solver)
    return np.array(prevalences[0:n_classes].value).squeeze()


def solve_l2(train_distrib, test_distrib, G, C, b):
    """ Solves AC, PAC, PDF and Friedman optimization problems for L2 loss function

        min    (test_distrib - train_distrib * prevalences).T (test_distrib - train_distrib * prevalences)
        s.t.   prevalences_i >=0
               sum prevalences_i = 1

        Expanding the objective function, we obtain:

        prevalences.T train_distrib.T train_distrib prevalences
        - 2 prevalences train_distrib.T test_distrib + test_distrib.T test_distrib

        Notice that the last term is constant w.r.t prevalences.

        Let G = 2 train_distrib.T train_distrib  and a = 2 * train_distrib.T test_distrib, we can use directly
        quadprog.solve_qp because it solves the following kind of problems:

        Minimize     1/2 x^T G x - a^T x
        Subject to   C.T x >= b

        `solve_l2` just computes the term a, shape (n_classes,1), and then calls quadprog.solve_qp.
        G, C and b were computed by `compute_l2_param_train` before, in the 'fit' method` of the PDF/Friedman object

        Parameters
        ----------
        train_distrib : array, shape depends on the optimization problem
            Represents the distribution of each class in the training set
            PDF: shape (n_bins * n_classes, n_classes)
            AC, PAC Friedman: shape (n_classes, n_classes)

        test_distrib : array, shape depends on the optimization problem
            Represents the distribution of the testing set
            PDF: shape shape (n_bins * n_classes, 1)
            AC, PAC, Friedman: shape (n_classes, 1)

        G : array, shape (n_classes, n_classes)

        C : array, shape (n_classes, n_constraints)
            n_constraints will be n_classes + 1

        b : array, shape (n_constraints,)

        G, C and b are computed by `compute_l2_param_train` in the 'fit' method

        Returns
        -------
        prevalences : array, shape=(n_classes, )
           Vector containing the predicted prevalence for each class
    """
    a = 2 * train_distrib.T.dot(test_distrib)
    a = np.squeeze(a)
    prevalences = quadprog.solve_qp(G=G, a=a, C=C, b=b, meq=1)
    return prevalences[0]


def compute_l2_param_train(train_distrib, classes):
    """ Computes params related to the train distribution for solving PDF optimization problems using
        L2 loss function

        Parameters
        ----------
        train_distrib : array, shape (n_bins * n_classes, n_classes)
            Represents the distribution of each class in the training set

        classes : ndarray, shape (n_classes, )
            Class labels

        Returns
        -------
        G : array, shape (n_classes, n_classes)

        C : array, shape (n_classes, n_constraints)
            n_constraints will be n_classes + 1  (n_classes constraints to guarantee that prevalences_i>=0, and
            an additional constraints for ensuring that sum(prevalences)==1

        b : array, shape (n_constraints,)

        quadprog.solve_qp solves the following kind of problems:

        Minimize     1/2 x^T G x  a^T x
        Subject to   C.T x >= b

        Thus, the values of G, C and b must be the following

        G = train_distrib.T train_distrib
        C = [[ 1, 1, ...,  1],
             [ 1, 0, ...,  0],
             [ 0, 1, 0,.., 0],
             ...
             [ 0, 0, ..,0, 1]].T
        C shape (n_classes+1, n_classes)
        b = [1, 0, ..., 0]
        b shape (n_classes, )
    """
    G = 2 * train_distrib.T.dot(train_distrib)
    if not is_pd(G):
        G = nearest_pd(G)
    #  constraints, sum prevalences = 1, every prevalence >=0
    n_classes = len(classes)
    C = np.vstack([np.ones((1, n_classes)), np.eye(n_classes)]).T
    b = np.array([1] + [0] * n_classes, dtype=np.float)
    return G, C, b


############
# Functions to check if a matrix is positive definite and to compute the nearest positive definite matrix
# if it is not
############
def nearest_pd(A):
    """ Find the nearest positive-definite matrix to input

        A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which credits [2].

        References
        ----------
        [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

        [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
        matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """
    B = (A + A.T) / 2
    _, s, V = np.linalg.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if is_pd(A3):
        return A3

    spacing = np.spacing(np.linalg.norm(A))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    indendity_matrix = np.eye(A.shape[0])
    k = 1
    while not is_pd(A3):
        mineig = np.min(np.real(np.linalg.eigvals(A3)))
        A3 += indendity_matrix * (-mineig * k ** 2 + spacing)
        k += 1

    return A3


def dpofa(m):
    """ Factors a symmetric positive definite matrix

        This is a version of the dpofa function included in quadprog library. Here, it is mainly used to check
        whether a matrix is positive definite or not

        Parameters
        ----------
        m : symmetric matrix, typically the shape is (n_classes, n_classes)
            The matrix to be factored. Only the diagonal and upper triangle are used

        Returns
        -------
        k : int,
            == 0  m is positive definite and the factorization has been completed
            >  0  the leading minor of order k is not positive definite

        r : array, an upper triangular matrix
            When k==0, the factorization is complete and r.T.dot(r) == m
            The strict lower triangle is unaltered (it is equal to the strict lower triangle of matrix m), so it
            could be different from 0.
   """
    r = np.array(m, copy=True)
    n = len(r)
    for k in range(n):
        s = 0.0
        if k >= 1:
            for i in range(k):
                t = r[i, k]
                if i > 0:
                    t = t - np.sum(r[0:i, i] * r[0:i, k])
                t = t / r[i, i]
                r[i, k] = t
                s = s + t * t
        s = r[k, k] - s
        if s <= 0.0:
            return k+1, r
        r[k, k] = np.sqrt(s)
    return 0, r


def is_pd(m):
    """ Checks whether a matrix is positive definite or not

        It is based on dpofa function, a version of the dpofa function included in quadprog library. When dpofa
        returns 0 the matrix is positive definite.

        Parameters
        ----------
        m : symmetric matrix, typically the shape is (n_classes, n_classes)
            The matrix to check whether it is positive definite or not

        Returns
        -------
        A boolean, True when m is positive definite and False otherwise

    """
    return dpofa(m)[0] == 0


############
# Functions for solving HD-based methods
############
def solve_hd(train_distrib, test_distrib, n_classes, solver='ECOS'):
    """ Solves the optimization problem for PDF methods using Hellinger Distance

        This method just uses cvxpy library

        Parameters
        ----------
        train_distrib : array, shape (n_bins * n_classes, n_classes)
            Represents the distribution of each class in the training set

        test_distrib : array, shape (n_bins * n_classes, 1)
            Represents the distribution of the testing set

        n_classes : int
            Number of classes

        solver : str, optional (default='ECOS')
            The solver to use. For example, 'ECOS', 'SCS', or 'OSQP'.

        Returns
        -------
        prevalences : array, shape=(n_classes, )
            Vector containing the predicted prevalence for each class
    """
    prevalences = cvxpy.Variable(n_classes)
    s = cvxpy.multiply(np.squeeze(test_distrib), train_distrib * prevalences)
    objective = cvxpy.Minimize(1 - cvxpy.sum(cvxpy.sqrt(s)))
    contraints = [cvxpy.sum(prevalences) == 1, prevalences >= 0]

    prob = cvxpy.Problem(objective, contraints)
    prob.solve(solver=solver)
    return np.array(prevalences.value).squeeze()


############
# Golden section Search
############
def golden_section_search(distance_func, mixture_func, test_distrib, tol, **kwargs):
    """ Golden section search

        Used by PDF and quantiles classes. Only useful for binary quantification
        Given a function `distance_func` with a single local minumum in the interval [0,1], `golden_section_search`
        returns the prevalence that minimizes the differente between the mixture training distribution and
        the testing distribution according to `distance_func`

        Parameters
        ----------
        distance_func : function
            This is the loss function minimized during the search

        mixture_func : function
            The function used to generated the training mixture distribution given a value for the prevalence

        test_distrib : array
            The distribution of the positive class. The exact shape depends on the representation (pdfs, quantiles...)

        tol : float
            The precision of the solution

        kwargs : keyword arguments
            Here we pass the set of arguments needed by mixture functions: mixture_two_pdfs (for pdf-based classes) and
            compute quantiles (for quantiles-based classes). See the help of this two functions

        Returns
        -------
        prevalences : array, shape(2,)
           The predicted prevalence for the negative and the positive class
    """
    #  uncomment the following line for checking whether the distance function is V-shape or not
    # if not is_V_shape(distance_func, mixture_func, test_distrib, 0.02, False, **kwargs):
    #     print(distance_func, mixture_func)
    #     is_V_shape(distance_func, mixture_func, test_distrib, 0.02, True, **kwargs)

    # some constants
    invphi = (math.sqrt(5) - 1) / 2  # 1/phi
    invphi2 = (3 - math.sqrt(5)) / 2  # 1/phi^2

    # v_shape, best_p = is_V_shape(distance_func, mixture_func, test_distrib, 0.01, False, **kwargs)
    # a = best_p - 0.01
    # b = best_p + 0.01

    a = 0
    b = 1

    h = b - a

    # required steps to achieve tolerance
    n = int(math.ceil(math.log(tol / h) / math.log(invphi)))

    c = a + invphi2 * h
    d = a + invphi * h

    train_mixture_distrib = mixture_func(prevalence=c, **kwargs)
    fc = distance_func(train_mixture_distrib, test_distrib)
    train_mixture_distrib = mixture_func(prevalence=d, **kwargs)
    fd = distance_func(train_mixture_distrib, test_distrib)

    for k in range(n - 1):
        if fc < fd:
            b = d
            d = c
            fd = fc
            h = invphi * h
            c = a + invphi2 * h
            train_mixture_distrib = mixture_func(prevalence=c, **kwargs)
            fc = distance_func(train_mixture_distrib, test_distrib)

        else:
            a = c
            c = d
            fc = fd
            h = invphi * h
            d = a + invphi * h
            train_mixture_distrib = mixture_func(prevalence=d, **kwargs)
            fd = distance_func(train_mixture_distrib, test_distrib)

    if fc < fd:
        return np.array([1 - (a + d) / 2, (a + d) / 2])
    else:
        return np.array([1 - (c + b) / 2, (c + b) / 2])


def mixture_two_pdfs(prevalence=None, pos_distrib=None, neg_distrib=None):
    """ Mix two pdfs given a value por the prevalence of the positive class

        Parameters
        ----------
        prevalence : float,
           The prevalence for the positive class

        pos_distrib : array, shape(n_bins,)
            The distribution of the positive class. The exact shape depends on the representation (pdfs, quantiles...)

        neg_distrib : array, shape(n_bins,)
            The distribution of the negative class. The exact shape depends on the representation (pdfs, quantiles...)

        Returns
        -------
        mixture : array, same shape of positives and negatives
           The pdf mixture of positives and negatives
    """
    mixture = pos_distrib * prevalence + neg_distrib * (1 - prevalence)
    return mixture


def compute_quantiles(prevalence=None, probabilities=None, n_quantiles=None, y=None, classes=None):
    """ Compute quantiles

        Used by quantiles-based classes. It computes the quantiles both for the testing distribution (in this case
        the value of the prevalence is ignored), and for the weighted mixture of positives and negatives (this depends
        on the value of the prevalence parameter)

        Parameters
        ----------
        prevalence : float or None
            The value of the prevalence of the positive class to compute the mixture of the positives and the negatives.
            To compute the quantiles of the testing set this parameter must be None

        probabilities : ndarray, shape (nexamples, 1)
            The ordered probabilities for all examples. Notice that in the case of computing the mixture of the
            positives and the negatives, this array contains the probability for all the examples of the training set

        n_quantiles : int
            Number of quantiles. This parameter is used with Quantiles-based algorithms.

        y : array, labels
            This parameter is used with Quantiles-based algorithms. They need the true label of each example

        classes: ndarray, shape (n_classes, )
            Class labels. Used by Quantiles-based algorithms

        Returns
        -------
        quantiles : array, shape(n_quantiles,)
           The value of the quantiles given the probabilities (and the value of the prevalence if we are computing the
           quantiles of the training mixture distribution)
    """

    # by default (test set) the weights are all equal
    p_weight = np.ones(len(probabilities))
    if prevalence is not None:
        # train set
        n = 1 - prevalence
        n_negatives = np.sum(y == classes[0])
        n_positives = np.sum(y == classes[1])
        p_weight[y == classes[0]] = n * len(probabilities) / n_negatives
        p_weight[y == classes[1]] = prevalence * len(probabilities) / n_positives

    cutpoints = np.array(range(1, n_quantiles + 1)) / n_quantiles * len(probabilities)

    quantiles = np.zeros(n_quantiles)
    accsum = 0
    j = 0
    for i in range(len(probabilities)):
        accsum = accsum + p_weight[i]
        if accsum < cutpoints[j]:
            quantiles[j] = quantiles[j] + probabilities[i] * p_weight[i]
        else:
            quantiles[j] = quantiles[j] + probabilities[i] * (p_weight[i] - (accsum - cutpoints[j]))
            withoutassign = accsum - cutpoints[j]
            while withoutassign > 0.1:
                j = j + 1
                assign = min(withoutassign, cutpoints[j] - cutpoints[j - 1])
                quantiles[j] = quantiles[j] + probabilities[i] * assign
                withoutassign = withoutassign - assign

    quantiles = quantiles / cutpoints[0]
    return quantiles


def compute_sord_weights(prevalence=None, union_labels=None, classes=None):
    """ Computes the weight for each example, depending on the prevalence, to compute afterwards the SORD distance

        Parameters
        ----------
        prevalence : float,
           The prevalence for the positive class

        union_labels  :  ndarray, shape (n_examples_train+n_examples_test, 1)
            Contains the set/or  the label of each prediction. If the prediction corresponds to
            a training example, the value is the true class of such example. If the example belongs to the testing
            distribution, the value is NaN

        classes : ndarray, shape (n_classes, )
            Class labels

        Returns
        -------
        weights : array, same shape of union_labels
           The weight of each example, that is equal to:

           negative class = (1-prevalence)*1/|D^-|
           positive class = prevalence*1/|D^+|
           testing examples  = - 1 / |T|
    """
    weights = np.zeros((len(union_labels), 1))
    for n_cls, cls in enumerate(classes):
        if n_cls == 0:
            weights[union_labels == cls] = (1 - prevalence) / np.sum(union_labels == cls)
        else:
            weights[union_labels == cls] = prevalence / np.sum(union_labels == cls)
    weights[union_labels == np.max(union_labels)] = -1.0 / np.sum(union_labels == np.max(union_labels))
    return weights


def sord(weights, union_distrib):
    total_cost = 0
    acc = weights[0]
    for i in range(1, len(weights)):
        delta = union_distrib[i] - union_distrib[i - 1]
        total_cost = total_cost + np.abs(delta * acc)
        acc = acc + weights[i]
    return total_cost


def is_V_shape(distance_func, mixture_func, test_distrib, step, verbose, **kwargs):
    """ Checks if the distance function is V-shaped

        Golden section search only works with V-shape distance (loss) functions

        Parameters
        ----------
        distance_func : function
            This is the loss function minimized during the search

        mixture_func : function
            The function used to generated the training mixture distribution given a value for the prevalence

        test_distrib : array
            The distribution of the positive class. The exact shape depends on the representation (pdfs, quantiles...)

        step: float
            The step to perform the linear search in the interval [0,1]

        verbose: bool
            True to print the distance for each prevalence and the best prevalence acording to the distance function

        kwargs : keyword arguments
            Here we pass the set of arguments needed by mixture functions: mixture_two_pdfs (for pdf-based classes) and
            compute quantiles (for quantiles-based classes). See the help of this two functions

        Returns
        -------
        True if the the distance function is v-shaped and False otherwise
    """
    n_mins = 0
    current_dist = distance_func(mixture_func(prevalence=0, **kwargs), test_distrib)
    next_dist = distance_func(mixture_func(prevalence=step, **kwargs), test_distrib)
    if verbose:
        # print('%.2f %.4f # %.2f %.4f # ' % (0, current_dist, step, next_dist), end='')
        print('%.8f, %.8f,' % (current_dist, next_dist), end='')
    best_p = 2
    min_dist = current_dist
    p = 2 * step
    while p <= 1:
        previous_dist = current_dist
        current_dist = next_dist
        next_dist = distance_func(mixture_func(prevalence=p, **kwargs), test_distrib)
        if verbose:
            # print('%.2f %.4f # ' % (p, next_dist), end='')
            print('%.8f, ' % (next_dist), end='')
        if current_dist < previous_dist and current_dist < next_dist:
            n_mins = n_mins + 1
            if current_dist < min_dist:
                best_p = p
        p = p + step
    if verbose:
        print('\nNumber of minimuns: %d # Best prevalence: %.2f' % (n_mins, best_p))
    return n_mins <= 1, best_p


def probs2crisps(preds, labels):
    """ Convert probability predictions to crisp predictions

        Parameters
        ----------
        preds : ndarray, shape (n_examples, 1) or (n_examples,) for binary problems, (n_examples, n_classes) multiclass
            The matrix with the probability predictions

        labels : ndarray, shape (n_classes, )
            Class labels
    """
    if len(preds) == 0:
        return preds
    if preds.ndim == 1 or preds.shape[1] == 1:
        #  binary problem
        if preds.ndim == 1:
            preds_mod = np.copy(preds)
        else:
            preds_mod = np.copy(preds.squeeze())
        if isinstance(preds_mod[0], np.float):
            # it contains probs
            preds_mod[preds_mod >= 0.5] = 1
            preds_mod[preds_mod < 0.5] = 0
            return preds_mod.astype(int)
        else:
            return preds_mod
    else:
        # multiclass problem
        if isinstance(preds[0, 0], np.float):
            # it contains probs
            #  preds_mod = np.copy(preds)
            return labels.take(preds.argmax(axis=1), axis=0)
        else:
            raise TypeError("probs2crips: error converting probabilities, the type of the values is int")


def create_bags_with_multiple_prevalence(X, y, n=1001, rng=None):
    """ Create bags of examples given a dataset with different prevalences

        The method proposed by Kramer is used to generate a uniform distribution of the prevalences

        Parameters
        ----------
        X : array-like, shape (n_examples, n_features)
            Data

        y : array-like, shape (n_examples, )
            True classes

        n : int, default (n=1001)
            Number of bags

        rng : int, RandomState instance, (default=None)
            To generate random numbers
            If type(rng) is int, rng is the seed used by the random number generator;
            If rng is a RandomState instance, rng is the own random number generator;

        Raises
        ------
        ValueError
            When rng is neither a int nor a RandomState object

        References
        ----------
        http://www.cs.cmu.edu/~nasmith/papers/smith+tromble.tr04.pdf

        http://blog.geomblog.org/2005/10/sampling-from-simplex.html
    """
    if isinstance(rng, (numbers.Integral, np.integer)):
        rng = np.random.RandomState(rng)
    if not isinstance(rng, np.random.RandomState):
        raise ValueError("Invalid random generaror object")

    X, y = check_X_y(X, y)
    classes = np.unique(y)
    n_classes = len(classes)
    m = len(X)

    for i in range(n):
        # Kraemer method:

        # to soft limits
        low = round(m * 0.05)
        high = round(m * 0.95)

        ps = rng.randint(low, high, n_classes - 1)
        ps = np.append(ps, [0, m])
        ps = np.diff(np.sort(ps))  # number of samples for each class
        prev = ps / m  # to obtain prevalences
        idxs = []
        for n, p in zip(classes, ps.tolist()):
            if p != 0:
                idx = rng.choice(np.where(y == n)[0], p, replace=True)
                idxs.append(idx)

        idxs = np.concatenate(idxs)
        yield X[idxs], y[idxs], prev, idxs


def absolute_error(p_true, p_pred):
    """Just the absolute difference between both prevalences.

        Parameters
        ----------
        p_true : array_like, shape=(n_classes)
            True prevalences. In case of binary quantification, this parameter could be a single float value.

        p_pred : array_like, shape=(n_classes)
            Predicted prevalences. In case of binary quantification, this parameter could be a single float value.
        """
    return np.abs(p_pred - p_true)


def relative_absolute_error(p_true, p_pred, eps=1e-12):
    """ A binary relative version of the absolute error

        It is the relation between the absolute error and the true prevalence.

            :math:`rae = | \hat{p} - p | / p`

        Parameters
        ----------
        p_true : float
            True prevalence for the positive class

        p_pred : float
            Predicted prevalence for the positive class

        eps : float, (default=1e-12)
            To prevent a division by zero exception

        Returns
        -------
        relative_absolute_error: float
            It is equal to :math:`| \hat{p} - p | / p`
    """
    if p_true == 0:
        return np.abs(p_pred - p_true) / (p_true + eps)
    else:
        return np.abs(p_pred - p_true) / p_true


def squared_error(p_true, p_pred):
    """ Binary version of the squared errro. Only the prevalence of the positive class is used

        It is the quadratic difference between the predicted prevalence (:math:`\hat{p}`) and
        the true prevalence (:math:`p`)

            :math:`se = (\hat{p} - p)^2`

        It penalizes larger errors

        Parameters
        ----------
        p_true : float
            True prevalence for the positive class

        p_pred : float
            Predicted prevalence for the positive class

        Returns
        -------
        squared_error: float
            It is equal to :math:`(\hat{p} - p)^2`
    """
    return (p_pred - p_true) ** 2


def check_prevalences(p_true, p_pred):
    """ Check that p_true and p_pred are valid and consistent

        Parameters
        ----------
        p_true : array_like, shape = (n_classes)
            True prevalences

        p_pred : array_like, shape = (n_classes)
            Predicted prevalences

        Returns
        -------
        p_true : array-like of shape = (n_classes, 1)
            The converted and validated p_true array

        p_pred : array-like of shape = (n_classes, 1)
            The converted and validated p_pred array
    """
    check_consistent_length(p_true, p_pred)
    p_true = check_array(p_true, ensure_2d=False)
    p_pred = check_array(p_pred, ensure_2d=False)

    if p_true.ndim == 1:
        p_true = p_true.reshape((-1, 1))

    if p_pred.ndim == 1:
        p_pred = p_pred.reshape((-1, 1))

    if p_true.shape[1] != p_pred.shape[1]:
        raise ValueError("p_true and p_pred have different length")

    return p_true, p_pred


def l2(p_true, p_pred):
    """ L2 loss function

            :math:`l2 = \sqrt{\sum_{j=1}^{j=l} (p_j - \hat{p}_j)^2}`

        being l the number of classes

        Parameters
        ----------
        p_true : array_like, shape = (n_classes)
            True prevalences

        p_pred : array_like, shape = (n_classes)
            Predicted prevalences

        Returns
        -------
        l2 : float
            It is equal to :math:`\sqrt{\sum_{j=1}^{j=l} (p_j - \hat{p}_j)^2}`
    """
    p_true, p_pred = check_prevalences(p_true, p_pred)
    return np.sqrt(np.sum((p_true - p_pred) ** 2))


def l1(p_true, p_pred):
    """ L1 loss function

            :math:`l1 = \sum_{j=1}^{j=l} | p_j - \hat{p}_j |`

        being l the number of classes

        Parameters
        ----------
        p_true : array_like, shape = (n_classes)
            True prevalences

        p_pred : array_like, shape = (n_classes)
            Predicted prevalences

        Returns
        -------
        l1 : float
            It is equal to :math:`\sum_{j=1}^{j=l} | p_j - \hat{p}_j |`
    """
    p_true, p_pred = check_prevalences(p_true, p_pred)
    return np.sum(np.abs(p_true - p_pred))


def mean_squared_error(p_true, p_pred):
    """ Mean squared error

            :math:`mse = 1/l \sum_{j=1}^{j=l} (p_j - \hat{p}_j)^2`

        being l the number of classes

        Parameters
        ----------
        p_true : array_like, shape = (n_classes)
            True prevalences

        p_pred : array_like, shape = (n_classes)
            Predicted prevalences

        Returns
        -------
        MSE : float
            It is equal to :math:`1/l \sum_{j=1}^{j=l} (p_j - \hat{p}_j)^2`
    """
    p_true, p_pred = check_prevalences(p_true, p_pred)
    return np.mean((p_pred - p_true)**2)


def hd(p_true, p_pred):
    """ Hellinger distance (HD)

            :math:`hd = \sqrt{\sum_{j=1}^{j=l} (\sqrt{p_j} - \sqrt{\hat{p}_j}}`

        being l the number of classes

        Parameters
        ----------
        p_true : array_like, shape = (n_classes)
            True prevalences

        p_pred : array_like, shape = (n_classes)
            Predicted prevalences

        Returns
        -------
        HD : float
            It is equal to :math:`\sqrt{\sum_{j=1}^{j=l} (\sqrt{p_j} - \sqrt{\hat{p}_j}}`
    """
    p_true, p_pred = check_prevalences(p_true, p_pred)
    dist = np.sqrt(np.sum((np.sqrt(p_pred) - np.sqrt(p_true)) ** 2))
    return dist



