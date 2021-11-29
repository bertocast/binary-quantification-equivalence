import numpy as np
import six
from abc import ABCMeta

from sklearn.base import BaseEstimator, ClassifierMixin, is_classifier
from sklearn.exceptions import NotFittedError
from sklearn.utils import check_X_y, check_array
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_validate
from sklearn.model_selection._split import check_cv

from scipy.sparse import issparse

from utils import probs2crisps
from utils import solve_hd, compute_l2_param_train, solve_l1, solve_l2, solve_l2cvx
from utils import golden_section_search, mixture_two_pdfs, compute_quantiles, l2, compute_sord_weights, sord


class BaseQuantifier(six.with_metaclass(ABCMeta, BaseEstimator)):
    pass


class WithoutClassifiers(BaseQuantifier):
    pass


class UsingClassifiers(BaseQuantifier):

    def __init__(self, estimator_train=None, estimator_test=None, needs_predictions_train=True,
                 probabilistic_predictions=True, verbose=0):
        # init attributes
        self.estimator_train = estimator_train
        self.estimator_test = estimator_test
        self.needs_predictions_train = needs_predictions_train
        self.probabilistic_predictions = probabilistic_predictions
        self.verbose = verbose
        # computed attributes
        self.predictions_test_ = None
        self.predictions_train_ = None
        self.classes_ = None
        self.y_ext_ = None

    def fit(self, X, y, predictions_train=None):
        self.classes_ = np.unique(y)

        if self.needs_predictions_train and self.estimator_train is None and predictions_train is None:
            raise ValueError("estimator_train or predictions_train must be not None "
                             "with objects of class %s", self.__class__.__name__)

        # Fit estimators if they are not already fitted
        if self.estimator_train is not None:
            if self.verbose > 0:
                print('Class %s: Fitting estimator for training distribution...' % self.__class__.__name__, end='')
            # we need to fit the estimator for the training distribution
            # we check if the estimator is trained or not
            try:
                self.estimator_train.predict(X[0:1, :].reshape(1, -1))
                if self.verbose > 0:
                    print('it was already fitted')

            except NotFittedError:

                X, y = check_X_y(X, y, accept_sparse=True)

                self.estimator_train.fit(X, y)

                if self.verbose > 0:
                    print('fitted')

        if self.estimator_test is not None:
            if self.verbose > 0:
                print('Class %s: Fitting estimator for testing distribution...' % self.__class__.__name__, end='')

            # we need to fit the estimator for the testing distribution
            # we check if the estimator is trained or not
            try:
                self.estimator_test.predict(X[0:1, :].reshape(1, -1))
                if self.verbose > 0:
                    print('it was already fitted')

            except NotFittedError:

                X, y = check_X_y(X, y, accept_sparse=True)

                self.estimator_test.fit(X, y)

                if self.verbose > 0:
                    print('fitted')

        # Compute predictions_train_
        if self.verbose > 0:
            print('Class %s: Computing predictions for training distribution...' % self.__class__.__name__, end='')

        if self.needs_predictions_train:
            if predictions_train is not None:
                if self.probabilistic_predictions:
                    self.predictions_train_ = predictions_train
                else:
                    self.predictions_train_ = probs2crisps(predictions_train, self.classes_)
            else:
                if self.probabilistic_predictions:
                    self.predictions_train_ = self.estimator_train.predict_proba(X)
                else:
                    self.predictions_train_ = self.estimator_train.predict(X)

            # Compute y_ext_
            if len(y) == len(self.predictions_train_):
                self.y_ext_ = y
            else:
                self.y_ext_ = np.tile(y, len(self.predictions_train_) // len(y))

        if self.verbose > 0:
            print('done')

        return self

    def predict(self, X, predictions_test=None):
        if self.estimator_test is None and predictions_test is None:
            #  or self.estimator_test is not None and predictions_test is not None:
            raise ValueError("estimator_test or predictions_test must be not None "
                             "with objects of class %s", self.__class__.__name__)

        if self.verbose > 0:
            print('Class %s: Computing predictions for testing distribution...' % self.__class__.__name__, end='')

        # At least one between estimator_test and predictions_test is not None
        if predictions_test is not None:
            if self.probabilistic_predictions:
                self.predictions_test_ = predictions_test
            else:
                self.predictions_test_ = probs2crisps(predictions_test, self.classes_)
        else:
            check_array(X, accept_sparse=True)
            if self.probabilistic_predictions:
                self.predictions_test_ = self.estimator_test.predict_proba(X)
            else:
                self.predictions_test_ = self.estimator_test.predict(X)

        if self.verbose > 0:
            print('done')

        return self


class CC(UsingClassifiers):
    def __init__(self, estimator_test=None, verbose=0):
        super(CC, self).__init__(estimator_test=estimator_test,
                                 needs_predictions_train=False, probabilistic_predictions=False, verbose=verbose)

    def fit(self, X, y, predictions_train=None):
        super().fit(X, y, predictions_train=[])

        return self

    def predict(self, X, predictions_test=None):
        super().predict(X, predictions_test=predictions_test)

        if self.verbose > 0:
            print('Class %s: Computing prevalences for testing distribution...' % self.__class__.__name__, end='')

        n_classes = len(self.classes_)

        freq = np.zeros((n_classes, 1))
        for n_cls, cls in enumerate(self.classes_):
            freq[n_cls, 0] = np.equal(self.predictions_test_, cls).sum()

        prevalences = freq / float(len(self.predictions_test_))

        if self.verbose > 0:
            print('done')

        return np.squeeze(prevalences)


class AC(UsingClassifiers):

    def __init__(self, estimator_train=None, estimator_test=None, verbose=0):
        super(AC, self).__init__(estimator_train=estimator_train, estimator_test=estimator_test,
                                 needs_predictions_train=True, probabilistic_predictions=False, verbose=verbose)
        # confusion matrix
        self.cm_ = None
        
    def fit(self, X, y, predictions_train=None):
        super().fit(X, y, predictions_train=predictions_train)

        if self.verbose > 0:
            print('Class %s: Estimating confusion matrix for training distribution...' % self.__class__.__name__,
                  end='')

        #  estimating the confusion matrix
        cm = confusion_matrix(self.y_ext_, self.predictions_train_, labels=self.classes_)
        #  normalizing cm by row
        self.cm_ = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # binary:  [[1-fpr  fpr]
        #                                                                          [1-tpr  tpr]]

        if self.verbose > 0:
            print('done')

        return self

    def predict(self, X, predictions_test=None):
        super().predict(X, predictions_test=predictions_test)

        if self.verbose > 0:
            print('Class %s: Computing prevalences for testing distribution...' % self.__class__.__name__, end='')

        n_classes = len(self.classes_)

        freq = np.zeros((n_classes, 1))
        for n_cls, cls in enumerate(self.classes_):
            freq[n_cls, 0] = np.equal(self.predictions_test_, cls).sum()

        prevalences_0 = freq / float(len(self.predictions_test_))

        
        if np.abs((self.cm_[1, 1] - self.cm_[0, 1])) > 0.001:
            p = (prevalences_0[1] - self.cm_[0, 1]) / (self.cm_[1, 1] - self.cm_[0, 1])
            prevalences = [1-p, p]
        else:
            prevalences = prevalences_0

        # clipping the quant_results according to (Forman 2008)
        prevalences = np.clip(prevalences, 0, 1)

        if np.sum(prevalences) > 0:
            prevalences = prevalences / float(np.sum(prevalences))

        prevalences = prevalences.squeeze()

        if self.verbose > 0:
            print('done')

        # print('AC: p %.5f l2 %.5f' %(prevalences[1], l2(self.cm_.T.dot(prevalences), prevalences_0.squeeze())))
        return prevalences


class PAC(UsingClassifiers):
    """ Multiclass Probabilistic Adjusted Count method

        This class works in two different ways:

        1) Two estimators are used to classify the examples of the training set and the testing set in order to
           compute the (probabilistic) confusion matrix of both sets. Estimators can be already trained

        2) You can directly provide the predictions for the examples in the `fit`/`predict methods. This is useful
           for synthetic/artificial experiments

        The idea in both cases is to guarantee that all methods based on distribution matching are using **exactly**
        the same predictions when you compare this kind of quantifiers (and others that also employ an underlying
        classifier, for instance, CC/PCC). In the first case, estimators are only trained once and can be shared
        for several quantifiers of this kind

        Parameters
        ----------
        estimator_train : estimator object (default=None)
            An estimator object implementing `fit` and `predict_proba`. It is used to classify the examples of the
            training set and to compute the confusion matrix

        estimator_test : estimator object (default=None)
            An estimator object implementing `fit` and `predict_proba`. It is used to classify the examples of the
            testing set and to obtain the confusion matrix of the testing set

        distance : str, representing the distance function (default='L2')
            It is the name of the distance used to compute the difference between the mixture of the training
            distribution and the testing distribution. Only used in multiclass problems.
            Distances supported: 'HD', 'L2' and 'L1'

        verbose : int, optional, (default=0)
            The verbosity level. The default value, zero, means silent mode

        For some experiments both estimators could be the same

        Attributes
        ----------
        estimator_train : estimator
            Estimator used to classify the examples of the training set

        estimator_test : estimator
            Estimator used to classify the examples of the testing bag

        distance : str
            A string with the name of the distance function ('HD'/'L1'/'L2')

        predictions_train_ : ndarray, shape (n_examples, n_classes) (probabilistic estimator)
            Predictions of the examples in the training set

        predictions_test_ : ndarray, shape (n_examples, n_classes) (probabilistic estimator)
            Predictions of the examples in the testing bag

        needs_predictions_train : bool, True
            It is True because PAC quantifiers need to estimate the training distribution

        probabilistic_predictions : bool, True
             This means that predictions_test_ contains probabilistic predictions

        classes_ : ndarray, shape (n_classes, )
            Class labels

        y_ext_ : ndarray, shape(len(predictions_train_, 1)
            Repmat of true labels of the training set. When CV_estimator is used with averaged_predictions=False,
            predictions_train_ will have a larger dimension (factor=n_repetitions * n_folds of the underlying CV)
            than y. In other cases, y_ext_ == y.
            y_ext_ i used in `fit` method whenever the true labels of the training set are needed, instead of y


        cm_ : ndarray, shape (n_classes, n_classes)
            Confusion matrix

        G_, C_, b_: variables of different kind for defining the optimization problem
            These variables are precomputed in the `fit` method and are used for solving the optimization problem
            using `quadprog.solve_qp`. See `compute_l2_param_train` function

        verbose : int
            The verbosity level

        Notes
        -----
        Notice that at least one between estimator_train/predictions_train and estimator_test/predictions_test
        must be not None. If both are None a ValueError exception will be raised. If both are not None,
        predictions_train/predictions_test are used

        References
        ----------
        Antonio Bella, Cèsar Ferri, José Hernández-Orallo, and María José Ramírez-Quintana. 2010. Quantification
        via probability estimators. In Proceedings of the IEEE International Conference on Data Mining (ICDM’10).
        IEEE, 737–742.
    """

    def __init__(self, estimator_test=None, estimator_train=None, distance='L2', verbose=0):
        super(PAC, self).__init__(estimator_test=estimator_test, estimator_train=estimator_train,
                                  needs_predictions_train=True, probabilistic_predictions=True, verbose=verbose)
        self.distance = distance
        # confusion matrix with average probabilities
        self.cm_ = None
        # variables for solving the optimization problem when n_classes > 2 and distance = 'L2'
        self.G_ = None
        self.C_ = None
        self.b_ = None

    def fit(self, X, y, predictions_train=None):
        """ This method performs the following operations: 1) fits the estimators for the training set and the
            testing set (if needed), and 2) computes predictions_train_ (probabilities) if needed. Both operations are
            performed by the `fit method of its superclass.
            Finally the method computes the (probabilistic) confusion matrix using predictions_train

            Parameters
            ----------
            X : array-like, shape (n_examples, n_features)
                Data

            y : array-like, shape (n_examples, )
                True classes

            predictions_train : ndarray, shape (n_examples, n_classes)
                Predictions of the training set

            Raises
            ------
            ValueError
                When estimator_train and predictions_train are both None
        """
        super().fit(X, y, predictions_train=predictions_train)

        if self.verbose > 0:
            print('Class %s: Estimating average probabilities for training distribution...'
                  % self.__class__.__name__, end='')

        n_classes = len(self.classes_)
        # estimating the confusion matrix
        # average probabilty distribution for each class
        self.cm_ = np.zeros((n_classes, n_classes))
        for n_cls, cls in enumerate(self.classes_):
            self.cm_[n_cls] = np.mean(self.predictions_train_[self.y_ext_ == cls], axis=0)

        if len(self.classes_) > 2 and self.distance == 'L2':
            self.G_, self.C_, self.b_ = compute_l2_param_train(self.cm_.T, self.classes_)

        if self.verbose > 0:
            print('done')

        return self

    def predict(self, X, predictions_test=None):
        """ Predict the class distribution of a testing bag

            First, predictions_test_ are computed (if needed, when predictions_test parameter is None) by
            `super().predict() method.

            After that, the prevalences are computed solving a system of linear scalar equations:

                         cm_.T * prevalences = PCC(X)

            For binary problems the system is directly solved using the original PAC algorithm proposed by Bella et al.

                        p = (p_0 - PA(negatives) ) / ( PA(positives) - PA(negatives) )

            in which PA stands for probability average.

            For multiclass problems, the system may not have a solution. Thus, instead we propose to solve an
            optimization problem of this kind:

                      Min   distance ( cm_.T * prevalences, PCC(X) )
                      s.t.  sum(prevalences) = 1
                            prevalecences_i >= 0

            in which distance can be 'HD', 'L1' or 'L2' (defect value)

            Parameters
            ----------
            X : array-like, shape (n_examples, n_features)
                Testing bag

            predictions_test : ndarray, shape (n_examples, n_classes) (default=None)
                They must be probabilities (the estimator used must have a `predict_proba` method)

                If predictions_test is not None they are copied on predictions_test_ and used.
                If predictions_test is None, predictions for the testing examples are computed using the `predict`
                method of estimator_test (it must be an actual estimator)

            Raises
            ------
            ValueError
                When estimator_test and predictions_test are both None

            Returns
            -------
            prevalences : ndarray, shape(n_classes, )
                Contains the predicted prevalence for each class
        """
        super().predict(X, predictions_test=predictions_test)

        if self.verbose > 0:
            print('Class %s: Computing prevalences for testing distribution...' % self.__class__.__name__, end='')

        n_classes = len(self.classes_)
        prevalences_0 = np.mean(self.predictions_test_, axis=0)

        if n_classes == 2:
            if np.abs(self.cm_[1, 1] - self.cm_[0, 1]) > 0.001:
                p = (prevalences_0[1] - self.cm_[0, 1]) / (self.cm_[1, 1] - self.cm_[0, 1])
                prevalences = [1 - p, p]
            else:
                prevalences = prevalences_0
            # prevalences = np.linalg.solve(self.cm_.T, prevalences_0)

            # clipping the quant_results according to (Forman 2008)
            prevalences = np.clip(prevalences, 0, 1)

            if np.sum(prevalences) > 0:
                prevalences = prevalences / float(np.sum(prevalences))

            prevalences = prevalences.squeeze()
        else:
            if self.distance == 'HD':
                prevalences = solve_hd(train_distrib=self.cm_.T, test_distrib=prevalences_0, n_classes=n_classes)
            elif self.distance == 'L2':
                prevalences = solve_l2(train_distrib=self.cm_.T, test_distrib=prevalences_0,
                                       G=self.G_, C=self.C_, b=self.b_)
            elif self.distance == 'L2cvx':
                prevalences = solve_l2cvx(train_distrib=self.cm_.T, test_distrib=prevalences_0, n_classes=n_classes)
            elif self.distance == 'L1':
                prevalences = solve_l1(train_distrib=self.cm_.T, test_distrib=prevalences_0, n_classes=n_classes)
            else:
                raise ValueError('Class %s": distance function not supported', self.__class__.__name__)

        if self.verbose > 0:
            print('done')

        # print('PAC: p %.5f l2 %.5f' %(prevalences[1], l2(self.cm_.T.dot(prevalences), prevalences_0.squeeze())))
        return prevalences


class DFy(UsingClassifiers):
    """ Generic Multiclass DFy method

        The idea is to represent the mixture of the training distribution and the testing distribution
        (using CDFs/PDFs) of the predictions given by a classifier (y). The difference between both is minimized
        using a distance/loss function. Originally, (González et al. 2013) propose the combination of PDF and
        Hellinger Distance, but also CDF and any other distance/loss function could be used, like L1 or L2. In fact,
        Forman (2005) propose to use CDF's an a measure equivalent to L1.

        The class has two parameters to select:

        - the method used to represent the distributions (CDFs or PDFs)
        - the distance used.

        This class (as every other class based on distribution matching using classifiers) works in two different ways:

        1) Two estimators are used to classify training examples and testing examples in order to
           compute the distribution of both sets. Estimators can be already trained

        2) You can directly provide the predictions for the examples in the fit/predict methods. This is useful
           for synthetic/artificial experiments

        The goal in both cases is to guarantee that all methods based on distribution matching are using **exactly**
        the same predictions when you compare this kind of quantifiers (and others that also employ an underlying
        classifier, for instance, CC/PCC and AC/PAC). In the first case, estimators are only trained once and can
        be shared for several quantifiers of this kind

        Parameters
        ----------
        estimator_train : estimator object (default=None)
            An estimator object implementing `fit` and `predict_proba`. It is used to classify the examples of the
            training set and to compute the distribution of each class individually

        estimator_test : estimator object (default=None)
            An estimator object implementing `fit` and `predict_proba`. It is used to classify the examples of the
            testing set and to compute the distribution of the whole testing set

        distribution_function : str, (default='PDF')
            Type of distribution function used. Two types are supported 'CDF' and 'PDF'

        n_bins : int  (default=8)
            Number of bins to compute the CDFs/PDFs

        distance : str, representing the distance function (default='HD')
            It is the name of the distance used to compute the difference between the mixture of the training
            distribution and the testing distribution

        tol : float, (default=1e-05)
            The precision of the solution when search is used to compute the prevalence

        verbose : int, optional, (default=0)
            The verbosity level. The default value, zero, means silent mode

        For some experiments both estimator_train and estimator_test could be the same

        Attributes
        ----------
        estimator_train : estimator
            Estimator used to classify the examples of the training set

        estimator_test : estimator
            Estimator used to classify the examples of the testing bag

        predictions_train_ : ndarray, shape (n_examples, n_classes) (probabilities)
            Predictions of the examples in the training set

        predictions_test_ : ndarray, shape (n_examples, n_classes) (probabilities)
            Predictions of the examples in the testing bag

        needs_predictions_train : bool, True
            It is True because PDFy quantifiers need to estimate the training distribution

        probabilistic_predictions : bool, True
             This means that predictions_train_/predictions_test_ contain probabilistic predictions

        distance : str or a distance function
            A string with the name of the distance function ('HD'/'L1'/'L2') or a distance function

        tol : float
            The precision of the solution when search is used to compute the solution

        classes_ : ndarray, shape (n_classes, )
            Class labels

        y_ext_ : ndarray, shape(len(predictions_train_, 1)
            Repmat of true labels of the training set. When CV_estimator is used with averaged_predictions=False,
            predictions_train_ will have a larger dimension (factor=n_repetitions * n_folds of the underlying CV)
            than y. In other cases, y_ext_ == y.
            y_ext_ i used in `fit` method whenever the true labels of the training set are needed, instead of y

        distribution_function : str
            Type of distribution function used. Two types are supported 'CDF' and 'PDF'

        n_bins : int
            The number of bins to compute the CDFs/PDFs

        train_distrib_ : ndarray, shape (n_bins * 1, n_classes) binary or (n_bins * n_classes_, n_classes) multiclass
            The CDF/PDF for each class in the training set

        test_distrib_ : ndarray, shape (n_bins * 1, 1) binary quantification or (n_bins * n_classes_, 1) multiclass q
            The CDF/PDF for the testing bag

        G_, C_, b_: variables of different kind for defining the optimization problem
            These variables are precomputed in the `fit` method and are used for solving the optimization problem
            using `quadprog.solve_qp`. See `compute_l2_param_train` function

        verbose : int
            The verbosity level

        Notes
        -----
        Notice that at least one between estimator_train/predictions_train and estimator_test/predictions_test
        must be not None. If both are None a ValueError exception will be raised. If both are not None,
        predictions_train/predictions_test are used

        References
        ----------
        Víctor González-Castro, Rocío Alaiz-Rodríguez, and Enrique Alegre: Class Distribution Estimation based
        on the Hellinger Distance. Information Sciences 218 (2013), 146–164.

        George Forman: Counting positives accurately despite inaccurate classification. In: Proceedings of the 16th
        European conference on machine learning (ECML'05), Porto, (2005) pp 564–575

        Aykut Firat. 2016. Unified Framework for Quantification. arXiv preprint arXiv:1606.00868 (2016).
    """

    def __init__(self, estimator_train=None, estimator_test=None, distribution_function='PDF', n_bins=8, distance='HD',
                 tol=1e-05, verbose=0):
        super(DFy, self).__init__(estimator_train=estimator_train, estimator_test=estimator_test,
                                  needs_predictions_train=True, probabilistic_predictions=True, verbose=verbose)
        # attributes
        self.distribution_function = distribution_function
        self.n_bins = n_bins
        self.distance = distance
        self.tol = tol
        # variables to represent the distributions
        self.train_distrib_ = None
        self.test_distrib_ = None
        # variables for solving the optimization problem
        self.G_ = None
        self.C_ = None
        self.b_ = None

    def fit(self, X, y, predictions_train=None):
        """ This method performs the following operations: 1) fits the estimators for the training set and the
            testing set (if needed), and 2) computes predictions_train_ (probabilities) if needed. Both operations are
            performed by the `fit` method of its superclass.
            After that, the method computes the pdfs for all the classes in the training set

            Parameters
            ----------
            X : array-like, shape (n_examples, n_features)
                Data

            y : array-like, shape (n_examples, )
                True classes

            predictions_train : ndarray, shape (n_examples, n_classes)
                Predictions of the examples in the training set

            Raises
            ------
            ValueError
                When estimator_train and predictions_train are both None
        """
        super().fit(X, y, predictions_train=predictions_train)

        if self.verbose > 0:
            print('Class %s: Estimating training distribution...' % self.__class__.__name__, end='')

        n_classes = len(self.classes_)
        if n_classes == 2:
            n_descriptors = 1  # number of groups of probabilities used to represent the distribution
        else:
            n_descriptors = n_classes

        self.train_distrib_ = np.zeros((self.n_bins * n_descriptors, n_classes))
        # compute pdf
        for n_cls, cls in enumerate(self.classes_):
            for descr in range(n_descriptors):
                self.train_distrib_[descr * self.n_bins:(descr + 1) * self.n_bins, n_cls] = \
                    np.histogram(self.predictions_train_[self.y_ext_ == cls, descr], bins=self.n_bins, range=(0., 1.))[
                        0]
            self.train_distrib_[:, n_cls] = self.train_distrib_[:, n_cls] / (np.sum(self.y_ext_ == cls))

        # compute cdf if necessary
        if self.distribution_function == 'CDF':
            self.train_distrib_ = np.cumsum(self.train_distrib_, axis=0)

        if self.distance == 'L2':
            self.G_, self.C_, self.b_ = compute_l2_param_train(self.train_distrib_, self.classes_)

        if self.verbose > 0:
            print('done')

        return self

    def predict(self, X, predictions_test=None):
        """ Predict the class distribution of a testing bag

            First, predictions_test_ are computed (if needed, when predictions_test parameter is None) by
            `super().predict()` method.

            After that, the method computes the PDF for the testing bag.

            Finally, the prevalences are computed using the corresponding function according to the value of
            distance attribute

            Parameters
            ----------
            X : array-like, shape (n_examples, n_features)
                Testing bag

            predictions_test : ndarray, shape (n_examples, n_classes) (default=None)
                They must be probabilities (the estimator used must have a `predict_proba` method)

                If predictions_test is not None they are copied on predictions_test_ and used.
                If predictions_test is None, predictions for the testing examples are computed using the `predict`
                method of estimator_test (it must be an actual estimator)

            Raises
            ------
            ValueError
                When estimator_test and predictions_test are both None

            Returns
            -------
            prevalences : ndarray, shape(n_classes, )
                Contains the predicted prevalence for each class
        """
        super().predict(X, predictions_test=predictions_test)

        if self.verbose > 0:
            print('Class %s: Estimating testing distribution...' % self.__class__.__name__, end='')

        n_classes = len(self.classes_)
        if n_classes == 2:
            n_descriptors = 1
        else:
            n_descriptors = n_classes

        self.test_distrib_ = np.zeros((self.n_bins * n_descriptors, 1))
        # compute pdf
        for descr in range(n_descriptors):
            self.test_distrib_[descr * self.n_bins:(descr + 1) * self.n_bins, 0] = \
                np.histogram(self.predictions_test_[:, descr], bins=self.n_bins, range=(0., 1.))[0]

        self.test_distrib_ = self.test_distrib_ / len(self.predictions_test_)

        #  compute cdf if necessary
        if self.distribution_function == 'CDF':
            self.test_distrib_ = np.cumsum(self.test_distrib_, axis=0)

        if self.verbose > 0:
            print('Class %s: Computing prevalences...' % self.__class__.__name__, end='')

        if self.distance == 'HD':
            prevalences = solve_hd(train_distrib=self.train_distrib_, test_distrib=self.test_distrib_,
                                   n_classes=n_classes)
        elif self.distance == 'L2':
            prevalences = solve_l2(train_distrib=self.train_distrib_, test_distrib=self.test_distrib_,
                                   G=self.G_, C=self.C_, b=self.b_)
        elif self.distance == 'L2cvx':
            prevalences = solve_l2cvx(train_distrib=self.train_distrib_, test_distrib=self.test_distrib_,
                                      n_classes=n_classes)
        elif self.distance == 'L1':
            prevalences = solve_l1(train_distrib=self.train_distrib_, test_distrib=self.test_distrib_,
                                   n_classes=n_classes)
        else:
            prevalences = golden_section_search(distance_func=self.distance, mixture_func=mixture_two_pdfs,
                                                test_distrib=self.test_distrib_, tol=self.tol,
                                                pos_distrib=self.train_distrib_[:, 1].reshape(-1, 1),
                                                neg_distrib=self.train_distrib_[:, 0].reshape(-1, 1))
        if self.verbose > 0:
            print('done')

        return prevalences


class HDy(DFy):
    """ Multiclass HDy method

        This class is just a wrapper. It just uses all the inherited methods of its superclass (DFy)

        References
        ----------
        Víctor González-Castro, Rocío Alaiz-Rodríguez, and Enrique Alegre: Class Distribution Estimation based
        on the Hellinger Distance. Information Sciences 218 (2013), 146–164.
    """

    def __init__(self, estimator_train=None, estimator_test=None, n_bins=8, tol=1e-05, verbose=0):
        super(HDy, self).__init__(estimator_train=estimator_train, estimator_test=estimator_test,
                                  distribution_function='PDF', n_bins=n_bins, distance='HD', tol=tol, verbose=verbose)


class QUANTy(UsingClassifiers):
    """ Generic binary methods for quantiles-y method

        The idea is to represent the mixture of the training distribution and the testing distribution using
        quantiles of the predictions given by a classifier (y). The difference between both is minimized using a
        distance/loss function. This method encapsulates PAC quantifier (Bella et al. 2013). PAC has just 1 quantile
        and with this class you can define more quantiles and use any distance/loss to measure distribution similarity.
        The class has a parameter to select the distance used.

        This class (as every other class based on distribution matching using classifiers) works in two different ways:

        1) Two estimators are used to classify training examples and testing examples in order to
           compute the distribution of both sets. Estimators can be already trained

        2) You can directly provide the predictions for the examples in the fit/predict methods. This is useful
           for synthetic/artificial experiments

        The goal in both cases is to guarantee that all methods based on distribution matching are using **exactly**
        the same predictions when you compare this kind of quantifiers (and others that also employ an underlying
        classifier, for instance, CC/PCC and AC/PAC). In the first case, estimators are only trained once and can
        be shared for several quantifiers of this kind

        Multiclass quantification is not implemented yet for this object. It would need a more complex searching
        algorithm (instead golden_section_search)

        Parameters
        ----------
        estimator_train : estimator object (default=None)
            An estimator object implementing `fit` and `predict_proba`. It is used to classify the examples of the
            training set and to compute the distribution of each class individually

        estimator_test : estimator object (default=None)
            An estimator object implementing `fit` and `predict_proba`. It is used to classify the examples of the
            testing set and to compute the distribution of the whole testing set

        n_quantiles : int
            Number of quantiles

        distance : distance function (default=l2)
            It is the name of the distance used to compute the difference between the mixture of the training
            distribution and the testing distribution

        tol : float, (default=1e-05)
            The precision of the solution when search is used to compute the prevalence

        verbose : int, optional, (default=0)
            The verbosity level. The default value, zero, means silent mode

        For some experiments both estimator_train and estimator_test could be the same

        Attributes
        ----------
        estimator_train : estimator
            Estimator used to classify the examples of the training set

        estimator_test : estimator
            Estimator used to classify the examples of the testing bag

        predictions_train_ : ndarray, shape (n_examples, n_classes) (probabilities)
            Predictions of the examples in the training set

        predictions_test_ : ndarray, shape (n_examples, n_classes) (probabilities)
            Predictions of the examples in the testing bag

        needs_predictions_train : bool, True
            It is True because QUANTy quantifiers need to estimate the training distribution

        probabilistic_predictions : bool, True
            This means that predictions_train_/predictions_test_ contain probabilistic predictions

        classes_ : ndarray, shape (n_classes, )
            Class labels

        y_ext_ : ndarray, shape(len(predictions_train_, 1)
            Repmat of true labels of the training set. When CV_estimator is used with averaged_predictions=False,
            predictions_train_ will have a larger dimension (factor=n_repetitions * n_folds of the underlying CV)
            than y. In other cases, y_ext_ == y.
            y_ext_ i used in `fit`/`predict` method whenever the true labels of the training set are needed,
            instead of y

        n_quantiles : int (default=8)
            The number of quantiles to represent data distribution

        distance : A distance function (default=l2)
            The name of the distance function used

        tol : float
            The precision of the solution when search is used to compute the solution

        train_distrib_ : ndarray, shape (n_examples, 1) binary quantification
            Contains predictions_train_ in ascending order

        train_labels_ : ndarray, shape (n_examples, 1) binary quantification
            Contains the corresponding labels of the examples in train_distrib_ in the same order

        test_distrib_ : ndarray, shape (n_quantiles, 1)
            Contains the quantiles of the test distribution

        verbose : int
            The verbosity level

        Notes
        -----
        Notice that at least one between estimator_train/predictions_train and estimator_test/predictions_test
        must be not None. If both are None a ValueError exception will be raised. If both are not None,
        predictions_train/predictions_test are used
    """

    def __init__(self, estimator_train=None, estimator_test=None, n_quantiles=8, distance=l2, tol=1e-05, verbose=0):
        super(QUANTy, self).__init__(estimator_train=estimator_train, estimator_test=estimator_test,
                                     needs_predictions_train=True, probabilistic_predictions=True, verbose=verbose)
        # attributes
        self.n_quantiles = n_quantiles
        self.distance = distance
        self.tol = tol
        # variables to represent the distributions
        self.train_distrib_ = None
        self.train_labels_ = None
        self.test_distrib_ = None

    def fit(self, X, y, predictions_train=None):
        """ This method performs the following operations: 1) fits the estimators for the training set and the
            testing set (if needed), and 2) computes predictions_train_ (probabilities) if needed. Both operations are
            performed by the `fit` method of its superclass.
            After that, the method orders the predictions for the train set. The actual quantiles are computed by
            a mixture function because it depends on the class prevalence

            Parameters
            ----------
            X : array-like, shape (n_examples, n_features)
                Data

            y : array-like, shape (n_examples, )
                True classes

            predictions_train : ndarray, shape (n_examples, n_classes)
                Predictions of the examples in the training set

            Raises
            ------
            ValueError
                When estimator_train and predictions_train are at the same time None or not None
        """
        super().fit(X, y, predictions_train=predictions_train)

        if len(self.classes_) > 2:
            raise TypeError("QUANTy is a binary method, multiclass quantification is not supported")

        if self.verbose > 0:
            print('Class %s: Collecting data from training distribution...' % self.__class__.__name__, end='')

        #   sorting the probabilities of belonging to the positive class, P(y=+1 | x)
        indx = np.argsort(self.predictions_train_[:, 1])
        self.train_distrib_ = self.predictions_train_[indx, 1]
        self.train_labels_ = self.y_ext_[indx]

        if self.verbose > 0:
            print('done')

        return self

    def predict(self, X, predictions_test=None):
        """ Predict the class distribution of a testing bag

            First, predictions_test_ are computed (if needed, when predictions_test parameter is None) by
            `super().predict()` method.

            After that, the method computes the quantiles for the testing bag.

            Finally, the prevalences are computed using golden section search and the distance function of the object

            Parameters
            ----------
            X : array-like, shape (n_examples, n_features)
                Testing bag

            predictions_test : ndarray, shape (n_examples, n_classes) (default=None)
                They must be probabilities (the estimator used must have a `predict_proba` method)

                If estimator_test == None then predictions_test can not be None.
                If predictions_test is None, predictions for the testing examples are computed using the `predict_proba`
                method of estimator_test (it must be an actual estimator)

            Raises
            ------
            ValueError
                When estimator_test and predictions_test are at the same time None or not None

            Returns
            -------
            prevalences : ndarray, shape(n_classes, )
                Contains the predicted prevalence for each class
        """
        super().predict(X, predictions_test=predictions_test)

        if self.verbose > 0:
            print('Class %s: Computing quantiles of testing distribution...' % self.__class__.__name__, end='')

        #  sorting the probabilities of belonging to the positive class, P(y=+1 | x)
        sorted_test_probabilities = np.sort(self.predictions_test_[:, 1])
        self.test_distrib_ = compute_quantiles(prevalence=None, probabilities=sorted_test_probabilities,
                                               n_quantiles=self.n_quantiles)

        if self.verbose > 0:
            print('done')
            print('Class %s: Computing prevalences for testing distribution...' % self.__class__.__name__, end='')

        prevalences = golden_section_search(distance_func=self.distance, mixture_func=compute_quantiles,
                                            test_distrib=self.test_distrib_, tol=self.tol,
                                            probabilities=self.train_distrib_, n_quantiles=self.n_quantiles,
                                            y=self.train_labels_, classes=self.classes_)
        if self.verbose > 0:
            print('done')

        return prevalences


class SORDy(UsingClassifiers):
    """ SORDy method

        The idea is to represent the mixture of the training distribution and the testing distribution using
        the whole set of predictions given by a classifier (y).

        This class (as every other class based on distribution matching using classifiers) works in two different ways:

        1) Two estimators are used to classify training examples and testing examples in order to
           compute the distribution of both sets. Estimators can be already trained

        2) You can directly provide the predictions for the examples in the fit/predict methods. This is useful
           for synthetic/artificial experiments

        The goal in both cases is to guarantee that all methods based on distribution matching are using **exactly**
        the same predictions when you compare this kind of quantifiers (and others that also employ an underlying
        classifier, for instance, CC/PCC and AC/PAC). In the first case, estimators are only trained once and can
        be shared for several quantifiers of this kind

        Multiclass quantification is not implemented yet for this object. It would need a more complex searching
        algorithm (instead golden_section_search)

        Parameters
        ----------
        estimator_train : estimator object (default=None)
            An estimator object implementing `fit` and `predict_proba`. It is used to classify the examples of the
            training set and to compute the distribution of each class individually

        estimator_test : estimator object (default=None)
            An estimator object implementing `fit` and `predict_proba`. It is used to classify the examples of the
            testing set and to compute the distribution of the whole testing set

        tol : float, (default=1e-05)
            The precision of the solution when search is used to compute the prevalence

        verbose : int, optional, (default=0)
            The verbosity level. The default value, zero, means silent mode

        For some experiments both estimator_train and estimator_test could be the same

        Attributes
        ----------
        estimator_train : estimator
            Estimator used to classify the examples of the training set

        estimator_test : estimator
            Estimator used to classify the examples of the testing bag

        predictions_train_ : ndarray, shape (n_examples, n_classes) (probabilities)
            Predictions of the examples in the training set

        predictions_test_ : ndarray, shape (n_examples, n_classes) (probabilities)
            Predictions of the examples in the testing bag

        needs_predictions_train : bool, True
            It is True because SORD quantifiers need to estimate the training distribution

        probabilistic_predictions : bool, True
            This means that predictions_train_/predictions_test_ contain probabilistic predictions

        classes_ : ndarray, shape (n_classes, )
            Class labels

        y_ext_ : ndarray, shape(len(predictions_train_, 1)
            Repmat of true labels of the training set. When CV_estimator is used with averaged_predictions=False,
            predictions_train_ will have a larger dimension (factor=n_repetitions * n_folds of the underlying CV)
            than y. In other cases, y_ext_ == y.
            y_ext_ i used in `fit`/`predict` method whenever the true labels of the training set are needed,
            instead of y

        tol : float
            The precision of the solution when search is used to compute the solution

        train_distrib_ : ndarray, shape (n_examples_train, 1) binary quantification
            Contains predictions_train_ in ascending order

        train_labels_ : ndarray, shape (n_examples_train, 1) binary quantification
            Contains the corresponding labels of the examples in train_distrib_ in the same order

        union_distrib_ : ndarray, shape (n_examples_train+n_examples_test, 1)
            Contains the union of predictions_train_ and predictions_test_ in ascending order

        union_labels  :  ndarray, shape (n_examples_train+n_examples_test, 1)
            Contains the set/or  the label of each prediction  in union_distrib_. If the prediction corresponds to
            a training example, the value is the true class of such example. If the example belongs to the testing
            distribution, the value is NaN

        verbose : int
            The verbosity level

        Notes
        -----
        Notice that at least one between estimator_train/predictions_train and estimator_test/predictions_test
        must be not None. If both are None a ValueError exception will be raised. If both are not None,
        predictions_train/predictions_test are used

        References
        ----------
        Maletzke, A., dos Reis, D., Cherman, E., and Batista, G. Dys: A framework for mixture models in quantification.
        In AAAI 2019, volume 33, pp. 4552–4560. 2019.
    """

    def __init__(self, estimator_train=None, estimator_test=None, tol=1e-05, verbose=0):
        super(SORDy, self).__init__(estimator_train=estimator_train, estimator_test=estimator_test,
                                     needs_predictions_train=True, probabilistic_predictions=True, verbose=verbose)
        # attributes
        self.tol = tol
        # variables to represent the distributions
        self.train_distrib_ = None
        self.train_labels_ = None
        self.union_distrib_ = None
        self.union_labels_ = None


    def fit(self, X, y, predictions_train=None):
        """ This method performs the following operations: 1) fits the estimators for the training set and the
            testing set (if needed), and 2) computes predictions_train_ (probabilities) if needed. Both operations are
            performed by the `fit` method of its superclass.
            After that, the method orders the predictions for the train set. The actual quantiles are computed by
            a mixture function because it depends on the class prevalence

            Parameters
            ----------
            X : array-like, shape (n_examples, n_features)
                Data

            y : array-like, shape (n_examples, )
                True classes

            predictions_train : ndarray, shape (n_examples, n_classes)
                Predictions of the examples in the training set

            Raises
            ------
            ValueError
                When estimator_train and predictions_train are at the same time None or not None
        """
        super().fit(X, y, predictions_train=predictions_train)

        if len(self.classes_) > 2:
            raise TypeError("SORDy is a binary method, multiclass quantification is not supported")

        if self.verbose > 0:
            print('Class %s: Collecting data from training distribution...' % self.__class__.__name__, end='')

        #   sorting the probabilities of belonging to the positive class, P(y=+1 | x)
        indx = np.argsort(self.predictions_train_[:, 1])
        self.train_distrib_ = self.predictions_train_[indx, 1]
        self.train_labels_ = self.y_ext_[indx]

        if self.verbose > 0:
            print('done')

        return self

    def predict(self, X, predictions_test=None):
        """ Predict the class distribution of a testing bag

            First, predictions_test_ are computed (if needed, when predictions_test parameter is None) by
            `super().predict()` method.

            After that, the method computes the quantiles for the testing bag.

            Finally, the prevalences are computed using golden section search and the distance function of the object

            Parameters
            ----------
            X : array-like, shape (n_examples, n_features)
                Testing bag

            predictions_test : ndarray, shape (n_examples, n_classes) (default=None)
                They must be probabilities (the estimator used must have a `predict_proba` method)

                If estimator_test == None then predictions_test can not be None.
                If predictions_test is None, predictions for the testing examples are computed using the `predict_proba`
                method of estimator_test (it must be an actual estimator)

            Raises
            ------
            ValueError
                When estimator_test and predictions_test are at the same time None or not None

            Returns
            -------
            prevalences : ndarray, shape(n_classes, )
                Contains the predicted prevalence for each class
        """
        super().predict(X, predictions_test=predictions_test)

        if self.verbose > 0:
            print('Class %s: Computing the testing distribution...' % self.__class__.__name__, end='')

        #  sorting the probabilities of belonging to the positive class, P(y=+1 | x)
        self.union_distrib_ = np.hstack((self.train_distrib_, self.predictions_test_[:, 1]))
        indx = np.argsort(self.union_distrib_)
        self.union_distrib_ = self.union_distrib_[indx]
        test_label = np.max(self.classes_)+1
        self.union_labels_ = np.hstack((self.train_labels_,
                                        np.full(len(self.predictions_test_[:, 1]), test_label)))
        self.union_labels_ = self.union_labels_[indx]

        if self.verbose > 0:
            print('done')
            print('Class %s: Computing prevalences for testing distribution...' % self.__class__.__name__, end='')

        prevalences = golden_section_search(distance_func=sord, mixture_func=compute_sord_weights,
                                            test_distrib=self.union_distrib_, tol=self.tol,
                                            union_labels=self.union_labels_,
                                            classes=self.classes_)
        if self.verbose > 0:
            print('done')

        return prevalences


class CV_estimator(BaseEstimator, ClassifierMixin):

    def __init__(self, estimator, groups=None, cv='warn', n_jobs=None, fit_params=None, pre_dispatch='2*n_jobs',
                 averaged_predictions=True, voting='hard', verbose=0):
        self.estimator = estimator
        self.groups = groups
        self.cv = cv
        self.n_jobs = n_jobs
        self.fit_params = fit_params
        self.pre_dispatch = pre_dispatch
        self.averaged_predictions = averaged_predictions
        self.voting = voting
        self.verbose = verbose
        self.estimators_ = ()
        self.le_ = None
        self.classes_ = None
        self.X_train_ = None
        self.y_train_ = None

    def fit(self, X, y):
        self.X_train_ = X
        self.y_train_ = y
        self.classes_ = np.unique(y)
        self.le_ = LabelEncoder().fit(y)
        # check cv
        self.cv = check_cv(self.cv, y, classifier=is_classifier(self.estimator))
        # train CV and save the estimators for each fold
        cvresults = cross_validate(self.estimator, X, y,
                                   groups=self.groups, cv=self.cv, n_jobs=self.n_jobs,
                                   verbose=self.verbose, fit_params=self.fit_params, pre_dispatch=self.pre_dispatch,
                                   return_estimator=True)
        self.estimators_ = cvresults['estimator']

        return self

    def predict(self, X):
        if len(self.estimators_) == 0:
            raise NotFittedError('CV_estimator not fitted')

        preds = self._predict_proba(X)

        if self.averaged_predictions:
            if self.voting == 'soft':
                preds = np.mean(preds, axis=0)
                preds = np.argmax(preds, axis=1)
                preds = self.le_.inverse_transform(preds)
            else:
                # hard
                #  for each example (axis=2), compute the class with the largest probability
                aux = np.apply_along_axis(np.argmax, axis=2, arr=preds)
                # compute the number of votes for each class
                aux2 = np.apply_along_axis(lambda x: np.bincount(x, minlength=len(self.classes_)), axis=0, arr=aux)
                # compute the class with more votes
                aux3 = np.argmax(aux2, axis=0)
                # transforming label position to class label
                preds = self.le_.inverse_transform(aux3)
        else:
            # not averaging, so each pred is treated independently
            preds = preds.reshape(-1, len(self.classes_))
            # computing the class with largest probability
            preds = np.argmax(preds, axis=1)
            # transforming label position to class label
            preds = self.le_.inverse_transform(preds)
        return preds

    def predict_proba(self, X):
        if len(self.estimators_) == 0:
            raise NotFittedError('CV_estimator not fitted')

        preds = self._predict_proba(X)

        if self.averaged_predictions:
            preds = np.mean(preds, axis=0)
        else:
            preds = preds.reshape(-1, len(self.classes_))
        return preds

    def _predict_proba(self, X):
        n_examples = X.shape[0]
        if (issparse(X) and (X != self.X_train_).nnz == 0) or np.array(X == self.X_train_).all():
            # predicting over training examples, same partitions
            #  computing number of repetitions
            n_preds = 0
            for (train_index, test_index) in self.cv.split(self.X_train_, self.y_train_):
                n_preds = n_preds + len(test_index)
            n_repeats = n_preds // n_examples
            #  storing predictions
            preds = np.zeros((n_repeats, n_examples, len(self.classes_)), dtype=float)
            n_rep = 0
            n_preds = 0
            for nfold, (train_index, test_index) in enumerate(self.cv.split(self.X_train_, self.y_train_)):
                X_test = X[test_index]
                preds[n_rep, test_index, :] = self.estimators_[nfold].predict_proba(X_test)
                n_preds = n_preds + len(test_index)
                if n_preds == n_examples:
                    n_rep += 1
                    n_preds = 0
        else:
            #   it is a test sample, predicting with each estimator
            preds = np.zeros((len(self.estimators_), n_examples, len(self.classes_)), dtype=float)
            for n_est, est in enumerate(self.estimators_):
                preds[n_est, :, :] = est.predict_proba(X)

        return preds

