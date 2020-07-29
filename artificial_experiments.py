import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import zero_one_loss, brier_score_loss
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold

from qbase import UsingClassifiers, CV_estimator, CC, AC, PAC, DFy, QUANTy, SORDy
from utils import absolute_error, relative_absolute_error, squared_error, l1, l2

import time

def indices_to_one_hot(data, n_classes):
    """Convert an iterable of indices to one-hot encoded labels."""
    targets = np.array(data).reshape(-1)
    return np.eye(n_classes)[targets]


def run_experiment(est_name, seed, dim, param, ntrain, ntest, nreps, nbags, nfolds, save_all):
    """ Run a single experiment

        Parameters
        ----------
        est_name: str
            Name of the estimator. 'LR' or 'SVM-RBF'

        seed: int
            Seed of the experiment

        dim: int
            Dimension of the dataset, 1 or 2

        param: int, str
            Extra oarameter for the definition of the problem.
            If dim==1, this value is the std.
            If dim=2 this value is an string to indicate if the dataset is the one designed to test HDX

        ntrain : list
            List with the number of training examples that must be tested, e.g.,[50, 100, 200]

        ntest: int
            Number of testing instances in each bag

        nreps: int
            Number of training datasets created

        nbags: int
            Number of testing bags created for each training datasets.
            The total number of experiments will be nreps * nbags

        nfolds: int
            Number of folds used to estimate the training distributions by the methods AC, HDy and EDy

        save_all: bool
            True if the results of each single experiment must be saved
    """

    # range of testing prevalences
    low = round(ntest * 0.05)
    high = round(ntest * 0.95)

    if est_name == 'LR':
        estimator = LogisticRegression(C=1, random_state=seed, max_iter=10000, solver='liblinear')
    else:
        estimator = SVC(C=1, kernel='rbf', random_state=seed, max_iter=10000, gamma=0.2, probability=True)

    rng = np.random.RandomState(seed)

    #   methods
    cc = CC()
    ac = AC()
    pac = PAC()
    sordy = SORDy()
    #  hdys
    pdfy_hd_4 = DFy(distribution_function='PDF', n_bins=4, distance='HD')
    pdfy_hd_8 = DFy(distribution_function='PDF', n_bins=8, distance='HD')
    pdfy_hd_12 = DFy(distribution_function='PDF', n_bins=12, distance='HD')
    pdfy_hd_16 = DFy(distribution_function='PDF', n_bins=16, distance='HD')
    #  cdf l1
    cdfy_l1_8 = DFy(distribution_function='CDF', n_bins=8, distance='L1')
    cdfy_l1_16 = DFy(distribution_function='CDF', n_bins=16, distance='L1')
    cdfy_l1_32 = DFy(distribution_function='CDF', n_bins=32, distance='L1')
    cdfy_l1_64 = DFy(distribution_function='CDF', n_bins=64, distance='L1')
    #  cdf l1
    # cdfy_l2_8 = DFy(distribution_function='CDF', n_bins=8, distance='L2')
    # cdfy_l2_16 = DFy(distribution_function='CDF', n_bins=16, distance='L2')
    # cdfy_l2_32 = DFy(distribution_function='CDF', n_bins=32, distance='L2')
    # cdfy_l2_64 = DFy(distribution_function='CDF', n_bins=64, distance='L2')
    #  QUANTy-L1
    # quanty_l1_4 = QUANTy(n_quantiles=4, distance=l1)
    # quanty_l1_10 = QUANTy(n_quantiles=10, distance=l1)
    # quanty_l1_20 = QUANTy(n_quantiles=20, distance=l1)
    # quanty_l1_40 = QUANTy(n_quantiles=40, distance=l1)
    #  QUANTy-L2
    quanty_l2_4 = QUANTy(n_quantiles=4, distance=l2)
    quanty_l2_10 = QUANTy(n_quantiles=10, distance=l2)
    quanty_l2_20 = QUANTy(n_quantiles=20, distance=l2)
    quanty_l2_40 = QUANTy(n_quantiles=40, distance=l2)

    #   methods
    methods = [cc, ac, pac, sordy,
               pdfy_hd_4, pdfy_hd_8, pdfy_hd_12, pdfy_hd_16,
               cdfy_l1_8, cdfy_l1_16, cdfy_l1_32, cdfy_l1_64,
               # cdfy_l2_8, cdfy_l2_16, cdfy_l2_32, cdfy_l2_64,
               # quanty_hd_4, quanty_hd_8, quanty_hd_12, quanty_hd_16,
               # quanty_l1_4, quanty_l1_8, quanty_l1_12, quanty_l1_16,
               quanty_l2_4, quanty_l2_10, quanty_l2_20, quanty_l2_40]

    methods_names = ['CC', 'AC', 'PAC', 'SORDy',
                     'PDFy_hd_4', 'PDFy_hd_8', 'PDFy_hd_12', 'PDFy_hd_16',
                     'CDFy_l1_8', 'CDFy_l1_16', 'CDFy_l1_32', 'CDFy_l1_64',
                     # 'CDFy_l2_8', 'CDFy_l2_16', 'CDFy_l2_32', 'CDFy_l2_64',
                     # 'QUANTy_l1_4', 'QUANTy_l1_10', 'QUANTy_l1_20', 'QUANTy_l1_40',
                     'QUANTy_l2_4', 'QUANTy_l2_10', 'QUANTy_l2_20', 'QUANTy_l2_40']

    #   to store the results
    mae_results = np.zeros((len(methods_names), len(ntrain)))
    sqe_results = np.zeros((len(methods_names), len(ntrain)))
    mrae_results = np.zeros((len(methods_names), len(ntrain)))
    classif_results = np.zeros((2, len(ntrain)))

    std1 = std2 = mu3 = mu4 = cov1 = cov2 = cov3 = cov4 = 0
    if dim == 1:
        # 1D
        mu1 = -1
        std1 = param
        mu2 = 1
        std2 = std1
    else:
        # 2D
        mu1 = [-1.00, 1.00]
        mu2 = [1.00, 1.00]
        mu3 = [1.00, -1.00]

        cov1 = [[0.4, 0],
                [0, 0.4]]
        cov2 = cov1
        cov3 = cov1

        x1 = rng.multivariate_normal(mu1, cov1, 400)
        x3 = rng.multivariate_normal(mu3, cov3, 400)

        plt.scatter(np.vstack((x1[:, 0], x3[:, 0])), np.vstack((x1[:, 1], x3[:, 1])), c='r', marker='+', s=12,
                    label='Class \u2212' + '1')

        if param == 'HDX':
            mu4 = [-1.00, -1.00]
            cov4 = cov1

            x2 = rng.multivariate_normal(mu2, cov2, 400)
            x4 = rng.multivariate_normal(mu4, cov4, 400)

            plt.scatter(np.vstack((x2[:, 0], x4[:, 0])), np.vstack((x2[:, 1], x4[:, 1])),
                        c='b', marker='x', s=8, label='Class +1')
        else:
            x2 = rng.multivariate_normal(mu2, cov2, 800)

            plt.scatter(x2[:, 0], x2[:, 1], c='b', marker='x', s=8, label='Class +1')

        plt.xlabel('$x_1$')
        plt.ylabel('$x_2$')
        plt.legend(loc='best')
        plt.savefig('./artificial-2D-' + param + '.png', dpi=300)

    name_file = 'times-avg-artificial-' + str(dim) + 'D-' + str(param) + '-' + est_name + '-rep' + str(nreps) + \
                '-ntest' + str(ntest) + '.txt'
    file_times = open(name_file, 'w')
    file_times.write('#examples, ')
    for index, m in enumerate(methods_names):
        file_times.write('%s, ' % m)

    for k in range(len(ntrain)):

        all_mae_results = np.zeros((len(methods_names), nreps * nbags))
        all_sqe_results = np.zeros((len(methods_names), nreps * nbags))
        all_mrae_results = np.zeros((len(methods_names), nreps * nbags))

        execution_times = np.zeros(len(methods_names))

        print()
        print('#Training examples ', ntrain[k], 'Rep#', end=' ')

        for rep in range(nreps):

            print(rep+1, end=' ')

            if dim == 1:
                x_train = np.vstack(((std1 * rng.randn(ntrain[k], 1) + mu1), (std2 * rng.randn(ntrain[k], 1) + mu2)))
            else:
                if param == 'HDX':
                    x_train = np.vstack((rng.multivariate_normal(mu1, cov1, ntrain[k] // 2),
                                         rng.multivariate_normal(mu3, cov3, ntrain[k] - ntrain[k] // 2),
                                         rng.multivariate_normal(mu2, cov2, ntrain[k] // 2),
                                         rng.multivariate_normal(mu4, cov4, ntrain[k] - ntrain[k] // 2)))
                else:
                    x_train = np.vstack((rng.multivariate_normal(mu1, cov1, ntrain[k] // 2),
                                         rng.multivariate_normal(mu3, cov3, ntrain[k] - ntrain[k] // 2),
                                         rng.multivariate_normal(mu2, cov2, ntrain[k])))

            y_train = np.hstack((np.zeros(ntrain[k], dtype=int), np.ones(ntrain[k], dtype=int)))

            skf_train = StratifiedKFold(n_splits=nfolds, shuffle=True, random_state=seed + rep)
            estimator_train = CV_estimator(estimator=estimator, n_jobs=None, cv=skf_train)

            estimator_train.fit(x_train, y_train)

            predictions_train = estimator_train.predict_proba(x_train)

            for nmethod, method in enumerate(methods):
                if isinstance(method, UsingClassifiers):
                    method.fit(X=x_train, y=y_train, predictions_train=predictions_train)
                else:
                    method.fit(X=x_train, y=y_train)

            #estimator_test = estimator
            #estimator_test.fit(x_train, y_train)
            estimator_test = estimator_train

            for n_bag in range(nbags):

                ps = rng.randint(low, high, 1)
                ps = np.append(ps, [0, ntest])
                ps = np.diff(np.sort(ps))

                if dim == 1:
                    x_test = np.vstack(((std1 * rng.randn(ps[0], 1) + mu1), (std2 * rng.randn(ps[1], 1) + mu2)))
                else:
                    if param == 'HDX':
                        x_test = np.vstack((rng.multivariate_normal(mu1, cov1, ps[0] // 2),
                                            rng.multivariate_normal(mu3, cov3, ps[0] - ps[0] // 2),
                                            rng.multivariate_normal(mu2, cov2, ps[1] // 2),
                                            rng.multivariate_normal(mu4, cov4, ps[1] - ps[1] // 2)))
                    else:
                        x_test = np.vstack((rng.multivariate_normal(mu1, cov1, ps[0] // 2),
                                            rng.multivariate_normal(mu3, cov3, ps[0] - ps[0] // 2),
                                            rng.multivariate_normal(mu2, cov2, ps[1])))

                y_test = np.hstack((np.zeros(ps[0], dtype=int), np.ones(ps[1], dtype=int)))

                predictions_test = estimator_test.predict_proba(x_test)

                # Error
                classif_results[0, k] = classif_results[0, k] + zero_one_loss(np.array(y_test),
                                                                              np.argmax(predictions_test, axis=1))
                # Brier loss
                classif_results[1, k] = classif_results[1, k] + brier_score_loss(indices_to_one_hot(y_test, 2)[:, 0],
                                                                                 predictions_test[:, 0])

                prev_true = ps[1] / ntest

                for nmethod, method in enumerate(methods):

                    t = time.process_time()
                    if isinstance(method, UsingClassifiers):
                        p_predicted = method.predict(X=None, predictions_test=predictions_test)[1]
                    else:
                        p_predicted = method.predict(X=x_test)[1]
                    elapsed_time = time.process_time()
                    execution_times[nmethod] = execution_times[nmethod] + elapsed_time - t

                    all_mae_results[nmethod, rep * nbags + n_bag] = absolute_error(prev_true, p_predicted)
                    all_mrae_results[nmethod, rep * nbags + n_bag] = relative_absolute_error(prev_true, p_predicted)
                    all_sqe_results[nmethod, rep * nbags + n_bag] = squared_error(prev_true, p_predicted)

                    mae_results[nmethod, k] = mae_results[nmethod, k] + all_mae_results[nmethod, rep * nbags + n_bag]
                    mrae_results[nmethod, k] = mrae_results[nmethod, k] + all_mrae_results[nmethod, rep * nbags + n_bag]
                    sqe_results[nmethod, k] = sqe_results[nmethod, k] + all_sqe_results[nmethod, rep * nbags + n_bag]

        execution_times = execution_times / (nreps * nbags)

        file_times.write('\n%d, ' % ntrain[k])
        for i in execution_times:
            file_times.write('%.5f, ' % i)


        if save_all:
            name_file = 'results-all-mae-artificial-' + str(dim) + 'D-' + str(param) + '-' + est_name + \
                        '-rep' + str(nreps) + '-value' + str(ntrain[k]) + '-ntest' + str(ntest) + '.txt'
            file_all = open(name_file, 'w')

            for method_name in methods_names:
                file_all.write('%s,' % method_name)
            file_all.write('\n')
            for nrep in range(nreps):
                for n_bag in range(nbags):
                    for n_method in range(len(methods_names)):
                        file_all.write('%.5f, ' % all_mae_results[n_method, nrep * nbags + n_bag])
                    file_all.write('\n')
            file_all.close()

            name_file = 'results-all-mrae-artificial-' + str(dim) + 'D-' + str(param) + '-' + est_name + \
                        '-rep' + str(nreps) + '-value' + str(ntrain[k]) + '-ntest' + str(ntest) + '.txt'
            file_all = open(name_file, 'w')

            for method_name in methods_names:
                file_all.write('%s,' % method_name)
            file_all.write('\n')
            for nrep in range(nreps):
                for n_bag in range(nbags):
                    for n_method in range(len(methods_names)):
                        file_all.write('%.5f, ' % all_mrae_results[n_method, nrep * nbags + n_bag])
                    file_all.write('\n')
            file_all.close()

            name_file = 'results-all-sqe-artificial-' + str(dim) + 'D-' + str(param) + '-' + est_name + \
                        '-rep' + str(nreps) + '-value' + str(ntrain[k]) + '-ntest' + str(ntest) + '.txt'
            file_all = open(name_file, 'w')

            for method_name in methods_names:
                file_all.write('%s,' % method_name)
            file_all.write('\n')
            for nrep in range(nreps):
                for n_bag in range(nbags):
                    for n_method in range(len(methods_names)):
                        file_all.write('%.5f, ' % all_sqe_results[n_method, nrep * nbags + n_bag])
                    file_all.write('\n')
            file_all.close()

    file_times.close()

    mae_results = mae_results / (nreps * nbags)
    mrae_results = mrae_results / (nreps * nbags)
    sqe_results = sqe_results / (nreps * nbags)
    classif_results = classif_results / (nreps * nbags)

    name_file = 'results-avg-artificial-' + str(dim) + 'D-' + str(param) + '-' + est_name + '-rep' + str(nreps) + \
                '-ntest' + str(ntest) + '.txt'
    file_avg = open(name_file, 'w')
    file_avg.write('MAE\n')
    file_avg.write('#examples, Error, ')
    for index, m in enumerate(methods_names):
        file_avg.write('%s, ' % m)
    file_avg.write('BrierLoss')
    for index, number in enumerate(ntrain):
        file_avg.write('\n%d, ' % number)
        # Error
        file_avg.write('%.5f, ' % classif_results[0, index])
        for i in mae_results[:, index]:
            file_avg.write('%.5f, ' % i)
        # Brier loss
        file_avg.write('%.5f' % classif_results[1, index])

    file_avg.write('\n\nMRAE\n')
    file_avg.write('#examples, Error, ')
    for index, m in enumerate(methods_names):
        file_avg.write('%s, ' % m)
    file_avg.write('BrierLoss')
    for index, number in enumerate(ntrain):
        file_avg.write('\n%d, ' % number)
        # Error
        file_avg.write('%.5f, ' % classif_results[0, index])
        for i in mrae_results[:, index]:
            file_avg.write('%.5f, ' % i)
        # Brier loss
        file_avg.write('%.5f' % classif_results[1, index])

    file_avg.write('\n\nSQE\n')
    file_avg.write('#examples, Error, ')
    for index, m in enumerate(methods_names):
        file_avg.write('%s, ' % m)
    file_avg.write('BrierLoss')
    for index, number in enumerate(ntrain):
        file_avg.write('\n%d, ' % number)
        # Error
        file_avg.write('%.5f, ' % classif_results[0, index])
        for i in sqe_results[:, index]:
            file_avg.write('%.5f, ' % i)
        # Brier loss
        file_avg.write('%.5f' % classif_results[1, index])

    file_avg.close()


# MAIN
# 1D synthetic experiments
run_experiment(est_name='LR', seed=42, dim=1, param=0.5, ntrain=[50, 100, 200, 500, 1000, 2000], ntest=50,
               nreps=40, nbags=50, nfolds=50, save_all=True)
run_experiment(est_name='LR', seed=42, dim=1, param=0.5, ntrain=[50, 100, 200, 500, 1000, 2000], ntest=100,
               nreps=40, nbags=50, nfolds=50, save_all=True)
run_experiment(est_name='LR', seed=42, dim=1, param=0.5, ntrain=[50, 100, 200, 500, 1000, 2000], ntest=200,
               nreps=40, nbags=50, nfolds=50, save_all=True)
run_experiment(est_name='LR', seed=42, dim=1, param=0.5, ntrain=[50, 100, 200, 500, 1000, 2000], ntest=500,
               nreps=40, nbags=50, nfolds=50, save_all=True)
run_experiment(est_name='LR', seed=42, dim=1, param=0.5, ntrain=[50, 100, 200, 500, 1000, 2000], ntest=1000,
               nreps=40, nbags=50, nfolds=50, save_all=True)
run_experiment(est_name='LR', seed=42, dim=1, param=0.5, ntrain=[50, 100, 200, 500, 1000, 2000], ntest=2000,
               nreps=40, nbags=50, nfolds=50, save_all=True)

run_experiment(est_name='LR', seed=42, dim=1, param=1.0, ntrain=[50, 100, 200, 500, 1000, 2000], ntest=50,
               nreps=40, nbags=50, nfolds=50, save_all=True)
run_experiment(est_name='LR', seed=42, dim=1, param=1.0, ntrain=[50, 100, 200, 500, 1000, 2000], ntest=100,
               nreps=40, nbags=50, nfolds=50, save_all=True)
run_experiment(est_name='LR', seed=42, dim=1, param=1.0, ntrain=[50, 100, 200, 500, 1000, 2000], ntest=200,
               nreps=40, nbags=50, nfolds=50, save_all=True)
run_experiment(est_name='LR', seed=42, dim=1, param=1.0, ntrain=[50, 100, 200, 500, 1000, 2000], ntest=500,
               nreps=40, nbags=50, nfolds=50, save_all=True)
run_experiment(est_name='LR', seed=42, dim=1, param=1.0, ntrain=[50, 100, 200, 500, 1000, 2000], ntest=1000,
               nreps=40, nbags=50, nfolds=50, save_all=True)
run_experiment(est_name='LR', seed=42, dim=1, param=1.0, ntrain=[50, 100, 200, 500, 1000, 2000], ntest=2000,
               nreps=40, nbags=50, nfolds=50, save_all=True)


