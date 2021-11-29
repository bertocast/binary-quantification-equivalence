import numpy as np
import os
import glob
import pandas as pd
import warnings

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import DataConversionWarning
from sklearn.metrics import zero_one_loss, brier_score_loss
from qbase import UsingClassifiers, CV_estimator, CC, AC, PAC, DFy, QUANTy, SORDy
from utils import absolute_error, l2
from utils import create_bags_with_multiple_prevalence
from pandas.core.common import SettingWithCopyWarning
from joblib import Parallel, delayed

pd.set_option('display.float_format', lambda x: '%.5f' % x)
warnings.simplefilter("ignore", DataConversionWarning)
warnings.simplefilter("ignore", SettingWithCopyWarning)


def main():
    # configuration params
    num_reps = 40
    num_bags = 50
    num_folds = 50
    master_seed = 2032

    estimator_grid = {
        "n_estimators": [10, 20, 40, 70, 100, 200, 250, 500],
        "max_depth": [1, 5, 10, 15, 20, 25, 30],
        "min_samples_leaf": [1, 2, 5, 10, 20]}

    datasets_dir = "./datasets"
    dataset_files = [file for file in glob.glob(os.path.join(datasets_dir, "*.csv"))]
    dataset_names = [os.path.split(name)[-1][:-4] for name in dataset_files]
    print("There are a total of {} datasets.".format(len(dataset_names)))

    filename_out = "results_ICML20_parallel_" + str(num_reps) + "x" + str(num_bags)

    methods_names = ['AC', 'CC', 'CDFy_l1_64', 'PAC', 'PDFy_hd_8', 'QUANTy_l2_20', 'SORDy']

    solutions = Parallel(n_jobs=-1)(delayed(compute_parallel)(dataset_names, dataset_files, rep, master_seed,
                                                              methods_names, filename_out, estimator_grid,
                                                              num_bags, num_folds)
                                    for rep in range(num_reps))

    total_errors_df = []
    for i in range(len(solutions)):
        for j in range (len(dataset_names)):
             total_errors_df.append(solutions[i][j])

    total_errors_df = pd.concat(total_errors_df)
    total_errors_df.to_csv(filename_out+  "_all.csv", index=None)

    means_df = total_errors_df.groupby(['dataset', 'method'])[['mae']].agg(["mean"]).unstack().round(5)
    means_df.to_csv(filename_out+ "_means_mae.csv", header=methods_names)
    means_df2 = total_errors_df.groupby(['dataset', 'method'])[['error_clf']].agg(["mean"]).unstack().round(5)
    means_df2.to_csv(filename_out+ "_means_error.csv", header=methods_names)
    means_df3 = total_errors_df.groupby(['dataset', 'method'])[['brier_clf']].agg(["mean"]).unstack().round(5)
    means_df3.to_csv(filename_out+ "_means_brier.csv", header=methods_names)
    print(means_df)


def compute_parallel(dataset_names, dataset_files, rep, master_seed, methods_names, filename_out,
                     estimator_grid, num_bags, num_folds):
    errors_df=[]
    for dname, dfile in zip(dataset_names, dataset_files):
        current_seed = master_seed + rep
        print("*** Training over {}, rep {}".format(dname, rep + 1))
        errors_df.append(train_on_a_dataset(methods_names, dname, dfile, filename_out, estimator_grid,
                                            master_seed, current_seed, rep, num_bags, num_folds))
    return errors_df

def indices_to_one_hot(data, n_classes):
    """Convert an iterable of indices to one-hot encoded labels."""
    targets = np.array(data).reshape(-1)
    return np.eye(n_classes)[targets]


def g_mean(clf, X, y):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y, clf.predict(X), labels=clf.classes_)
    fpr = cm[0, 1] / float(cm[0, 1] + cm[0, 0])
    tpr = cm[1, 1] / float(cm[1, 1] + cm[1, 0])
    return np.sqrt((1 - fpr) * tpr)


def normalize(X_train, X_test):
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test


def load_data(dfile, current_seed):
    df = pd.read_csv(dfile, header=None)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values.astype(np.int)
    if -1 in np.unique(y):
        y[y == -1] = 0
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=current_seed)
    X_train, X_test = normalize(X_train, X_test)
    return X_train, X_test, y_train, y_test


def select_estimator(X_train, y_train, estimator_grid, master_seed, current_seed):
    clf_ = RandomForestClassifier(random_state=master_seed, class_weight='balanced')
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=current_seed)
    gs = GridSearchCV(clf_, param_grid=estimator_grid, verbose=False, cv=skf, scoring=g_mean, n_jobs=-1, iid=False)
    gs.fit(X_train, y_train)
    # print("Best grid params:", gs.best_params_)
    clf = gs.best_estimator_
    return clf


def train_on_a_dataset(methods_names, dname, dfile, filename_out,  estimator_grid,
                       master_seed, current_seed, n_rep, num_bags, num_folds):

    columns = ['dataset', 'method', "rep", "bag", 'truth', 'predictions', 'mae', "error_clf", "brier_clf"]
    errors_df = pd.DataFrame(columns=columns)

    X_train, X_test, y_train, y_test = load_data(dfile, current_seed)
    folds = np.min([num_folds, np.min(np.unique(y_train, return_counts=True)[1])])

    clf = select_estimator(X_train, y_train, estimator_grid, master_seed, current_seed)

    skf_train = StratifiedKFold(n_splits=folds, shuffle=True, random_state=current_seed)
    estimator_train = CV_estimator(estimator=clf, n_jobs=None, cv=skf_train)

    estimator_train.fit(X_train, y_train)
    predictions_train = estimator_train.predict_proba(X_train)

    #   methods
    cc = CC()
    ac = AC()
    pac = PAC()
    pdfy_hd_8 = DFy(distribution_function='PDF', n_bins=8, distance='HD')
    cdfy_l1_64 = DFy(distribution_function='CDF', n_bins=64, distance='L1')
    quanty_l2_20 = QUANTy(n_quantiles=20, distance=l2)
    sordy = SORDy()

    methods = [ac, cc, cdfy_l1_64, pac, pdfy_hd_8, quanty_l2_20, sordy]

    for nmethod, method in enumerate(methods):
        if isinstance(method, UsingClassifiers):
            method.fit(X=X_train, y=y_train, predictions_train=predictions_train)
        else:
            method.fit(X=X_train, y=y_train)

    estimator_test = estimator_train

    for n_bag, (X_test_, y_test_, prev_true, unused) in enumerate(
               create_bags_with_multiple_prevalence(X_test, y_test, num_bags, current_seed)):

        predictions_test = estimator_test.predict_proba(X_test_)

        # Error
        error_clf = zero_one_loss(np.array(y_test_), np.argmax(predictions_test, axis=1))
        # Brier loss
        brier_clf = brier_score_loss(indices_to_one_hot(y_test_, 2)[:, 0], predictions_test[:, 0])

        prev_true = prev_true[1]
        prev_preds = []

        for nmethod, method in enumerate(methods):
            if isinstance(method, UsingClassifiers):
                p_predicted = method.predict(X=None, predictions_test=predictions_test)[1]
            else:
                p_predicted = method.predict(X=X_test_)[1]
            prev_preds.append(p_predicted)

        for n_method, (method, prev_pred) in enumerate(zip(methods_names, prev_preds)):
            mae = absolute_error(prev_true, prev_pred)
            errors_df = errors_df.append(
                pd.DataFrame([[dname, method, n_rep, n_bag, prev_true, prev_pred, mae, error_clf, brier_clf]],
                             columns=columns))

    # uncomment to save intermediate results
    # errors_df.to_csv(filename_out + "_all.csv", mode='a', index=None)

    return errors_df

if __name__ == '__main__':
    main()
