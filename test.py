
import ast

import joblib
from pandas import read_csv
import plotly.graph_objects as go
from plotly.graph_objs.layout import scene
from plotly.subplots import make_subplots
from typing import Tuple, Callable
import pickle
from sklearn import tree
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier


def split_data(X_file_name, y_file_name):
    """
    Splits the data.
    :param X_file_name: data file name
    :param y_file_name: labels file name
    :return: X_train, y_train, X_dev, y_dev, X_test, y_test
    """
    X = pd.read_csv(X_file_name)
    y = pd.read_csv(y_file_name)
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.3)
    #X_dev, X_test, y_dev, y_test = sklearn.model_selection.train_test_split(X_test, y_test, test_size=0.5)
    return X_train, y_train, X_test, y_test


def tree_baseline(X_train, y_train):
    # min_samples_per_leaf = [8, (X_train.shape[0] / 16),(X_train.shape[0] / 8)]
    min_samples_per_leaf = [(X_train.shape[0] / 8)]
    res = []
    for m in min_samples_per_leaf:
        tree_clf = sklearn.tree.DecisionTreeClassifier().fit(X_train, y_train)
        y_train_pred = tree_clf.predict(X_train)
        micro_score = sklearn.metrics.f1_score(y_train, y_train_pred, average="micro")
        macro_score = sklearn.metrics.f1_score(y_train, y_train_pred, average="macro")
        print("min_sample_per_leaf = ", m)
        print("           micro score: ", micro_score)
        print("           macro score: ", macro_score)
        res.append(y_train_pred)
    return res

def knn_baseline(X_train, y_train, num_neighbors):
    # min_samples_per_leaf = [8, (X_train.shape[0] / 16),(X_train.shape[0] / 8)]
    res = []
    model = nearest_neighbors(num_neighbors)
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    return y_train_pred







import pandas as pd
import sklearn


def to_binary(df):  # TODO
    to_binary = ['Ivi -Lymphovascular invasion']


def drop_cols(df):
    cols_to_drop = ["KI67 protein", 'Diagnosis date', 'Surgery date1', 'Surgery date2', 'Surgery date3',
                    "surgery before or after-Activity date", "surgery before or after-Actual activity",
                    "id-hushed_internalpatientid", "er", "pr", "Ivi -Lymphovascular invasion", "Her2",
                    "Tumor depth"]
    for col in cols_to_drop:
        df = df.drop(col, axis=1)

    return df


# dates: diagnosis and surgeries


def set_null(df):
    df = df.replace({np.nan: None})
    df['Histopatological degree'] = df['Histopatological degree'].str.replace('Null', "0")
    return df


def set_dummies(df):
    to_dummies = [' Form Name', ' Hospital', 'User Name', 'Basic stage', 'Histological diagnosis', "Surgery name1",
                  "Surgery name2", "Surgery name3", "M -metastases mark (TNM)", "N -lymph nodes mark (TNM)",
                  "T -Tumor mark (TNM)", "Lymphatic penetration", "Stage", "Side", "Histopatological degree",
                  'Margin Type']
    for dummy in to_dummies:
        df = pd.get_dummies(df, prefix=dummy, columns=[dummy])
    return df


def check_outliers(df):
    df = df[df["Age"] > 0]
    df = df[df["Age"] < 100]

    df['Nodes exam'] = pd.to_numeric(df['Nodes exam'],
                                     errors='coerce').fillna(0)
    df = df[df['Nodes exam'] >= 0]
    df['Positive nodes'] = pd.to_numeric(df["Positive nodes"],
                                         errors='coerce').fillna(0)
    df = df[df['Positive nodes'] >= 0]

    df["Surgery sum"] = pd.to_numeric(df["Surgery sum"],
                                      errors='coerce').fillna(0)
    df = df[df["Surgery sum"] >= 0]
    df["Tumor width"] = pd.to_numeric(df["Tumor width"], errors='coerce').fillna(0)
    df = df[df["Tumor width"] >= 0]

    return df


def prepare_train(df):
    df.columns = df.columns.str.replace('אבחנה-', '')
    df = drop_cols(df)
    df = check_outliers(df)
    df = set_null(df)
    df = set_dummies(df)

    return df


def create_responses(df):
    responses = ['BON - Bones', 'LYM - Lymph nodes', 'HEP - Hepatic', 'PUL - Pulmonary', 'PLE - Pleura', 'SKI - Skin',
                 'OTH - Other', 'BRA - Brain', 'MAR - Bone Marrow', 'PER - Peritoneum', 'ADR - Adrenals']
    for response in responses:
        df[response] = np.zeros(df.shape[0], )
    return df


def prepare_labels(df):
    df.columns = df.columns.str.replace('אבחנה-', '')
    df = create_responses(df)
    all_lists = df['Location of distal metastases']
    for index, label_list in enumerate(all_lists):
        sublist = ast.literal_eval(label_list)
        for sub in sublist:
            df.loc[index, sub] += 1
    df = df.iloc[:, 1:]
    return df


def match_data(X_train, X_test):
    train_cols = X_train.columns
    test_cols = X_test.columns
    for col in test_cols:
        if col not in train_cols:
            X_test = X_test.drop(col, axis=1)
    for col in train_cols:
        if col not in test_cols:
            X_test[col] = np.zeros(len(X_test), )
    X_test = X_test.reindex(columns=train_cols)
    return X_train, X_test


def nearest_neighbors(num_neighbors):

    classifier = KNeighborsClassifier(n_neighbors=num_neighbors)
    # classifier.fit(X_train, y_train)
    # y_pred = classifier.predict(X_dev)
    return classifier

def multi_classification_pred_knn(X_train, y_train, num_neighbors):
    predictions_knn = np.empty((11, y_train.shape[0]))
    for i in range(y_train.shape[1]):
        knn_pred = (knn_baseline(X_train, y_train[y_train.columns[i]], num_neighbors))
        predictions_knn[i] = knn_pred
    cols = y_train.columns
    res = pd.DataFrame(predictions_knn.transpose(), columns=cols)
    return res




def cross_validation(estimator, X: pd.DataFrame, y: pd.DataFrame, cv, scoring: Callable[[np.ndarray, np.ndarray, ...], Tuple[float, float]],num_neighbors):
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds

    """
    micro = 0
    macro = 0
    X_cols = X.columns
    y_cols = y.columns
    X_split = np.array_split(X.to_numpy(), cv)
    y_split = np.array_split(y.to_numpy(), cv)

    for i in range(cv):
        train = pd.DataFrame(columns=X_cols)
        train_res = pd.DataFrame(columns=y_cols)
        for j in range(cv):
            if (i != j):
                if train.shape[0] == 0:
                    frames = [train_res, pd.DataFrame(y_split[j], columns=y_cols)]
                    train = pd.DataFrame(X_split[j], columns=X_cols)
                    train_res = pd.concat(frames)
                else:
                    frames1 = [train, pd.DataFrame(X_split[j], columns=X_cols)]
                    frames2 = [train_res, pd.DataFrame(y_split[j], columns=y_cols)]
                    train = pd.concat(frames1)
                    train_res = pd.concat(frames2)
        validation = X_split[i]
        validation = pd.DataFrame(validation, columns=X_cols)
        train, validation = match_data(train, validation)
        validation_res = prepare_labels(pd.DataFrame(y_split[i], columns=y_cols))
        estimator.fit(train, train_res)
        y_dev_pred = multi_classification_pred_knn(validation, validation_res, num_neighbors)
        #y_dev_pred = prepare_labels(pd.DataFrame(y_dev_pred, columns=y_cols))
        micro += scoring(validation_res, y_dev_pred)[0]
        macro += scoring(validation_res, y_dev_pred)[1]
    return micro * (1 / cv), macro * (1 / cv)


def evaluate(y_pred, y_dev):

    tp = 0
    macro =0
    micro = 0
    fp = 0
    for i in range(y_pred.shape[1]):
        res_col = y_pred[y_pred.columns[i]].to_numpy()
        cur_true = y_dev[y_dev.columns[i]].to_numpy()

        fp_temp = np.sum(res_col - cur_true == np.ones(y_pred.shape[0], ))
        fp += fp_temp
        tp_temp = np.sum(res_col * cur_true)
        tp += tp_temp
        if tp_temp + fp_temp != 0:
            macro += tp_temp / (tp_temp + fp_temp)
    macro /= y_pred.shape[1]

    if tp + fp != 0:
        micro = tp / (tp + fp)

    return micro, macro


if __name__ == '__main__':
    np.random.seed(0)
    X_train, y_train, X_test, y_test = split_data("train.feats.csv", "train.labels.0 (1).csv")
    X_train['אבחנה-Location of distal metastases'] = y_train[
        'אבחנה-Location of distal metastases']

    X_train['אבחנה-Location of distal metastases'] = y_train['אבחנה-Location of distal metastases']
    X_train = prepare_train(X_train)
    y_train = pd.DataFrame()
    y_train["Location of distal metastases"] = X_train["Location of distal metastases"]
    y_train = y_train.reset_index(drop=True)
    X_train = X_train.drop("Location of distal metastases", axis=1)
    X_train = X_train.reset_index(drop=True)




    # find best k
    model = None
    neighbors_range = np.arange(4, 10)

    micros = np.zeros(len(neighbors_range), )
    macros = np.zeros(len(neighbors_range), )
    for index, num_neighbors in enumerate(neighbors_range):
        model = nearest_neighbors(num_neighbors)
        results = cross_validation(estimator=model,
                                   X=X_train,
                                   y=y_train,
                                   cv=4,
                                   scoring=evaluate, num_neighbors=num_neighbors)
        micro, macro = results
        micros[index]=micro
        macros[index]=macro

    fig1 = go.Figure([go.Scatter(name='micro',
                                 x=neighbors_range,
                                 y=micros,
                                 mode='lines',
                                 marker=dict(color='green', size=10),
                                 ),
                      go.Scatter(
                          name='macro',
                          x=neighbors_range,
                          y=macros,
                          mode='lines',
                          marker=dict(color='blue', size=10)
                      )
                      ])
    fig1.update_layout(
        xaxis_title='Number of KN Neighbors',
        yaxis_title='F1 Score',
        title='Choosing the Hyperparameter of K nearest Neighbors using cross validation',
        hovermode="x",
        font=dict(
            size=18,
            color="black"
        )
    )
    fig1.show()


