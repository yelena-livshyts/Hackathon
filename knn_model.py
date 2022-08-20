
import ast
import plotly.graph_objects as go
from typing import Tuple, Callable
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import sklearn

def split_data(X_file_name, y_file_name):
    """
    Splits the data. test size is 0.3 from all train data received.
    :param X_file_name: data file name
    :param y_file_name: labels file name
    :return: X_train, y_train, X_dev, y_dev, X_test, y_test
    """
    X = pd.read_csv(X_file_name)
    y = pd.read_csv(y_file_name)
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.3)
    return X_train, y_train, X_test, y_test

def knn_baseline(X_train, y_train, num_neighbors):
    """
    The baseline knn model
    :param X_train: data samples
    :param y_train: labels of train samples
    :param num_neighbors: hyperparameter, number of neighbors for model
    :return:
    """

    model = nearest_neighbors(num_neighbors)
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    return y_train_pred




def drop_cols(df):
    """

    :param df: data frame of samples
    :return: data frame with dropped columns
    """
    cols_to_drop = ["KI67 protein", 'Diagnosis date', 'Surgery date1', 'Surgery date2', 'Surgery date3',
                    "surgery before or after-Activity date", "surgery before or after-Actual activity",
                    "id-hushed_internalpatientid", "er", "pr", "Ivi -Lymphovascular invasion", "Her2",
                    "Tumor depth"]
    for col in cols_to_drop:
        df = df.drop(col, axis=1)

    return df




def set_null(df):
    """

    :param df: data frame of samples
    :return: data frame with replaced null values
    """
    df = df.replace({np.nan: None})
    df['Histopatological degree'] = df['Histopatological degree'].str.replace('Null', "0")
    return df


def set_dummies(df):
    """

    :param df: data frame of samples
    :return: data frame with dummies of categorical variables
    """
    to_dummies = [' Form Name', ' Hospital', 'User Name', 'Basic stage', 'Histological diagnosis', "Surgery name1",
                  "Surgery name2", "Surgery name3", "M -metastases mark (TNM)", "N -lymph nodes mark (TNM)",
                  "T -Tumor mark (TNM)", "Lymphatic penetration", "Stage", "Side", "Histopatological degree",
                  'Margin Type']
    for dummy in to_dummies:
        df = pd.get_dummies(df, prefix=dummy, columns=[dummy])
    return df


def check_outliers(df):
    """

    :param df: data frame wos samples
    :return: data frame without outliers
    """
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
    """

    :param df: data frame of samples
    :return: data frame after preprocessing
    """
    df.columns = df.columns.str.replace('אבחנה-', '')
    df = drop_cols(df)
    df = check_outliers(df)
    df = set_null(df)
    df = set_dummies(df)

    return df


def create_responses(df):
    """

    :param df: data frame of samples
    :return: df with column response for every cancer type
    """
    responses = ['BON - Bones', 'LYM - Lymph nodes', 'HEP - Hepatic', 'PUL - Pulmonary', 'PLE - Pleura', 'SKI - Skin',
                 'OTH - Other', 'BRA - Brain', 'MAR - Bone Marrow', 'PER - Peritoneum', 'ADR - Adrenals']
    for response in responses:
        df[response] = np.zeros(df.shape[0], )
    return df


def prepare_labels(df):
    """

    :param df: data frame of samples
    :return: df with corrected format of labels
    """
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
    """

    :param X_train: data frame of samples- train
    :param X_test: data frame of samples- test
    :return: train and test data frames with same format and columns
    """
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
    """

    :param num_neighbors: hyperparameter of knn
    :return: initialized knn model with given num of neighbors
    """
    classifier = KNeighborsClassifier(n_neighbors=num_neighbors)
    return classifier

def multi_classification_pred_knn(X, y, num_neighbors):
    """

    :param X_train: data frame of samples
    :param y_train: responses os given samples
    :param num_neighbors: hyperparameter of knn
    :return: the prediction that the algorithm made on the data
    """
    predictions_knn = np.empty((11, y.shape[0]))
    for i in range(y.shape[1]):
        knn_pred = (knn_baseline(X, y[y.columns[i]], num_neighbors))
        predictions_knn[i] = knn_pred
    cols = y.columns
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
        micro += scoring(validation_res, y_dev_pred)[0]
        macro += scoring(validation_res, y_dev_pred)[1]
    return micro * (1 / cv), macro * (1 / cv)


def loss(y_pred, y_dev):
    """

    :param y_pred: prediction labels
    :param y_dev: true labels
    :return: micro, macro scores
    """
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
    
    
def find_best_k():
 """

    :return: k=5, as seen in plotted graph, k=5 gives highest micro&macro scores
    """
    micros = []
    macros = []
    neighbors_range = np.arange(4, 10)
        for index, num_neighbors in enumerate(neighbors_range):
        model = nearest_neighbors(num_neighbors)
        results = cross_validation(estimator=model,
                                   X=X_train,
                                   y=y_train,
                                   cv=4,
                                   scoring=loss, num_neighbors=num_neighbors)
        micro, macro = results
        micros[index]=micro
        macros[index]=macro
    # plot micro and macro scores for all Number of KN Neighbors
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
    return 5  # based on scores and graph
    
def predict(X_train, y_train, X_test, k):
 """

    :param X_train, y_train, X_test, k: train data and labels, test data to make predictions on
    and k, the hyperparameter of knn
    :return: prediction of test data, made with the model with hyperparameter =k
    """
    X_train, X_test = match_data(X_train, X_test)
    classifier = KNeighborsClassifier(n_neighbors=k)
    classifier.fit(X_train, y_train)
    labels_predictions = pd.DataFrame(classifier.predict(X_test), columns=y_train.columns)
    initial = np.empty(X_test.shape[0], dtype=list)
    for i in range(len(initial)):
            initial[i] = []
    for i in range(len(labels_predictions.columns)):
        for j in range(len(initial)):
            if (labels_predictions[labels_predictions.columns[i]]).iloc[j] == 1.0:
                initial[j].append(y_train.columns[i])
    y_pred = pd.DataFrame()
    y_pred["Location of distal metastases"] = initial
    y_pred["Location of distal metastases"] = y_pred["Location of distal metastases"].astype(str)
    return y_pred
    
    
if __name__ == '__main__':
   np.random.seed(0)
    gold_labels_file_name = sys.argv[1]
    X_train, y_train, X_dev, y_dev, X_test, y_test = split_data("train.feats.csv", "train.labels.0.csv")

    X_train['אבחנה-Location of distal metastases'] = y_train['אבחנה-Location of distal metastases']
    X_train = prepare_train(X_train)
    y_train = pd.DataFrame()
    y_train["Location of distal metastases"] = X_train["Location of distal metastases"]
    y_train = y_train.reset_index(drop=True)
    X_train = X_train.drop("Location of distal metastases", axis=1)
    X_train = X_train.reset_index(drop=True)

    X_dev['אבחנה-Location of distal metastases'] = y_dev['אבחנה-Location of distal metastases']
    X_dev = prepare_train(X_dev)
    y_dev = pd.DataFrame()
    y_dev["Location of distal metastases"] = X_dev["Location of distal metastases"]
    y_dev = y_dev.reset_index(drop=True)
    X_dev = X_dev.drop("Location of distal metastases", axis=1)
    X_dev = X_dev.reset_index(drop=True)

    y_train = prepare_labels(pd.DataFrame(y_train))
    y_dev = prepare_labels(y_dev)

    X_test = pd.read_csv("test.feats.csv")
    gold_labels = pd.read_csv(gold_labels_file_name)

    X_test['אבחנה-Location of distal metastases'] = gold_labels['אבחנה-Location of distal metastases']
    X_test = prepare_train(X_test)
    gold_labels = pd.DataFrame()
    gold_labels["Location of distal metastases"] = X_test["Location of distal metastases"]
    gold_labels = gold_labels.reset_index(drop=True)
    X_test = X_test.drop("Location of distal metastases", axis=1)
    X_test = X_test.reset_index(drop=True)
    gold_labels.to_csv("gold_labels_processed.csv", index=False)
    k = find_best_k()
    y_pred = predict(X_train, y_train, X_test, k)
    y_pred.to_csv("predictions.csv", index=False)


