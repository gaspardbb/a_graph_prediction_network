from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import f1_score
from collections import defaultdict
from sklearn.tree import DecisionTreeClassifier


def metrics_print(y_validation, y_pred):
    print("F1_score :  %4.3f \n"
          "Accuracy :  %4.3f \n"
          "Precision : %4.3f \n"
          "Recall :    %4.3f \n" % (
              f1_score(y_validation, y_pred), accuracy_score(y_validation, y_pred),
              precision_score(y_validation, y_pred),
              recall_score(y_validation, y_pred)))


def bagging(X_train, y_train, X_validation, y_validation, return_f1: bool):
    print("BAGGING")
    model = BaggingClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_validation)
    if return_f1:
        return f1_score(y_validation, y_pred)
    metrics_print(y_validation, y_pred)
    return model


def sgd(X_train, y_train, X_validation, y_validation, return_f1: bool):
    print("SGD")
    model = SGDClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_validation)
    if return_f1:
        return f1_score(y_validation, y_pred)
    metrics_print(y_validation, y_pred)
    return model


def neural_net(X_train, y_train, X_validation, y_validation, return_f1: bool):
    print("NEURAL NET")
    early_stop = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=2, mode='auto')

    model = Sequential()
    model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='Adam',
                  metrics=['accuracy']
                  )

    model.fit(X_train, y_train,
              verbose=0,
              epochs=200,
              batch_size=100,
              callbacks=[early_stop],
              validation_data=(X_validation, y_validation))

    y_pred = np.round(model.predict(X_validation))
    if return_f1:
        return f1_score(y_validation, y_pred)
    metrics_print(y_validation, y_pred)
    return model


def svc(X_train, y_train, X_validation, y_validation, return_f1: bool):
    print("SVM")

    model = SVC()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_validation)
    if return_f1:
        return f1_score(y_validation, y_pred)
    metrics_print(y_validation, y_pred)
    return model


def linear_svc(X_train, y_train, X_validation, y_validation, return_f1: bool):
    print("Linear SVC")

    model = LinearSVC()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_validation)
    if return_f1:
        return f1_score(y_validation, y_pred)
    metrics_print(y_validation, y_pred)
    return model


def naive_bayes(X_train, y_train, X_validation, y_validation, return_f1: bool):
    print("GaussianNB")

    model = GaussianNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_validation)
    if return_f1:
        return f1_score(y_validation, y_pred)
    metrics_print(y_validation, y_pred)
    return model


def f1_of_variables(c, variables):
    y_train80 = c.train_array[:, 2]
    y_validation80 = c.test_array[:, 2]
    result = dict()
    for var in variables:
        result[var] = dict()
        result[var]["80"] = dict()
        result[var]["20"] = dict()

        X_train80 = c.compute_multiple_variables([var], train=True, scale=True)
        X_validation80 = c.compute_multiple_variables([var], train=False, scale=True)
        X_tot = X_validation80
        X_train20, X_validation20, y_train20, y_validation20 = train_test_split(X_tot, y_validation80, test_size=0.10)

        # TEST ON 80%
        result[var]["80"]["bayes"] = naive_bayes(X_train80, y_train80, X_validation80, y_validation80, return_f1=True)
        res_of_bag = []
        for i in range(5):
            b = bagging(X_train80, y_train80, X_validation80, y_validation80, return_f1=True)
            res_of_bag.append(b)
        result[var]["80"]["bag"] = res_of_bag
        # result[var]["80"]["svc"] = linear_svc(X_train80, y_train80, X_validation80, y_validation80, return_f1=True)

        # TEST ON 20%
        result[var]["20"]["bayes"] = naive_bayes(X_train20, y_train20, X_validation20, y_validation20, return_f1=True)
        res_of_bag = []
        for i in range(5):
            b = bagging(X_train20, y_train20, X_validation20, y_validation20, return_f1=True)
            res_of_bag.append(b)
        result[var]["20"]["bag"] = res_of_bag
        # result[var]["20"]["svc"] = linear_svc(X_train20, y_train20, X_validation20, y_validation20, return_f1=True)
    return result


def decrease_of_acc(X, Y, names):
    names = c.handled_variables

    rf = DecisionTreeClassifier()
    shuffle_split = ShuffleSplit(n_splits=10)

    scores = defaultdict(list)

    # crossvalidate the scores on a number of different random splits of the data
    for train_idx, test_idx in shuffle_split.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]
        r = rf.fit(X_train, Y_train)
        acc = f1_score(Y_test, rf.predict(X_test))
        for i in range(X.shape[1]):
            X_t = X_test.copy()
            np.random.shuffle(X_t[:, i])
            shuff_acc = f1_score(Y_test, rf.predict(X_t))
            scores[names[i]].append((acc - shuff_acc) / acc)
    print("Features sorted by their score:")
    print(sorted([(round(np.mean(score), 4), feat) for feat, score in scores.items()], reverse=True))
