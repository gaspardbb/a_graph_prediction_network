from keras.models import Sequential
from keras.layers import Conv2D, Dropout, Dense, Reshape
from keras.callbacks import EarlyStopping
import numpy as np
from compute_features import ComputeFeatures


def create_net(wv_size, n):
    model = Sequential()

    model.add(Conv2D(filters=2, kernel_size=(1, n), input_shape=(wv_size, n, 2)))
    model.add(Conv2D(filters=2, kernel_size=(1, 1)))
    model.add(Conv2D(filters=1, kernel_size=(1, 1)))
    model.add(Reshape((wv_size,)))
    model.add(Dense(units=300, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=150, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=100, activation='relu'))
    model.add(Dense(units=1, activation='sigmoid'))
    return model


def get_least_frequent_words_embeddings(n, w2v_model, abstract):
    """Return the n least frequent word embedding of abstract with word embedding `model` representation.
    :param abstract: string of words
    :param n: number of words to keep, starting with the least frequent
    :param w2v_model: word2vec model
    """
    words_list = []
    count_list = []
    for word in abstract.split(" "):
        try:
            count_list.append(w2v_model.wv.vocab[word].count)
            words_list.append(word)
        except KeyError:
            pass
    sorted_indices = np.argsort(count_list)
    result = np.zeros(shape=(w2v_model.vector_size, n))
    for i in range(n):
        try:
            result[:, i] = w2v_model.wv[words_list[sorted_indices[i]]]
        except IndexError:
            pass
    return result


def compute_X_train(n: int, indices, c: ComputeFeatures):
    """
From a list of indices, use the training_set corresponding to these indices to compute a matrix
of size (len(indices) * w2v.size_of_vector * n * 2).
    :param indices: set of indices to look at in train_set.
    :param n: number of words to take in abstract
    :param c: a ComputeFeatures object, already initialized
    :return: (len(indices) * w2v.size_of_vector * n * 2) matrix
    """
    node_information = c.node_information
    model = c.wv.model
    train_set = c.train_array
    X_train = np.zeros(shape=(indices.shape[0], model.vector_size, n, 2))
    for i in range(indices.shape[0]):
        X_train[i, :, :, 0] = get_least_frequent_words_embeddings(n, model,
                                                                  node_information.loc[
                                                                      train_set[indices[i]][0], "abstract"])
        X_train[i, :, :, 1] = get_least_frequent_words_embeddings(n, model,
                                                                  node_information.loc[
                                                                      train_set[indices[i]][1], "abstract"])
    return X_train


def datagenerator_x_train(n: int, c: ComputeFeatures, batch_size=500, max=600000):
    """
Datagenerator for net.
    :param n: number of words to keep in an abstract
    :param c: a ComputeFeatures object, already initialized
    :param batch_size: number of abstracts to look at at each iteration
    :param max: number of samples used
    """
    while True:
        indices = np.random.choice(np.arange(0, max, 1), batch_size)
        yield compute_X_train(n, indices, c), c.train_array[indices, 2]


def train_my_net(c: ComputeFeatures, n: int, steps_per_epoch=100, epochs=50):
    early_stop = EarlyStopping(monitor='acc', min_delta=0, patience=3, verbose=2, mode='auto')

    model = create_net(c.wv.model.vector_size, n)

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy']
                  )
    # model.fit(X_train, y_train, batch_size=1000, validation_split=0.1
    #           epochs=100)

    model.fit_generator(datagenerator_x_train(n, c),
                        steps_per_epoch=steps_per_epoch, epochs=epochs, callbacks=[early_stop])

    c.abstract_feature_model = model

    return model


def get_prediction(c: ComputeFeatures, id1, id2, n: int):
    model = c.abstract_feature_model
    X = np.zeros(shape=(c.wv.model.vector_size, n, 2))
    X[:, :, 0] = get_least_frequent_words_embeddings(n, c.wv.model, c.node_information.loc[id1, "abstract"])
    X[:, :, 1] = get_least_frequent_words_embeddings(n, c.wv.model, c.node_information.loc[id2, "abstract"])
    return model.predict(X)
