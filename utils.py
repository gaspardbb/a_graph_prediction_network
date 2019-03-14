from collections import defaultdict
import community

from gensim.models import word2vec
import networkx as nx
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import ShuffleSplit
from sklearn.tree import DecisionTreeClassifier

import matplotlib.pyplot as plt
import seaborn as sns


class WordEmbedding:

    def __init__(self, stemmer, stpwrds):
        self.model = word2vec.Word2Vec()
        self.stop_words = stemmer
        self.stop_words = stpwrds
        self.initialized = False

    def load_model(self, path_to_model):
        self.model = word2vec.Word2Vec.load(path_to_model)

    def train_model(self, node_information, num_features=300, min_word_count=5, num_workers=4, context=20,
                    downsampling=1e-4):
        training_words = list(node_information['abstract'].apply(
            lambda row: [word for word in row.split() if word not in self.stop_words]))
        self.model = word2vec.Word2Vec(training_words, workers=num_workers, size=num_features, min_count=min_word_count,
                                       window=context, sample=downsampling)

    def save_model(self, model_name):
        self.model.save("wv_model/%s" % model_name)


class GraphStructure:
    save_dir = "graph_dict/"

    def __init__(self, c, load_from_file: bool):
        self.g = nx.Graph()
        edges = [(element[0], element[1]) for element in c.train_array if element[2] == 1]
        nodes = c.node_information.index.values
        self.handled_dicts = ["betweenness", "closeness", "clustering_coeff", "degree", "eigenvector", "pagerank"]
        self.g.add_edges_from(edges)
        self.g.add_nodes_from(nodes)
        self.graph_dicts = dict()
        for d in self.handled_dicts:
            self.graph_dicts[d] = dict()
        if load_from_file:
            for d in self.handled_dicts:
                try:
                    self.graph_dicts[d] = np.load("%sdict_%s.npy" % (self.save_dir, d)).item()
                    print("Loaded %s dictionary!" % d)
                except FileNotFoundError:
                    print("Could not load %s." % d)
            try:
                self.partition = np.load("%spartition.npy" % self.save_dir).item()
                print("Loaded partition file.")
            except FileNotFoundError:
                print("Could not load partition file.")
        else:
            print("WARN: You're attempting to compute the graph dictionnaries yourself. Process may be VERY long.")
            print("Doing dict_clustering_coeff...")
            self.graph_dicts["clustering_coeff"] = nx.algorithms.cluster.clustering(self.g)
            np.save("%sdict_clustering_coeff.npy" % self.save_dir, self.graph_dicts["clustering_coeff"])
            print("Doing dict_betweenness...")
            self.graph_dicts["betweenness"] = nx.algorithms.centrality.betweenness_centrality(self.g, k=10000)
            np.save("%sdict_betweenness.npy" % self.save_dir, self.graph_dicts["betweenness"])
            print("Doing dict_closeness...")
            self.graph_dicts["closeness"] = nx.algorithms.centrality.closeness_centrality(self.g)
            np.save("%sdict_closeness.npy" % self.save_dir, self.graph_dicts["closeness"])
            print("Doing dict_degree...")
            self.graph_dicts["degree"] = nx.algorithms.centrality.degree_centrality(self.g)
            np.save("%sdict_degree.npy" % self.save_dir, self.graph_dicts["degree"])
            print("Doing dict_eigenvector...")
            self.graph_dicts["eigenvector"] = nx.algorithms.centrality.eigenvector_centrality(self.g)
            np.save("%sdict_eigenvector.npy" % self.save_dir, self.graph_dicts["eigenvector"])
            print("Doing dict_pagerank...")
            self.graph_dicts["pagerank"] = nx.algorithms.link_analysis.pagerank_alg.pagerank(self.g, alpha=0.85,
                                                                                             max_iter=200)
            np.save("%sdict_pagerank.npy" % self.save_dir, self.graph_dicts["pagerank"])
            print("Doing partition...")
            self.partition = community.best_partition(self.g)
            np.save("%spartition.npy" % self.save_dir, self.partition)


def plot_distribution_of_prediction(nrows, ncols, list_of_variables, x_test, y_test=None):
    """
Give distribution plot to test a list of features, in a plt.subplots grid. Ensure there's enough axes for each variable,
which should be given as a list of string.
    :param nrows: number of rows in the plt.subplots
    :param ncols: number of cols in the plt.subplots
    :param list_of_variables: list of variable to plot the distribution of. Mind the order!
    See ComputeFeatures.handled_variables
    :param x_test: the array used to predict
    :param y_test: if given, enables to look at the distribution of connected/disjoint
    """

    fig, axes = plt.subplots(nrows, ncols)
    for i in range(nrows * ncols):
        ax = axes[i // ncols, i % ncols]
        if y_test is None:
            sns.distplot(x_test[:, i], ax=ax)
        else:
            var = x_test[:, i]
            connected = var[np.where(y_test == 1)]
            disjoint = var[np.where(y_test == 0)]
            sns.distplot(connected, label="connected", color="b", ax=ax)
            sns.distplot(disjoint, label="disjoint", color="r", ax=ax)
        ax.set_title(list_of_variables[i])
    fig.suptitle("Blue: connected. Red: disjoint.")


def get_features_importance(X, y, variables):
    """
Run random forest on data, multiple times. Look for loss of f1_score for each variable.
Return sorted list of most important variables.
    :param X: train array
    :param y: target array
    :param variables: list of variables to look at, in right order. See ComputeFeatures.handled_variables.
    :return: dictionary with list of f1_score for each variable.
    """

    sp = ShuffleSplit(n_splits=5, test_size=.2)
    sp.get_n_splits(X)

    scores = defaultdict(list)

    names = variables

    for train_idx, test_idx in sp.split(X):
        x_train, x_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        model = DecisionTreeClassifier()
        model.fit(x_train, y_train)
        acc = f1_score(y_test, model.predict(x_test))
        for i in range(X.shape[1]):
            X_t = x_test.copy()
            np.random.shuffle(X_t[:, i])
            shuff_acc = f1_score(y_test, model.predict(X_t))
            scores[names[i]].append((acc - shuff_acc) / acc)
    print("Features sorted by their score:")
    print(sorted([(np.round(np.mean(score), 4), feat) for feat, score in scores.items()], reverse=True))
    return scores


def nbr_common_authors(string1, string2):
    if string1 is np.nan or string2 is np.nan or type(string1) is not str or type(string2) is not str:
        return 0
    else:
        return len(set(string1.lower().split(', ')) & set(string2.lower().split(', ')))


def process_string(text, stemmer, stpwds):
    """Process string by removing stopwords and keeping only the roots of words.
    """
    return [stemmer.stem(token) for token in text.split() if token not in stpwds]


def compute_intersection(string1, string2, stemmer, stpwds):
    """Compute the intersection between two strings having preprocessed them before."""
    return len(set(process_string(string1, stemmer, stpwds)) & set(process_string(string2, stemmer, stpwds)))


def list_of_publication_2(row_of_df, df):
    """From a list of publication p1, ..., pN, returns a list of every publication quoted by p1, ..., pN
    params:
    row_of_df: a row of the dataframe. Should have a definite id and a 'publication' column containing list p1, ..., pN
    df : the dataframe indexed by ID, with a column "publication"
    """
    res = []
    for list_of_publi in row_of_df['publication']:
        res.extend(df.loc[list_of_publi, 'publication'])
    # We take out the publications which are self in the row.
    res = [publi for publi in res if publi != row_of_df.name]
    return res


def compute_affinity_between_authors(auth_str_1, auth_str_2, auth_dict):
    if auth_str_1 is not np.nan and auth_str_2 is not np.nan and type(auth_str_1) is str and type(auth_str_2) is str:
        auth_list_1 = auth_str_1.split(', ')
        auth_list_2 = auth_str_2.split(', ')
        all_auth_1 = []
        all_auth_2 = []
        for auth in auth_list_1:
            try:
                all_auth_1.extend(auth_dict[auth])
            except KeyError:
                pass
        for auth in auth_list_2:
            try:
                all_auth_2.extend(auth_dict[auth])
            except KeyError:
                pass
        # There is a huge variability.
        # We normalize by the sum of the authors
        return len(intersection_between_list(all_auth_1, all_auth_2)) / (len(auth_list_1) + len(auth_list_2))
    else:
        return 0


def make_feature_vector(abstract, model):
    num_features = model.trainables.layer1_size
    result = np.zeros(num_features)
    words = abstract.split()
    not_in_vocab = 0
    for word in words:
        if word in model.wv.vocab:
            result += model.wv[word]
        else:
            not_in_vocab += 1
    if len(words) - not_in_vocab != 0:
        result /= (len(words) - not_in_vocab)
    else:
        result = 0
    return result


def put_authors_in_dict(auth_string: str, auth_dict: dict):
    """Add authors in dictionary.
    CAREFUL : author is in its own list !"""
    if auth_string is not np.nan and type(auth_string) is str:
        auth_list = auth_string.split(', ')
        for auth in auth_list:
            try:
                auth_dict[auth].extend(auth_list)
            except KeyError:
                # the author name was removed during preprocessing
                pass


def intersection_between_list(l1, l2):
    """Intersection between two lists with multiplicity : duplicates are not removed."""
    inter = set(l1) & set(l2)
    res = []
    for x in inter:
        res.extend([x] * (np.min((l1.count(x), l2.count(x)))))
    return res


def coeff_adam(source, target, df):
    coeff = 0.
    for ID in (set(source['publication']) & set(target['publication'])):
        length = len(set(df.loc[int(ID)]['publication'])) + 1
        coeff += 1 / np.log(length)
    return coeff


def stack_lists(string_authors: str, my_list: list):
    if string_authors is not np.nan and type(string_authors) is str:
        my_list.extend(string_authors.split(', '))
