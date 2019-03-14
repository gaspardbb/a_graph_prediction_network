import pandas as pd
import numpy as np
import os
import nltk
import pickle
from time import time
import networkx as nx
import sklearn.preprocessing as preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from utils import WordEmbedding, GraphStructure, nbr_common_authors, compute_intersection, list_of_publication_2, \
    compute_affinity_between_authors, make_feature_vector, put_authors_in_dict, stack_lists


class ComputeFeatures:

    def __init__(self, path_to_node_info='data/node_information.csv', path_to_training_set='data/training_set.txt',
                 path_to_test_set='data/testing_set.txt',
                 path_to_wv_model=None, load_graph_dict=True):
        """The ComputeFeatures class enables the user to compute multiple features.
        Consider using the static `import_from_file` static function so as not to have to compute the dataframe every single time.
        :param path_to_node_info: the `node_information.csv` file from the kaggle competition
        :param path_to_training_set: the `training_set`
        :param path_to_wv_model: the path to the wv model, if you do not want to train it from scratch."""

        self.handled_variables = ["publication_2",
                                  "adam_coeff",
                                  "overlapping_words_in_title",
                                  "number_of_common_authors",
                                  "difference_of_years",
                                  "affinity_between_authors",
                                  "identical_journal",
                                  "l2_distance",
                                  "cosine_distance_tfid",
                                  "l2_distance_between_titles",
                                  "common_neighbors",
                                  "clustering_coeff",
                                  "betweenness",
                                  "closeness",
                                  "degree",
                                  "eigenvector",
                                  "jaccard_coeff",
                                  "shortest_path",
                                  "pagerank",
                                  "community",
                                  "lp_within_inter_cluster",
                                  "lp_ra_index_soundarajan",
                                  "lp_cn_soundarajan",
                                  "lp_preferential_attachment",
                                  "lp_resource_allocation_index"]

        print("Loading node information...")
        self.node_information = pd.read_csv(path_to_node_info,
                                            names=['id', 'year', 'title', 'author', 'journal', 'abstract'])
        self.node_information = self.node_information.set_index('id')

        print("Loading train array...")
        self.train_array = np.loadtxt(path_to_training_set, dtype=int)
        self.nb_training_samples = self.train_array.shape[0]

        print("Loading test array...")
        self.test_array = np.loadtxt(path_to_test_set, dtype=int)
        self.nb_testing_samples = self.test_array.shape[0]

        # for tokenization
        print("Loading stemmer and stop words...")
        nltk.download('punkt')
        self.stemmer = nltk.stem.PorterStemmer()
        nltk.download('stopwords')
        self.stpwds = set(nltk.corpus.stopwords.words("english"))

        print("TfidVectorizer...")
        training_words = list(self.node_information['abstract'])
        vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0, stop_words="english")
        features_tfid = vectorizer.fit_transform(training_words)
        self.node_information["wv_tfid"] = pd.Series([x for x in features_tfid])

        print("Creating word embeddings...")
        self.wv = WordEmbedding(self.stemmer, self.stpwds)
        if path_to_wv_model is None:
            print("Training wv model with standard params...")
            self.wv.train_model(self.node_information)
        else:
            print("Loading wv model from %s" % path_to_wv_model)
            self.wv.load_model(path_to_wv_model)

        # Create publication column
        print("Creating publication column...")
        self.node_information['publication'] = self.node_information.apply(lambda row: [], axis=1)
        for t in self.train_array:
            if t[2] == 1:
                self.node_information.loc[t[0], 'publication'].append(t[1])
                self.node_information.loc[t[1], 'publication'].append(t[0])

        # Create publication_II column
        print("Creating publication II column...")
        self.node_information['publication_2'] = self.node_information.apply(
            lambda row: list_of_publication_2(row, self.node_information), axis=1)

        # Authors dict
        print("Creating authors dictionary...")
        authors_list = []
        self.node_information['author'].apply(lambda row: stack_lists(row, authors_list))
        authors_list = [auth for auth in authors_list if auth not in ['', '&', "(", ")"] and len(auth) > 2]
        self.authors_dict = dict((auth, []) for auth in np.unique(authors_list))
        del authors_list
        self.node_information['author'].apply(lambda row: put_authors_in_dict(row, self.authors_dict))
        for k in self.authors_dict.keys():
            while k in self.authors_dict[k]:
                self.authors_dict[k].remove(k)
            while '' in self.authors_dict[k]:
                self.authors_dict[k].remove('')

        # Feature vector for the abstract
        print("Making feature vectors for the abstract...")
        self.node_information['wv'] = self.node_information.apply(
            lambda row: make_feature_vector(row.loc['abstract'], self.wv.model), axis=1)

        print("Making feature vectors for the title...")
        self.node_information['title_wv'] = self.node_information.apply(
            lambda row: make_feature_vector(row.loc['title'], self.wv.model), axis=1)

        # Graph
        print("Making graph structure...")
        self.graph_structure = GraphStructure(self, load_graph_dict)

        self.abstract_feature_model = None

    @staticmethod
    def import_from_file(path_to_pickle):
        try:
            with open(path_to_pickle, "rb") as f:
                print("Loading ComputeFeatures object from file %s" % path_to_pickle)
                return pickle.load(f)
        except FileNotFoundError:
            print("File not found ! Check '.obj' extension.")
            return -1

    def save_in_file(self, path_to_file):
        with open(path_to_file, "wb") as f:
            pickle.dump(self, f)

    def compute_multiple_variables(self, iter_of_variables, train: bool, scale: bool, load=True, save=True):
        if iter_of_variables == "all":
            iter_of_variables = self.handled_variables
        else:
            for var in iter_of_variables:
                assert var in self.handled_variables, "Variable %s is not handled. Handled variables are : %s" % (
                    var, str(self.handled_variables))
        if train:
            result = np.zeros(shape=(self.nb_training_samples, len(iter_of_variables)))
        else:
            result = np.zeros(shape=(self.nb_testing_samples, len(iter_of_variables)))
        for i in range(len(iter_of_variables)):
            result[:, i] = np.transpose(self.compute_variable(iter_of_variables[i], train=train, load=load, save=save))
        if scale:
            result = preprocessing.scale(result)
        for i in range(len(iter_of_variables)):
            if np.all(result[:, i] == 0):
                print("WARNING: %i th column is void ! (%s)" % (i, iter_of_variables[i]))
        return result

    def compute_variable(self, variable_name, train: bool, load=True, path_to_file=None, save=True):

        assert variable_name in self.handled_variables, "Variable %s is not handled. Handled variables are : %s" % (
            variable_name, str(self.handled_variables))

        if load and train:
            if path_to_file is None and os.path.isfile("variables/%s.npy" % variable_name):
                print("Loading STANDARD %s file!" % variable_name)
                result = np.load("variables/%s.npy" % variable_name)
                return result[:self.nb_training_samples]
            elif path_to_file is not None and os.path.isfile(path_to_file):
                print("Loading CUSTOM %s file!" % variable_name)
                result = np.load(path_to_file)
                return result[:self.nb_training_samples]
            print("Did not find saved %s in `variables` folder." % variable_name)

        if load and not train:
            if path_to_file is None and os.path.isfile("variables/TEST_%s.npy" % variable_name):
                print("Loading STANDARD TEST_%s file!" % variable_name)
                result = np.load("variables/TEST_%s.npy" % variable_name)
                return result[:self.nb_training_samples]
            elif path_to_file is not None and os.path.isfile(path_to_file):
                print("Loading CUSTOM %s file!" % variable_name)
                result = np.load(path_to_file)
                return result[:self.nb_training_samples]
            print("Did not find saved TEST_%s in `variables` folder." % variable_name)

        print("Starting computation of %s..." % variable_name)
        t1 = time()
        gd = self.graph_structure.graph_dicts  # "graph_dictionaries
        if train:
            nb_of_samples = self.nb_training_samples
        else:
            nb_of_samples = self.nb_testing_samples
        result = np.zeros(shape=nb_of_samples)
        for i in range(nb_of_samples):
            if train:
                t = self.train_array[i]
            else:
                t = self.test_array[i]
            if variable_name == "publication_2":
                result[i] = np.log(len(set(self.node_information.loc[t[0], "publication_2"]) & set(
                    self.node_information.loc[t[1], "publication_2"])) + 1)
            elif variable_name == "adam_coeff":
                if train:
                    if t[2] == 1:
                        self.graph_structure.g.remove_edge(t[0], t[1])
                        result[i] = \
                            next(nx.algorithms.link_prediction.adamic_adar_index(self.graph_structure.g,
                                                                                 [(t[0], t[1])]))[2]
                        self.graph_structure.g.add_edge(t[0], t[1])
                    else:
                        result[i] = \
                            next(nx.algorithms.link_prediction.adamic_adar_index(self.graph_structure.g,
                                                                                 [(t[0], t[1])]))[2]
                else:
                    result[i] = \
                        next(nx.algorithms.link_prediction.adamic_adar_index(self.graph_structure.g, [(t[0], t[1])]))[2]
            elif variable_name == "overlapping_words_in_title":
                result[i] = compute_intersection(self.node_information.loc[t[0], "title"],
                                                 self.node_information.loc[t[1], "title"], self.stemmer,
                                                 self.stpwds)
            elif variable_name == "number_of_common_authors":
                result[i] = nbr_common_authors(self.node_information.loc[t[0], "author"],
                                               self.node_information.loc[t[1], "author"])

            elif variable_name == "difference_of_years":
                result[i] = abs(self.node_information.loc[t[0], 'year'] - self.node_information.loc[t[1], 'year'])

            elif variable_name == "affinity_between_authors":
                result[i] = compute_affinity_between_authors(self.node_information.loc[t[0], 'author'],
                                                             self.node_information.loc[t[1], 'author'],
                                                             self.authors_dict)
            elif variable_name == "identical_journal":
                result[i] = np.int(
                    self.node_information.loc[t[0], 'journal'] == self.node_information.loc[t[1], 'journal'])

            elif variable_name == "l2_distance":
                result[i] = np.linalg.norm(
                    self.node_information.loc[t[0], 'wv'] - self.node_information.loc[t[1], 'wv'])

            elif variable_name == "cosine_distance_tfid":
                v1 = self.node_information.loc[t[0], "wv_tfid"]
                v2 = self.node_information.loc[t[1], "wv_tfid"]
                try:
                    b1 = np.isnan(v1)
                except TypeError:
                    b1 = False
                try:
                    b2 = np.isnan(v2)
                except TypeError:
                    b2 = False
                if not b1 and not b2:
                    result[i] = cosine_similarity(v1, v2)
                else:
                    result[i] = 0

            elif variable_name == "l2_distance_between_titles":
                dst = np.linalg.norm(
                    self.node_information.loc[t[0], 'title_wv'] - self.node_information.loc[t[1], 'title_wv']
                )
                if np.isnan(dst):
                    result[i] = 0
                else:
                    result[i] = dst

            # elif variable_name == "cosine_distance_between_titles":
            #     result[i] = cosine_distances(
            #         np.nan_to_num(self.node_information.loc[t[0], 'title_wv']).reshape(-1, 1) - (self.node_information.loc[t[1], 'title_wv']).reshape(-1, 1)
            #     )[0][0]

            elif variable_name == "common_neighbors":
                result[i] = len(sorted(nx.common_neighbors(self.graph_structure.g, t[0], t[1])))

            elif variable_name == "clustering_coeff":
                result[i] = gd["clustering_coeff"][t[0]] * gd["clustering_coeff"][t[1]]

            elif variable_name == "betweenness":
                result[i] = gd["betweenness"][t[0]] * gd["betweenness"][t[1]]

            elif variable_name == "closeness":
                result[i] = gd["closeness"][t[0]] * gd["closeness"][t[1]]

            elif variable_name == "degree":
                result[i] = gd["degree"][t[0]] * gd["degree"][t[1]]

            elif variable_name == "eigenvector":
                result[i] = gd["eigenvector"][t[0]] * gd["eigenvector"][t[1]]

            elif variable_name == "jaccard_coeff":
                if train:
                    if t[2] == 1:
                        self.graph_structure.g.remove_edge(t[0], t[1])
                        result[i] = next(nx.jaccard_coefficient(self.graph_structure.g, [(t[0], t[1])]))[2]
                        self.graph_structure.g.add_edge(t[0], t[1])
                    else:
                        result[i] = next(nx.jaccard_coefficient(self.graph_structure.g, [(t[0], t[1])]))[2]
                else:
                    result[i] = next(nx.jaccard_coefficient(self.graph_structure.g, [(t[0], t[1])]))[2]
            elif variable_name == "shortest_path":
                if train:
                    if t[2] == 1:
                        assert self.graph_structure.g.has_edge(t[0], t[
                            1]), "There's a problem with the structure of the graph for id %i and %i" % (t[0], t[1])
                        self.graph_structure.g.remove_edge(t[0], t[1])
                        try:
                            result[i] = 1 / nx.algorithms.shortest_paths.generic.shortest_path_length(
                                self.graph_structure.g, t[0], t[1])
                        except nx.NetworkXNoPath:
                            result[i] = 0
                        self.graph_structure.g.add_edge(t[0], t[1])
                    else:
                        try:
                            result[i] = 1 / nx.algorithms.shortest_paths.generic.shortest_path_length(
                                self.graph_structure.g, t[0], t[1])
                        except nx.NetworkXNoPath:
                            result[i] = 0
                else:
                    try:
                        result[i] = 1 / nx.algorithms.shortest_paths.generic.shortest_path_length(
                            self.graph_structure.g, t[0], t[1])
                    except nx.NetworkXNoPath:
                        result[i] = 0

            elif variable_name == "pagerank":
                result[i] = gd["pagerank"][t[0]] * gd["pagerank"][t[1]]

            elif variable_name == "community":
                if self.graph_structure.partition[t[0]] == self.graph_structure.partition[t[1]]:
                    result[i] = 1
                else:
                    result[i] = 0

            elif variable_name == "lp_resource_allocation_index":
                if train:
                    if t[2] == 1:
                        self.graph_structure.g.remove_edge(t[0], t[1])
                        result[i] = sorted(nx.resource_allocation_index(self.graph_structure.g, [(t[0], t[1])]))[0][2]
                        self.graph_structure.g.add_edge(t[0], t[1])
                    else:
                        result[i] = sorted(nx.resource_allocation_index(self.graph_structure.g, [(t[0], t[1])]))[0][2]
                else:
                    result[i] = sorted(nx.resource_allocation_index(self.graph_structure.g, [(t[0], t[1])]))[0][2]

            elif variable_name == "lp_preferential_attachment":
                if train:
                    if t[2] == 1:
                        self.graph_structure.g.remove_edge(t[0], t[1])
                        result[i] = sorted(nx.preferential_attachment(self.graph_structure.g, [(t[0], t[1])]))[0][2]
                        self.graph_structure.g.add_edge(t[0], t[1])
                    else:
                        result[i] = sorted(nx.preferential_attachment(self.graph_structure.g, [(t[0], t[1])]))[0][2]
                else:
                    result[i] = sorted(nx.preferential_attachment(self.graph_structure.g, [(t[0], t[1])]))[0][2]
            elif variable_name == "lp_cn_soundarajan":
                if train:
                    if t[2] == 1:
                        self.graph_structure.g.remove_edge(t[0], t[1])
                        result[i] = sorted(nx.cn_soundarajan_hopcroft(self.graph_structure.g, [(t[0], t[1])]))[0][2]
                        self.graph_structure.g.add_edge(t[0], t[1])
                    else:
                        result[i] = sorted(nx.cn_soundarajan_hopcroft(self.graph_structure.g, [(t[0], t[1])]))[0][2]
                else:
                    result[i] = sorted(nx.cn_soundarajan_hopcroft(self.graph_structure.g, [(t[0], t[1])]))[0][2]
            elif variable_name == "lp_ra_index_soundarajan":
                if train:
                    if t[2] == 1:
                        self.graph_structure.g.remove_edge(t[0], t[1])
                        result[i] = sorted(nx.ra_index_soundarajan_hopcroft(self.graph_structure.g, [(t[0], t[1])]))[0][
                            2]
                        self.graph_structure.g.add_edge(t[0], t[1])
                    else:
                        result[i] = sorted(nx.ra_index_soundarajan_hopcroft(self.graph_structure.g, [(t[0], t[1])]))[0][
                            2]
                else:
                    result[i] = sorted(nx.ra_index_soundarajan_hopcroft(self.graph_structure.g, [(t[0], t[1])]))[0][2]

            elif variable_name == "lp_within_inter_cluster":

                if train:
                    if t[2] == 1:
                        self.graph_structure.g.remove_edge(t[0], t[1])
                        result[i] = sorted(nx.within_inter_cluster(self.graph_structure.g, [(t[0], t[1])]))[0][2]
                        self.graph_structure.g.add_edge(t[0], t[1])
                    else:
                        result[i] = sorted(nx.within_inter_cluster(self.graph_structure.g, [(t[0], t[1])]))[0][2]
                else:
                    result[i] = sorted(nx.within_inter_cluster(self.graph_structure.g, [(t[0], t[1])]))[0][2]

        print("Did %s column in %5.1fs" % (variable_name, time() - t1))
        if save and train:
            print("Saved variable %s in `variables` directory." % variable_name)
            np.save("variables/" + variable_name, result)
        if save and not train:
            np.save("variables/TEST_" + variable_name, result)
            print("Saved variable TEST_%s in `variables` directory." % variable_name)
        if np.isnan(result).shape[0] >= 1:
            print("Careful, you have nan values !")
            result[np.isnan(result)] = 0
        return result

# X_tot = c.compute_multiple_variables("all", train=True, scale=True, save=False)
# X_submission = c.compute_multiple_variables("all", train=False, scale=False, save=False)
#
# X_train, X_test = np.split(X_tot, [np.int(X_tot.shape[0]*0.9)])
# y = c.train_array[:, 2]
# y_train = y[:np.int(y.shape[0]*0.9)]
# y_test = y[np.int(y.shape[0]*0.9):]

# print(sorted(zip(map(lambda x: round(x, 4), model.feature_importances_), c.handled_variables), reverse=True))
