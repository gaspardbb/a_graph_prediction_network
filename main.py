from sklearn.model_selection import train_test_split

from compute_features import ComputeFeatures
from train import bagging, neural_net, naive_bayes, svc, sgd

c = ComputeFeatures()

X_tot = c.compute_multiple_variables("all", train=True, scale=False)
X_test = c.compute_multiple_variables("all", train=False, scale=True)
X_train, X_validation, y_train, y_validation = train_test_split(X_tot, c.train_array[:, 2])

# Compute models and print F1 score
models = []
for f in naive_bayes, bagging, neural_net, svc, sgd:
    models.append(f(X_train, y_train, X_validation, y_validation, False))

# Compute change of accuracy - for validation set
# result = f1_of_variables(c, c.handled_variables)
# result_sgd = dict()
# for k in result.keys():
#     result_sgd[k] = dict()
#     result_sgd[k]["80"] = np.mean(result[k]["80"]["bag"])
#     result_sgd[k]["20"] = np.mean(result[k]["20"]["bag"])
# result_sgd = pd.DataFrame(result_sgd).transpose()
