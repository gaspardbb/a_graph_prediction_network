# Manual

Here's a quick manual to reproduce our result. Run the next commands from the `main.py` file. 

## Mandatory files and directories
 
* 4 folders :
    * A `data` dir containing  `node_information.csv`, `testing_set.txt`, `training_set.txt`
    * A `graph_dict` dir containing the files 5 `npz` arrays, if you want to speed up the process and not having to recompute them.
    * A `variables` dir containing `npz` files if you want to speed up the process, even though it's easier to compute than the graph dictionaries.
    * A `wv_model` dir, where you can store your word embedding models.
* 3 `.py` file :
    * `compute_features.py` contains the `ComputeFeatures` class and many useful functions
    * `my_nets.py` contain some nets used for prediction
    * `utils.py` contain the `WordEmbedding` and `GraphStructure` classes.
    
## Get train and test matrices

1. Build your `ComputeFeatures` object : 
    * `c = ComputeFeatures()` to do it from scratch (~20s with dict loaded). **Warning** : the `load_graph_dict` is set to `True` by default, and will load the dictionaries in the `graph_dict` directory. If not, please note that the computation is **very** expensive.
    * `c = ComputeFeatures.import_from_file(<path>)` to load it directly
2. Get your train matrix with : `X_train = c.compute_multiple_variables("all", train=True, scale=<bool>)`.
You can specify which variables you want with a list of handled variables instead of `"all"`, e.g. `["l2_distance", "betweenness"]`
2. Get your test matrix with : `X_test = c.compute_multiple_variables("all", train=False)`

## Example 

The following code load the train and test matrices and run and various classifiers:
```python
from sklearn.model_selection import train_test_split
from compute_features import ComputeFeatures
from train import bagging, neural_net, naive_bayes, svc, sgd
from sklearn.preprocessing import scale

c = ComputeFeatures()

X_tot = c.compute_multiple_variables("all", train=True, scale=False)
X_test = c.compute_multiple_variables("all", train=False, scale=True)
X_train, X_validation, y_train, y_validation = train_test_split(X_tot, c.train_array[:, 2])
X_train = scale(X_train)
X_validation = scale(X_validation)

models = []
for f in naive_bayes, bagging, neural_net, svc, sgd:
    models.append(f(X_train, y_train, X_validation, y_validation, False))
```

If you want to train your model only using specific variables, you just need to do: 
```python
my_var = ["l2_distance", "common_neighbors", "degree"]
X_tot =  c.compute_multiple_variables(my_var, train=True, scale=False)
```
Which can turn to be *really* useful for making tests.

## Other useful functions

In the `utils.py` file, you can find the `plot_distribution_of_prediction`, useful to have an idea of how well do your features 
separate the data. E.g :

```python
plot_distribution_of_prediction(5, 3, c.handled_variables[:15], X_validation, y_validation):
```

Will plot the distribution of every variables! Nice, right? 

From the `train.py` file, you can run the `f1_of_variables` or the `decrease_of_accuracy functions`.

This will print the F1 score obtained by using the bagging classifier on every variable alone, to see how well it fares on the training
or validation set (require the computation of a proper validation set).
```python
result = f1_of_variables(c, c.handled_variables)
result_bag = dict()
for k in result.keys():
    result_bag[k] = dict()
    result_bag[k]["80"] = np.mean(result[k]["80"]["bag"])
    result_bag[k]["20"] = np.mean(result[k]["20"]["bag"])
result_bag = pd.DataFrame(result_bag).transpose()
```

This will print the decrease of accuracy by putting out each feature (using RF properties). 
```python
decrease_of_acc(X_train, y_train, <my_var>)
```