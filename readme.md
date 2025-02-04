# Multi-Event Survival Analysis-Informed Clustering
This repository hosts the code used in the publication "Multi-Event Survival Analysis-Informed Clustering with MESA-LVQ: Functional System Worsening in Multiple Sclerosis".

# Setup
- Environment setup: in the paper, Python 3.11 was used with the following packages (for more details on version specifics, see `requirements.txt`):
    ```bash
    pip install numpy pandas matplotlib seaborn torch scikit-learn scikit-survival genieclust
    ```
- Clone this repository, including SurvivalLVQ which is included as a submodule:
    ```bash
    git clone --recurse-submodules git@github.com:robbedhondt/MESALVQ.git
    ```
    Note: if you cloned without `--recurse-submodules`, you can retrieve the submodule with `git submodule update --init` or, alternatively, by cloning the SurvivalLVQ repo into the mesaLVQ folder `cd mesaLVQ; git clone git@gitlab.kuleuven.be:u0125808/survivallvq.git`.

# Using the code in your application
First, load in your data. Make sure it is in the following format: 
- `X` is a pandas DataFrame or a 2D numpy array (can contain missing values, but is advised to be skew transformed and z-standardized -- see SkewTransformer in the original SurvivalLVQ repository).
- `y` is a structured arrays, where the first element is the event indicator and the second element the observed time (time-to-event or time-to-censoring). Similar as in `scikit-surv`, but now each element is a 2D array. If you have a 2D array of event indicators `cens` and a 2D array of observed times `time` (where rows = observations, columns = different events), you can transform these into the 2D structured array with a utility function:
    ```python
    from mesaLVQ.data import multi_surv_from_arrays
    y = multi_surv_from_arrays(cens, time)
    ```

Fitting MESA-LVQ then becomes as simple as (assuming this repository clone is part of the Python path, which can be achieved for example through `import sys; sys.path.append("path/to/this/clone/")`)
```python
from mesaLVQ.data import split_train_test
from mesaLVQ.model import MultiEventSurvivalLVQ
X_train, X_test, y_train, y_test = split_train_test(X, y)
mesalvq = MultiEventSurvivalLVQ(n_prototypes=4)
mesalvq.fit(X_train, y_train)
```

You can also run the single-event variant through the provided wrapper function:
```python
from mesaLVQ.model import SurvivalLVQ, LocalMultiEventSurvivalLVQ
seslvq = [SurvivalLVQ(n_prototypes=4) for _ in range(y_train.shape[1])]
seslvq = LocalMultiEventSurvivalLVQ(seslvq)
seslvq.fit(X_train, y_train)
```

Some internal cluster validity metrics for the two variants on the train set can then be computed as follows:
```python
from mesaLVQ.plot import compute_cluster_validity
y_pred_global = mesalvq.predict(X_train, closest=True)
y_pred_local  = seslvq.predict(X_train, closest=True)
compute_cluster_validity(X_train, y_pred_global, y_pred_local)
```

Predictive performance can be evaluated using a helper function in `main.py`:
```python
from main import score_single_model
times = np.linspace(1.1 * min(y["event"]), 0.9 * max(y["event"]), 100) # to calculate IBS / IAUROC
score_single_model(mesalvq, X_train, X_test, y_train, y_test, times=times) 
score_single_model(seslvq , X_train, X_test, y_train, y_test, times=times) 
```

Some graphs of interest you can make then:
```python
from mesaLVQ.plot import *
names = X_train.columns
event_names = None # list of event names

# Feature relevances: diagonal of the relevance matrix
plot_feature_relevances(mesalvq, names=names)
plot_feature_relevances_local(seslvq, names=names, event_names=event_names)

# Kaplan-Meier disambiguation per event on the train set
y_pred = mesalvq.predict(X_train, closest=True)
plot_kaplan_meier_per_cluster(X_train, y_train, y_pred, subplots_params=dict(nrows=2, ncols=4, figsize=(9,4)))
y_pred = seslvq.predict(X_train, closest=True)
plot_kaplan_meier_per_cluster(X_train, y_train, y_pred, subplots_params=dict(nrows=2, ncols=4, figsize=(9,4)))

# Prototype loadings
loadings = mesalvq.w.detach().numpy()
plot_prototypes(loadings, names)
```

For more graphs of interest, see `main.py:plot_models` (e.g. logrank comparisons of the clusters, plotting the AUROC over time...)
