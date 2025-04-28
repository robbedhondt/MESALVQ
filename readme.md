# Multi-Event Survival Analysis-Informed Clustering
This repository hosts the code used in the publication "Multi-Event Survival-Informed Clustering for Time-to-Worsening in Multiple Sclerosis".

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
- To make sure this clone is part of the python path, you can make an editable install via pip with `pip install -e .` in the current directory. Alternatively, in your Python script or notebook, add `import sys; sys.path.append("path/to/this/clone/")`.

# Using the code in your application
First, load in your data. Make sure it is in the following format: 
- `X` is a pandas DataFrame or a 2D numpy array (can contain missing values, but is advised to be skew transformed and z-standardized -- see SkewTransformer in the original SurvivalLVQ repository).
- `y` is a structured array, where the first element is the event indicator and the second element the observed time (time-to-event or time-to-censoring). Similar as in `scikit-surv`, but now each element is a 2D array. If you have a 2D array of event indicators `cens` and a 2D array of observed times `time` (where rows = observations, columns = different events), you can transform these into the 2D structured array with a utility function:
    ```python
    from mesaLVQ.data import multi_surv_from_arrays
    y = multi_surv_from_arrays(cens, time)
    ```

Fitting MESA-LVQ then becomes as simple as
```python
from mesaLVQ.model import MultiEventSurvivalLVQ
mesalvq = MultiEventSurvivalLVQ(n_prototypes=4)
mesalvq.fit(X, y)
```

The clustering labels are then available through `mesalvq.predict(X, closest=True)`. 
Individualized risk scores and survival functions can be obtained with `mesalvq.predict(X)` and `mesalvq.predict_survival_function(X)`.
For a more detailed usage use case, see `example.ipynb`.

# Replicating the results from the paper
Follow the following steps to replicate the results from the paper:
- Create the following subfolders: `data`, `figures`, `models`
- Apply for MSOAC Placebo Database
- Place all the `.xpt` data files into `data`
- In `main.py`, at the bottom, write `run_models_tuned(retune=True, refit=True)`
- Run `python3.11 main.py`
