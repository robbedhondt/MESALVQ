import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from genieclust.cluster_validity import (
    negated_ball_hall_index,
    calinski_harabasz_index,
    negated_davies_bouldin_index,
    # negated_wcss_index,
    # wcnn_index,
    silhouette_index,
    # silhouette_w_index,
    generalised_dunn_index,
)
from sksurv.metrics import (
    concordance_index_censored, 
    concordance_index_ipcw, 
    integrated_brier_score, 
    cumulative_dynamic_auc
)

def compute_cluster_validity(X, y_pred):
    """
    Compute internal clustering validity criteria for the given clustering `y_pred`.
    
    @param X: Input dataframe. Missing values will be imputed with mean imputation.
    @param y_pred: Vector of clustering labels.
    @return: Dictionary {k:v} with k=metricname and v=score.
    """
    metrics = [
        negated_ball_hall_index,
        calinski_harabasz_index,
        negated_davies_bouldin_index,
        silhouette_index,
        # silhouette_w_index,
        generalised_dunn_index,
    ]
    X_imputed = SimpleImputer().fit_transform(X)
    scores = {}
    # Make sure y_pred consists of the consecutive 0, 1, 2... without gaps
    y_pred = np.unique(y_pred, return_inverse=True)[1] 
    for metric in metrics:
        scores[metric.__name__] = metric(X_imputed, y_pred)
    return scores

def compare_cluster_validity(X, y_pred_global, y_pred_local):
    """
    Compares the cluster validity metrics between global and local approach.

    @param X: Input dataframe. Missing values will be imputed with mean imputation.
    @param y_pred_global: Vector with the global clustering labels.
    @param y_pred_local: Matrix where each column represents the clustering 
        labels for one of the local models.
    @return: Pandas dataframe with scores, a.o. average local score.
    """
    scores_global =  compute_cluster_validity(X, y_pred_global)
    scores_local  = [compute_cluster_validity(X, y_pred_local[:,j]) 
                        for j in range(y_pred_local.shape[1])]
    scores_local_mean = {k:np.mean([scores[k] for scores in scores_local]) 
                        for k in scores_local[0].keys()}
    index = ["global","local (mean)"]+[f"local_{j}" for j in range(y_pred_local.shape[1])]
    scores = pd.DataFrame((scores_global, scores_local_mean, *scores_local), index=index).T
    scores["global_is_better"] = scores["global"] > scores["local (mean)"]
    return scores.T#.round(3)

def compute_predictive_performance(y_train, y_test, y_pred, y_pred_survfunc, times=None):
    """
    Compute predictive performance metrics for the given predictions.

    @param y_train: Multi-event survival times on the training set.
    @param y_test: Multi-event survival times on the test set.
    @param y_pred: Matrix where each column contains the predicted risk scores
        for one event.
    @param y_pred_survfunc: Matrix where each column contains the predicted
        survival function (as stepfunctions) for one event.
    @param times: Time points for which integrated metrics (Brier, AUROC) are 
        calculcated and averaged. If not given, the 10\% and 90\% quantiles in
        y_test are used.
    @return: Dictionary {k:v} with k=metricname and v=vector of scores (one for
        each event).
    """
    if times is None:
        times = np.linspace(*np.quantile(y_test["time"], [0.1, 0.9]), 100)
    metrics = ["harrell_c", "unos_c", "integr_brier", "integr_auroc"]
    results = {}
    for metric in metrics:
        results[metric] = compute_metric(metric, y_train, y_test, y_pred, y_pred_survfunc, times)
    return results

def compare_predictive_performance(mesalvq, sesalvq, X_train, y_train, X_test, y_test, times=None, event_names=None):
    if times is None:
        times = np.linspace(*np.quantile(y_test["time"], [0.1, 0.9]), 100)
    if event_names is None:
        event_names = np.array(range(y_test.shape[1]))
    metrics = ["harrell_c", "unos_c", "integr_brier", "integr_auroc"]
    results = {}
    for name, model in {"MESA-LVQ":mesalvq, "SurvivalLVQ":sesalvq}.items():
        y_pred = model.predict(X_test)
        y_pred_survfunc = model.predict_survival_function(X_test)
        res = pd.DataFrame(np.nan, index=metrics, columns=event_names)
        for metric in metrics:
            res.loc[metric, :] = compute_metric(
                metric, y_train, y_test, y_pred, y_pred_survfunc, times=times
            )
        results[name] = res
    results = pd.concat(results).swaplevel().sort_index()
    results.index = results.index.rename(["metric","model"])
    return results

def compute_metric(name, y_train, y_test, y_test_pred, y_test_pred_survfunc, times=np.linspace(30, 1150, 100)):
    # Make the code work also for single-event
    y_train              = np.atleast_2d(y_train             )
    y_test               = np.atleast_2d(y_test              )
    y_test_pred          = np.atleast_2d(y_test_pred         )
    if y_test_pred_survfunc is not None:
        y_test_pred_survfunc = np.atleast_2d(y_test_pred_survfunc)
    # Compute all the scores
    scores = []
    for k in range(y_test.shape[1]):
        scores.append(__compute_metric_single_event(
            name, y_train[:,k], y_test[:,k], y_test_pred[:,k], y_test_pred_survfunc[:,k], times=times
        ))
    return np.squeeze(scores) # squeeze superfluous dimension if single-target

def __compute_metric_single_event(name, y_train, y_test, y_test_pred, y_test_pred_survfunc, times=np.linspace(30, 1150, 100)): # since max(y_test["time"][y_test["event"] == 1]) >= 1187
    if name == "harrell_c":
        return concordance_index_censored(y_test["event"], y_test["time"], y_test_pred)[0]
    elif name == "unos_c":
        # tau = None
        # if max(y_test["time"]) >= max(y_train["time"]):
        #     tau = y_train["time"]
        tau = max(y_train["time"])
        return concordance_index_ipcw(y_train, y_test, y_test_pred, tau=tau)[0]
    elif name == "integr_brier":
        y_test_pred_survfunc = [yy(times) for yy in y_test_pred_survfunc]
        return integrated_brier_score(y_train, y_test, y_test_pred_survfunc, times)
    elif name == "integr_auroc":
        # One individual has the event at the maximum censoring time, leading to
        # a ValueError ("censoring survival function is zero at one or more time 
        # points"). Skipping this individual in the evaluation resolves the error.
        iloc = (y_test["time"] == max(y_test["time"])) & (y_test["event"] == 1)
        # return cumulative_dynamic_auc(y_train, y_test[~iloc], y_test_pred[~iloc], times)[1]
        y_test_pred_survfunc = np.array([yy(times) for yy in y_test_pred_survfunc]) 
        y_test_pred_survfunc *= -1 # some ranking problem otherwise...
        return cumulative_dynamic_auc(y_train, y_test[~iloc], y_test_pred_survfunc[~iloc,:], times)[1]
    else:
        raise ValueError(f"Unknown metric '{name}'")
