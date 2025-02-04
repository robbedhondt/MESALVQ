import os
import time
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sksurv.util import Surv
from sklearn.model_selection import train_test_split, ParameterGrid #, GridSearchCV, StratifiedKFold
# from sklearn.impute import SimpleImputer, KNNImputer
# from sklearn.metrics import make_scorer
# from sksurv.compare import compare_survival
# from sksurv.nonparametric import kaplan_meier_estimator

from mesaLVQ.data import load_kfss_and_edss, load_time_to_worsening, load_xy, split_train_test
from mesaLVQ.plot import *
from mesaLVQ.model import SurvivalLVQ, MultiEventSurvivalLVQ, LocalMultiEventSurvivalLVQ, compute_metric
from mesaLVQ.constants import EVENT_NAMES, PATH_FIGS, TARGET_NAME_MAPPING

def eda_figures(subset=None, save_path=PATH_FIGS):
    # Load relevant datasets
    qs = load_kfss_and_edss()
    cens, time = load_time_to_worsening()
    X, y = load_xy()
    X = X.rename(FEATURE_NAME_MAPPING, axis="columns")

    # Generate and save plots
    fignames = { # figname: (func, [args])
        "edss_visits"    : (plot_visit_dates    , [qs]        ),
        "kaplan_meier"   : (plot_kaplan_meier   , [cens, time]),
        "heatmap_logrank": (plot_heatmap_logrank, [cens, time]),
        "lifelines"      : (plot_lifelines      , [cens, time]),
        "missingness"    : (plot_missingness    , [X]         ),
    }
    for figname, (func, args) in fignames.items():
        func(*args)
        plt.savefig(os.path.join(save_path, f"eda_{figname}.pdf"), bbox_inches="tight")
        plt.close()

def fit_models(meslvq, seslvq, X, y, save_prefix=None):
    # - Multi-event
    t1 = time.time()
    meslvq.fit(X, y)
    t1 = time.time() - t1
    # - Single-event
    t2 = time.time()
    seslvq.fit(X, y)
    t2 = time.time() - t2
    if save_prefix is not None:
        fpath = lambda mname: os.path.join("models", f"{save_prefix}_{mname}.pkl")
        pickle.dump(meslvq, open(fpath("MTSLVQ"), 'wb'))
        pickle.dump(seslvq, open(fpath("STSLVQ"), 'wb'))
    return t1, t2

def score_single_model(model, X_train, X_test, y_train, y_test, subset=False, times=np.linspace(30, 1150, 100)):
    """subset: boolean flag indicating whether to only compute a fast subset"""
    y_pred = model.predict(X_test)
    if subset:
        y_pred_survfunc = None
        isub = slice(0,2)
    else:
        y_pred_survfunc = model.predict_survival_function(X_test)
        isub = slice(None)
    metrics = ["harrell_c", "unos_c", "integr_brier", "integr_auroc"][isub]
    results = pd.DataFrame(np.nan, index=metrics, columns=EVENT_NAMES)
    for metric in results.index:
        results.loc[metric, :] = compute_metric(
            metric, y_train, y_test, y_pred, y_pred_survfunc, times=times
        )
    return results

def score_models(meslvq, seslvq, X_train, X_test, y_train, y_test, subset=False):
    results = {}
    for name, model in {"MTSLVQ":meslvq, "STSLVQ":seslvq}.items():
        results[name] = score_single_model(
            model, X_train, X_test, y_train, y_test, subset=subset)
    results = pd.concat(results).swaplevel().sort_index()
    results.index = results.index.rename(["metric","model"])
    return results

def plot_models(meslvq, seslvq, X_train, X_test, y_train, y_test, names=None, save_prefix=None, save_path=PATH_FIGS, presentation_subset=False):
    event_names = [TARGET_NAME_MAPPING[EVENT_NAMES[j]] for j in range(len(models))]
    if names is None:
        names = list(range(X_train.shape[0]))
    def savefig(name):
        if save_prefix is not None:
            plt.savefig(os.path.join(save_path, f"{save_prefix}_{name}.pdf"), bbox_inches="tight")
    def savetab(tab, name):
        if save_prefix is not None:
            tab.to_csv(os.path.join(save_path, f"{save_prefix}_{name}.csv"))
    y_pred_global_train = meslvq.predict(X_train, closest=True)
    y_pred_global_test  = meslvq.predict(X_test , closest=True)
    y_pred_local_train  = seslvq.predict(X_train, closest=True)
    y_pred_local_test   = seslvq.predict(X_test , closest=True)

    # FEATURE RELEVANCES
    plot_feature_relevances(meslvq, names=names)
    savefig("relevances_global")
    plot_feature_relevances_local(seslvq, names=names, event_names=event_names)
    savefig("relevances_local")

    # KAPLAN-MEIER
    # > Global
    plot_kaplan_meier_alt(X_train, y_train, y_pred_global_train)
    savefig("kaplan_meier_global_train")
    if not presentation_subset:
        plot_kaplan_meier_alt(X_test, y_test, y_pred_global_test)
        savefig("kaplan_meier_global_test")
    # > Local
    plot_kaplan_meier_alt(X_train, y_train, y_pred_local_train)
    savefig("kaplan_meier_local_train")
    if not presentation_subset:
        plot_kaplan_meier_alt(X_test, y_test, y_pred_local_test)
        savefig("kaplan_meier_local_test")
    # > Logrank comparison
    if not presentation_subset:
        print("global logrank comparison of clusters")
        logrank_cluster_comparison(X_train, X_test, y_train, y_test, y_pred_global_train, y_pred_global_test)
        print("local logrank comparison of clusters")
        logrank_cluster_comparison(X_train, X_test, y_train, y_test, y_pred_local_train, y_pred_local_test)

    # CLUSTER COUNTS
    if not presentation_subset:
        vc_train = cluster_counts(y_pred_global_train, y_pred_local_train)
        vc_test  = cluster_counts(y_pred_global_test , y_pred_local_test )
        print("train cluster counts\n", vc_train)
        print("test  cluster counts\n", vc_test)
        savetab(vc_train, "clustercounts_train")
        savetab(vc_test , "clustercounts_test" )

    # AUROC OVER TIME
    if not presentation_subset:
        y_pred_test = meslvq.predict_survival_function(X_test)
        plot_auroc(y_train, y_test, y_pred_test)
        savefig("AUROC_global")
        y_pred_test = seslvq.predict_survival_function(X_test)
        plot_auroc(y_train, y_test, y_pred_test)
        savefig("AUROC_local")

    # PROTOTYPE LOADINGS (only for global model)
    loadings = meslvq.w.detach().numpy() # prototype x features matrix
    plot_prototypes(loadings, names)
    savefig("loadings_global")
    plot_prototypes_transposed(loadings, names)
    savefig("loadings_global_transposed")

    # CLUSTER VALIDITY CRITERIA
    if not presentation_subset:
        scores = compute_cluster_validity(X_train, y_pred_global_train, y_pred_local_train)
        savetab(scores, "cluster_validity")
        print(scores)

def run_models_simple(refit=False):
    """Run models and evaluation metrics with default parameters."""
    # Load the data
    X, y = load_xy()
    X_train, X_test, y_train, y_test = split_train_test(X, y)

    # Get the trained models
    if refit:
        meslvq = MultiEventSurvivalLVQ()
        seslvq = LocalMultiEventSurvivalLVQ([SurvivalLVQ() for _ in range(y.shape[1])])
        t1, t2 = fit_models(meslvq, seslvq, X_train, y_train, save_prefix="simple")
        # Report timing results
        print(f"Fitting mesaLVQ    : {t1:.3f} seconds") # 21.001 seconds
        print(f"Fitting SurvivalLVQ: {t2:.3f} seconds") # 38.978 seconds
    else:
        fpath = lambda mname: os.path.join("models", f"simple_{mname}.pkl")
        meslvq = pickle.load(open(fpath("MTSLVQ"), "rb"))
        seslvq = pickle.load(open(fpath("STSLVQ"), "rb"))

    # Generate performance metrics
    results = score_models(meslvq, seslvq, X_train, X_test, y_train, y_test)
    print(results)
    results.to_csv(os.path.join(PATH_FIGS, "simple_results.csv"))

    # Generate comparison figures
    names = np.array([FEATURE_NAME_MAPPING[name] for name in X.columns])
    plot_models(meslvq, seslvq, X_train, X_test, y_train, y_test, names=names, save_prefix="simple")

def run_models_tuned(retune=False, refit=False):
    # Load the data
    X, y = load_xy()
    X_train     , X_test , y_train     , y_test  = split_train_test(X, y)
    X_train_tune, X_valid, y_train_tune, y_valid = split_train_test(pd.DataFrame(X_train), y_train)

    # Tune the models
    param_grid = ParameterGrid({
        "lr"          : [1e-3, 1e-2, 1e-1],
        "epochs"      : [16, 32, 64, 128],
        "n_prototypes": [2, 3, 4, 5],
    })
    if retune:
        times = []
        scores = []
        for i,param_comb in enumerate(param_grid):
            # Update hyperparameters and refit models
            meslvq = MultiEventSurvivalLVQ(verbose=False, **param_comb)
            seslvq = LocalMultiEventSurvivalLVQ([SurvivalLVQ(verbose=False, 
                    **param_comb) for _ in range(y.shape[1])])
            t1, t2 = fit_models(meslvq, seslvq, X_train_tune, y_train_tune)
            times.append([t1,t2])
            # Score models and save to the list
            results = score_models(meslvq, seslvq, X_train_tune, X_valid, y_train_tune, y_valid)
            param_comb = pd.Series(param_comb)
            results[param_comb.index] = param_comb.values
            results = results.reset_index().set_index(list(param_comb.index) + ["metric","model"])
            scores.append(results)
            # Intermediate save
            pd.concat(scores, axis=0).to_csv(os.path.join(PATH_FIGS, "tuned_results.csv"))
            c_index_edss = results.loc[
                results.index.get_level_values("metric") == "harrell_c", 
                results.columns[0]
            ].values
            print(f"{i:02d}/{len(param_grid)}: {t1:.2f} <-> {t2:.2f} || C-index EDSS: {c_index_edss}")
        scores = pd.concat(scores, axis=0)
        scores.to_csv(os.path.join(PATH_FIGS, "tuned_results.csv"))
        times = pd.DataFrame(times, columns=["t1","t2"])
        times.to_csv(os.path.join(PATH_FIGS, "tuned_times.csv"))
    scores = pd.read_csv(os.path.join(PATH_FIGS, "tuned_results.csv"))
    scores = scores.convert_dtypes()
    scores = scores.set_index(list(param_grid[0].keys()) + ["model","metric"])
    times = pd.read_csv(os.path.join(PATH_FIGS,"tuned_times.csv"), index_col=0)
    td = times.t2 - times.t1
    print(f"Tuning: on {times.shape[0]} paramcombs, MESALVQ was on average " +
        f"{td.mean():.3f} +- {td.std():.3f} seconds faster than local " +
        f"({td.describe().round(3)})")

    # # Find the best parameters based on HCI (Harrell's C index)
    # uci = scores.loc[scores.index.get_level_values("metric") == "harrell_c"]
    # names = scores.index.names[:-2] # drop ("model", "metric")
    # # > global model
    # pglobal = uci.loc[uci.index.get_level_values("model") == "MTSLVQ"].mean(axis=1)
    # pglobal = pglobal.index[pglobal.argmax()][:-2]
    # pglobal = {k:v for k,v in zip(names, pglobal)}
    # print(f"global model - optimal parameters {pglobal}")
    # # > local model
    # plocal = {}
    # print("local model - optimal parameters")
    # for col in uci.columns:
    #     plocal[col] = uci.loc[uci.index.get_level_values("model") == "STSLVQ", col]
    #     plocal[col] = plocal[col].index[plocal[col].argmax()][:-2]
    #     plocal[col] = {k:v for k,v in zip(names, plocal[col])}
    #     print(f"    {col}: {plocal[col]}")
    pnames = list(param_grid.param_grid[0].keys())
    def find_optimal_hyperparameters(scores, only_hci=True):
        from sklearn.preprocessing import StandardScaler
        tt = scores.copy()
        tt.loc[tt.index.get_level_values("metric") == "integr_brier"] *= -1
        tt = tt.reset_index()
        tt = tt.pivot(columns="metric", values=tt.columns[-1], index=pnames)
        if only_hci:
            tt = tt[["harrell_c"]]
        tt = pd.DataFrame(StandardScaler().fit_transform(tt), index=tt.index, columns=tt.columns)
        tt = tt.mean(axis=1)
        return tt.index[tt.argmax()]
    pglobal = find_optimal_hyperparameters(
        scores.loc[scores.index.get_level_values("model") == "MTSLVQ"].mean(axis=1))
    pglobal = {k:v for k,v in zip(pnames, pglobal)}
    print(pglobal)
    plocal = {}
    for target in scores.columns:
        plocal[target] = find_optimal_hyperparameters(
            scores.loc[scores.index.get_level_values("model") == "STSLVQ", target])
        plocal[target] = {k:v for k,v in zip(pnames, plocal[target])}
        print(plocal[target])

    # Retrain on train+valid
    if refit:
        meslvq = MultiEventSurvivalLVQ(verbose=False, **pglobal)
        seslvq = LocalMultiEventSurvivalLVQ([SurvivalLVQ(verbose=False, 
                **plocal[col]) for col in scores.columns])
        t1, t2 = fit_models(meslvq, seslvq, X_train, y_train, save_prefix="tuned")
    else:
        fpath = lambda mname: os.path.join("models", f"tuned_{mname}.pkl")
        meslvq = pickle.load(open(fpath("MTSLVQ"), "rb"))
        seslvq = pickle.load(open(fpath("STSLVQ"), "rb"))

    # Generate performance metrics
    results = score_models(meslvq, seslvq, X_train, X_test, y_train, y_test)
    print(results)
    results.to_csv(os.path.join(PATH_FIGS, "tuned_results_test.csv"))

    # Generate comparison figures
    names = np.array([FEATURE_NAME_MAPPING[name] for name in X.columns])
    plot_models(meslvq, seslvq, X_train, X_test, y_train, y_test, names=names, save_prefix="tuned")

    plot_timing_results(times)
    plt.savefig(os.path.join(PATH_FIGS, "fitting_time.pdf"))

def presentation():
    # SETUP
    plt.rcParams["font.family"] = "Latin Modern Sans"
    save_path = "/mnt/c/Users/u0131222/PhD/Presenting/2025-01-30_LabMeeting_MESALVQ/Figures/"
    def savefig(name):
        plt.savefig(f"{save_path}{name}.pdf", bbox_inches="tight")
    
    # EDA
    eda_figures(subset=["edss_visits","missingness"], save_path=save_path)
    # > kaplan meier separate so we can adapt figsize
    cens, time = load_time_to_worsening()  
    plot_kaplan_meier(cens, time, figsize=(6,4.5), ylim=[0,1])
    savefig("eda_kaplan_meier")

    # MODELS
    X, y = load_xy()
    X_train, X_test, y_train, y_test = split_train_test(X, y)
    fpath = lambda mname: os.path.join("models", f"tuned_{mname}.pkl")
    meslvq = pickle.load(open(fpath("MTSLVQ"), "rb"))
    seslvq = pickle.load(open(fpath("STSLVQ"), "rb"))
    names = np.array([FEATURE_NAME_MAPPING[name] for name in X.columns])
    plot_models(meslvq, seslvq, X_train, X_test, y_train, y_test, names=names, save_prefix="tuned", save_path=save_path, presentation_subset=True)
    # New Kaplan-Meier
    y_pred = meslvq.predict(X_train, closest=True)
    plot_kaplan_meier_global_oneplot(X_train, y_train, y_pred)
    savefig("kaplan_meier_global_oneplot")
    
    # OTHER
    plot_cluster_validity()
    savefig("cluster_validity")
    plot_performance_metrics()
    savefig("performance_metrics")


if __name__ == "__main__":
    # eda_figures()
    # run_models_simple(refit=False)
    # run_models_tuned(retune=False, refit=False)
    presentation()


