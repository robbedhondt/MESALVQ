import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import AutoMinorLocator
from scipy.stats import logrank, CensoredData
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.compare import compare_survival
from sksurv.metrics import cumulative_dynamic_auc
# from matplotlib.ticker import AutoMinorLocator
from .constants import (EVENT_NAMES, TARGET_NAME_MAPPING, FEATURE_NAME_MAPPING, 
    COLORS, LINESTYLES)

def plot_patient_trajectory(qs, pid):
    """
    Makes a line plot of the scores of patient `pid` on the different tests.
    Currently not very informative.
    """
    patient = qs.loc[qs.USUBJID == pid]
    plt.figure()
    for test in patient.QSTEST.unique():
        plt.plot(patient.QSDY.loc[patient.QSTEST == test], patient.QSSTRESC.loc[patient.QSTEST == test], ".-")
    return plt.gcf()

def plot_visit_dates(qs):
    """
    Gives summary overviews of the visit dates found in qs.
    """
    timestamps = np.concatenate(qs.groupby("USUBJID").QSDY.unique().values)
    # # Old implementation in separate plots
    # # Plot 1: enhanced boxplot
    # fig1 = plt.figure(figsize=(10,2))
    # sns.boxenplot(timestamps, orient="h", zorder=10, color="gray")
    # plt.xticks(np.arange(-500, 1900, 100), rotation=90)
    # plt.grid(axis="x")
    # plt.xlabel("Number of days after study start")
    # # Plot 2: histogram
    # fig2 = plt.figure(figsize=(10,3))
    # plt.hist(timestamps, bins=list(range(int(qs.QSDY.min()), int(qs.QSDY.max()))), color="gray")
    # plt.yscale("log")
    # plt.xticks(np.arange(-500, 1900, 100), rotation=90)
    # plt.ylim([0.75, plt.ylim()[1]])
    # plt.xlabel("Number of days after study start")
    # plt.ylabel("Number of visits")
    # return fig1, fig2
    fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(10,5), 
        height_ratios=(4,2), gridspec_kw=dict(hspace=0.05))
    # Plot 1: histogram (tried gaussian kde, doesn't look nice)
    bins = np.arange( np.floor(np.min(timestamps)), 
                      np.ceil( np.max(timestamps)) )
    ax[0].hist(timestamps, bins=bins, color="gray")
    ax[0].set_yscale("log")
    # ax[0].set_ylim([0.75, plt.ylim()[1]])
    ax[0].set_ylabel("Number of visits")
    # Plot 2: enhanced boxplot
    sns.boxenplot(timestamps, orient="h", zorder=10, color="gray", ax=ax[1])
    ticks = np.arange(-500, 1900, 100)
    ax[1].set_xticks(ticks, ticks, rotation=90)
    ax[1].grid(axis="x")
    ax[1].set_xlabel("Number of days after study start")

def plot_kaplan_meier(cens, time, showconf=False, figsize=(5,2), ylim=[0.33,1]):
    """this is specific to the MS application... plot_multievent_kaplan_meier is
    more general, use that one instead."""
    # from cycler import cycler
    # custom_cycler = cycler(linestyle=['-', '--', ':', '-.']) + plt.rcParams['axes.prop_cycle']
    linestyles = ['-', '--', ':', '-.', '-', '--', ':', '-.']

    plt.figure(figsize=figsize)
    # plt.gca().set_prop_cycle(custom_cycler)
    for j,test in enumerate(cens.columns):
        nona = ~time[test].isna()
        x, y, conf = kaplan_meier_estimator(cens[test].loc[nona], time[test].loc[nona], conf_type="log-log")
        plt.step(x, y, where="post", label=TARGET_NAME_MAPPING[test], ls=linestyles[j])#, ls=(0,(1,j)))
        if showconf:
            plt.fill_between(x, conf[0], conf[1], alpha=0.2)
    tmax = 100*np.ceil(np.max(time)/100) # Round to nearest 100 from below
    plt.legend(loc="center left", bbox_to_anchor=[1,0.5])
    plt.xticks(np.arange(0, tmax+1, 200), rotation=0)
    plt.yticks(np.arange(0,1.1,0.1))
    plt.xlim([0,tmax])
    plt.ylim(ylim)
    plt.xlabel("Time since study enrolment [days]")
    plt.ylabel("Probability of\nremaining stable")
    plt.grid()

def plot_multievent_kaplan_meier(y, showconf=False, figsize=(5,5), ylim=[0,1], event_names=None):
    if event_names is None:
        event_names = [f"event {j+1}" for j in range(y.shape[1])]
    linestyles = ['-', '--', ':', '-.', '-', '--', ':', '-.']
    cens = y[y.dtype.names[0]]
    time = y[y.dtype.names[1]]

    plt.figure(figsize=figsize)
    for j in range(y.shape[1]):
        nona = ~np.isnan(time[:,j])
        x, y, conf = kaplan_meier_estimator(cens[nona,j], time[nona,j], conf_type="log-log")
        plt.step(x, y, where="post", label=event_names[j], ls=linestyles[j])#, ls=(0,(1,j)))
        if showconf:
            plt.fill_between(x, conf[0], conf[1], alpha=0.2)
    plt.legend(loc="center left", bbox_to_anchor=[1,0.5])
    plt.xlim([0,np.max(time)])
    plt.ylim(ylim)
    plt.xlabel("Time")
    plt.ylabel("Survival probability")
    plt.grid()


def plot_heatmap_logrank(cens, time):
    pvals = pd.DataFrame(index=cens.columns, columns=cens.columns, dtype=float)
    for test1 in cens.columns:
        for test2 in cens.columns:
            lr = logrank(CensoredData.right_censored(time[test1].dropna(), cens[test1].dropna()),
                        CensoredData.right_censored(time[test2].dropna(), cens[test2].dropna()) )
            pvals.loc[test1, test2] = lr.pvalue
    sns.heatmap(pvals, cmap="hot", annot=True)
    plt.title("Logrank test p-value")

def plot_lifelines(cens, time):
    fig, ax = plt.subplots(figsize=(8,4), nrows=2, ncols=4, sharex=True)
    ax = ax.flatten()
    for i,test in enumerate(cens.columns):
        nona = ~time[test].isna()
        __show_survival_times(cens[test].loc[nona], time[test].loc[nona], ax=ax[i])
        ax[i].set_title(test[:12])
    fig.supxlabel("Days since study start")
    fig.supylabel("Patient (sorted by time)")
    plt.tight_layout()

def __show_survival_times(event, time, ax=None):
    assert len(event) == len(time)
    isort = np.argsort(time)
    time  = time.iloc[isort]
    colors = ["g" if c == 0 else "r" for c in event.iloc[isort]]

    if ax == None:
        plt.figure(figsize=(7,7))
    else:
        plt.sca(ax)
    plt.hlines(y=np.arange(len(time)), xmin=np.zeros(len(time)), xmax=time, colors=colors)
    plt.yticks([])
    plt.xlim([0, plt.xlim()[1]])
    plt.grid()

def plot_missingness(df, figsize=(15, 5)):
    import missingno as msno
    df = df.rename(FEATURE_NAME_MAPPING, axis="columns")
    isna = df.isna()
    colsums = isna.sum(axis=0)
    colsort = np.argsort(colsums.values) 
    isna = isna.iloc[:,colsort] # sort columns least --> most amount of missingness
    isna = isna.sort_values(by=list(isna.columns)) # sort rows with better grouping than msno itself
    isna = isna.replace(True, None)
    msno.matrix(isna, color=[0,0,0], figsize=figsize)
    # plt.gca().set_yticklabels(plt.gca().get_yticklabels(), fontsize=plt.rcParams["ytick.labelsize"])

def plot_relevance_matrix(slvq, names=None):
    rel_mat = slvq.lambda_mat().detach().numpy()
    assert rel_mat.shape[0] == rel_mat.shape[1]
    if names is None:
        names = np.array(range(rel_mat.shape[0]))
    assert len(names) == rel_mat.shape[0]
    fig = plt.matshow(rel_mat)
    cbar = plt.colorbar(fig)
    cbar.set_label('relevance', rotation=270)
    plt.xticks(ticks=range(rel_mat.shape[0]), labels=names, rotation=90)
    plt.yticks(ticks=range(rel_mat.shape[1]), labels=names)

def plot_feature_relevances(slvq, names=None, sort=True):
    rel_mat = slvq.lambda_mat().detach().numpy()
    if names is None:
        names = np.array(range(rel_mat.shape[0]))
    isort = np.argsort(np.diag(rel_mat))
    plt.figure(figsize=(5, max(5, len(names) / 6)))
    plt.barh(names[isort], np.diag(rel_mat)[isort], fc="k")
    plt.xlabel('Relevance score')
    plt.ylabel('Feature')
    plt.grid(axis="x")
    plt.tight_layout()

def plot_feature_relevances_local(models, names=None, event_names=None):
    if names is None:
        names = np.array(range(models[0].n_features))
    if event_names is None:
        event_names = [f"event {j+1}" for j in range(len(models))]
    fig, ax = plt.subplots(ncols=len(models), figsize=(max(10, 1.25*len(models)), 6), sharey=True, sharex=True)
    relevances = np.stack([np.diag(model.lambda_mat().detach().numpy()) for model in models], axis=1)
    isort = np.argsort(relevances.max(axis=1))
    for j,model in enumerate(models):
        ax[j].barh(names[isort], relevances[isort,j], fc="k")
        ax[j].grid()
        ax[j].set_title(event_names[j])
        ax[j].invert_yaxis()
    plt.subplots_adjust(wspace=0)
    ax[0].set_ylabel('Feature')
    fig.supxlabel('Relevance score')
    plt.tight_layout()

def plot_kaplan_meier_together(slvq, X, y_true):
    y_pred = slvq.predict(X, closest=True)
    
    linestyles = ['dashed', 'dotted', 'dashdot', 'solid']
    plt.figure(figsize=(5,5))
    for j in range(slvq.n_prototypes):
        for i in range(slvq.n_events):
            iloc = (y_pred == j)
            x, y = kaplan_meier_estimator(y_true["event"][iloc,i], y_true["time"][iloc,i])
            plt.step(x, y, where="post", label=f"{j+1} (event {i+1})", linestyle=linestyles[j])
            # plt.ylim(0, 1)
        plt.gca().set_prop_cycle(None)
    plt.legend(title="Prototype", loc="center left", bbox_to_anchor=(1,0.5))
    plt.xlabel("Time since study enrolment [days]")
    plt.ylabel("Probability of remaining stable")
    plt.tight_layout()

def plot_kaplan_meier_per_cluster(X, y_true, y_pred, subplots_params=dict(nrows=2, ncols=4, figsize=(8.5, 3.75)), event_names=None):
    if event_names is None:
        event_names = [TARGET_NAME_MAPPING[name] for name in EVENT_NAMES]
    # If just 1 global clustering is given, repeat it based on the number of targets
    if len(y_true.shape) != len(y_pred.shape):
        y_pred = np.array([y_pred]*y_true.shape[1]).T
    # figsize=(9,5) is the "unsqueezed" version
    fig, ax = plt.subplots(**subplots_params, sharex=True, sharey=True)
    ax = ax.flatten()
    for j in range(y_pred.shape[1]):
        for c in range(len(np.unique(y_pred[:,j]))):
            iloc = (y_pred[:,j] == c)
            if any(iloc):
                x, y = kaplan_meier_estimator(y_true["event"][iloc,j], y_true["time"][iloc,j])
                ax[j].step(x, y, where="post", label=c+1, ls=LINESTYLES[c], zorder=5)
        ax[j].grid()
        ax[j].set_title(event_names[j])
    ax[0].set_ylim([0,1])
    ax[0].set_xlim([0,ax[0].get_xlim()[1]])
    # fig.subplots_adjust(hspace=0.15, wspace=0.15)
    fig.supylabel("Probability of remaining stable")
    fig.supxlabel("Time since study enrolment [days]")
    plt.tight_layout()
    ilegend = min(3, y_true.shape[1]-1) # small hack to make it work for less than 4 events
    ax[ilegend].legend(loc="upper left", bbox_to_anchor=(1.1,0.2), title="Prototype", 
        framealpha=1.0, borderaxespad=0.0, edgecolor="black", fancybox=False)

def plot_kaplan_meier_per_cluster_oneplot(X, y_true, y_pred):
    # If just 1 global clustering is given, repeat it based on the number of targets
    if len(y_true.shape) != len(y_pred.shape):
        y_pred = np.array([y_pred]*y_true.shape[1]).T
    plt.figure(figsize=(6,4.5))
    colors = ["blue", "orange", "green", "red"]
    colors = plt.rcParams['axes.prop_cycle'].by_key()["color"][:4]
    for j in range(y_pred.shape[1]):
        for c in range(len(np.unique(y_pred[:,j]))):
            iloc = (y_pred[:,j] == c)
            if any(iloc):
                x, y = kaplan_meier_estimator(y_true["event"][iloc,j], y_true["time"][iloc,j])
                plt.step(x, y, where="post", label=["",c+1][j==0], 
                    ls=LINESTYLES[c], zorder=5, color=colors[c])
    tmax = 100*np.ceil(np.max(y_true["time"])/100) # Round to nearest 100 from below
    plt.xticks(np.arange(0, tmax+1, 200), rotation=0)
    plt.yticks(np.arange(0,1.1,0.1))
    plt.grid()
    plt.ylim([0,1])
    plt.xlim([0,plt.xlim()[1]])
    # fig.subplots_adjust(hspace=0.15, wspace=0.15)
    plt.ylabel("Probability of\nremaining stable")
    plt.xlabel("Time since study enrolment [days]")
    plt.tight_layout()
    plt.legend(
        loc="center left", bbox_to_anchor=(1,0.5), title="Prototype", 
        # framealpha=1.0, borderaxespad=0.0, edgecolor="black", fancybox=False
    )

def plot_timing_results(times):
    plt.figure(figsize=(5,5))
    plt.plot(times.t1, times.t2, ".")
    mean = times.mean().mean()
    plt.axline(xy1=[mean,mean], slope=1, c="k")
    plt.xlim([0.8*times.min().min(), 1.1*times.max().max()])
    plt.ylim([0.8*times.min().min(), 1.1*times.max().max()])
    plt.grid()
    plt.xlabel("Time to fit global model")
    plt.ylabel("Time to fit local models")

def plot_tuning_results():
    results = pd.read_csv(os.path.join("figures","tuned_results.csv"))

    for param in results.columns[:3]:
        fig, ax = plt.subplots(nrows=4, ncols=8, figsize=(20,10), sharex=True, sharey="row")
        plt.subplots_adjust(hspace=0.1, wspace=0.1)
        if (param == "epochs") | (param == "lr"):
            ax[0,0].set_xscale("log")
        for i, metric in enumerate(results.metric.unique()):
            for j, target in enumerate(results.columns[-8:]):
                for model in ["MTSLVQ", "STSLVQ"]:
                    loc = (results.model == model) & (results.metric == metric)
                    ax[i,j].plot(results.loc[loc, param], results.loc[loc, target], ".", label=model)
                ax[0,j].set_title(target[6:6+15])
                ax[-1,j].set_xlabel(param)
                ax[i,0].set_ylabel(metric)
        plt.savefig(os.path.join("figures", f"tuned_results_{param}.pdf"))
        plt.show()

def logrank_cluster_comparison(X_train, X_test, y_train, y_test, y_pred_train, y_pred_test):
    # # Pairwise logrank heatmap
    # j_event = 0
    # for j_event in range(8):
    #     plt.figure(figsize=(5,5))
    #     cens_per_cluster = [y_test["event"][y_pred_test == c, j_event] for c in np.unique(y_pred_test)]
    #     time_per_cluster = [y_test["time" ][y_pred_test == c, j_event] for c in np.unique(y_pred_test)]
    #     cens_per_cluster = pd.concat([pd.Series(c) for c in cens_per_cluster], axis=1)
    #     time_per_cluster = pd.concat([pd.Series(c) for c in time_per_cluster], axis=1)
    #     plot_heatmap_logrank(cens_per_cluster, time_per_cluster)

    # If just 1 global clustering is given, repeat it based on the number of targets
    if len(y_train.shape) != len(y_pred_train.shape):
        y_pred_train = np.array([y_pred_train]*y_train.shape[1]).T
        y_pred_test  = np.array([y_pred_test ]*y_test.shape[1]).T
    print(f"{' '*10} | train | test  | logxx | xx (multiplicator)")
    for j in range(y_train.shape[1]):
        _, pval_train  = compare_survival(y_train[:,j], y_pred_train[:,j])
        _, pval_test   = compare_survival( y_test[:,j], y_pred_test[:,j])
        print(f"{EVENT_NAMES[j][6:6+10]} | {pval_train:4.0e} | {pval_test :4.0e} | {np.log10(pval_test/pval_train):5.2f} | {pval_test/pval_train:.0f}")

def cluster_counts(y_pred_global, y_pred_local):
    vc = [pd.Series(y_pred_global    ).value_counts().rename("MTSVLQ")] + [
          pd.Series(y_pred_local[:,j]).value_counts().rename(f"SESLVQ_{j}")
                for j in range(y_pred_local.shape[1])]
    vc = pd.concat(vc, axis=1).fillna(0).astype(int)
    return vc

def plot_auroc(y_train, y_test, y_pred_test_survfunc):
    times = np.linspace(30, 1150, 100)
    means = []
    plt.figure()
    for j in range(y_test.shape[1]):
        y_pred_test = np.array([yy(times) for yy in y_pred_test_survfunc[:,j]])
        y_pred_test *= -1

        iloc = (y_test[:,j]["time"] == max(y_test[:,j]["time"])) & (y_test[:,j]["event"] == 1)

        auc, mean_auc = cumulative_dynamic_auc(y_train[:,j], y_test[~iloc,j], y_pred_test[~iloc], times)
        means.append(mean_auc)
        plt.plot(times, auc, "-")
        plt.ylim([0,1])
        plt.axhline(y=0.5, c="red")
    means = [str(m.round(3)) for m in means]
    plt.title("Average AUROC\n" + " | ".join(means))

def plot_prototypes(loadings, names, ticks_right=True, sort_by=None):
    # Organize the loadings into a dataframe
    df = pd.DataFrame(loadings.T, index=names, columns=range(1, loadings.shape[0]+1))
    if sort_by is not None:
        # # Sort by raw value of that prototype
        # df = df.sort_values(by=sort_by, key=lambda t: -t.abs())
        # # Sort by mean absolute difference of that prototype w.r.t the other ones
        isort = np.argsort(np.mean(np.abs(df[[sort_by]].values - df.values), axis=1))[::-1]
        df = df.iloc[isort]

    # Choose a colormap
    colors = plt.get_cmap("rocket")(np.linspace(0.1,0.9,loadings.shape[0]))
    # colors = plt.get_cmap("magma", loadings.shape[0])(range(loadings.shape[0]))
    # colors = sns.color_palette("rocket", loadings.shape[0])

    # Plot the bars
    plt.figure(figsize=(5, len(df.index)/4))
    for col in df.columns: # init the bar legend
        plt.barh(y=-5, width=0, color=colors[col-1], label=col)
    for i,feature in enumerate(df.index):
        isort = np.argsort(df.loc[feature].abs())[::-1] # make smallest bars drawn on top
        for col in df.columns[isort]:
            plt.barh(y=i, width=df.loc[feature,col], color=colors[col-1], zorder=5)
    plt.yticks(range(len(df.index)), df.index)
    plt.ylim([len(df.index)-0.5, -0.5])
    xrange = max(np.abs(plt.xlim()))
    plt.xlim([-xrange, xrange])
    plt.xlabel("$z$-score")
    plt.gca().yaxis.set_minor_locator(AutoMinorLocator(2))
    plt.grid(which="minor")
    plt.grid(axis="x")
    if ticks_right:
        plt.gca().yaxis.set_label_position("right")
        plt.gca().yaxis.tick_right()
    plt.gca().yaxis.set_ticks_position('none')
    plt.legend(title="Prototype", loc="upper "+["right","left"][ticks_right], 
            framealpha=1.0, borderaxespad=0.0, edgecolor="black", fancybox=False)

def plot_prototypes_transposed(loadings, names, ticks_top=True):
    # Organize the loadings into a dataframe
    df = pd.DataFrame(loadings.T, index=names, columns=range(1, loadings.shape[0]+1))
    # Choose a colormap
    colors = plt.get_cmap("rocket")(np.linspace(0.1,0.9,loadings.shape[0]))
    # Plot the bars
    plt.figure(figsize=(9, 3))
    for col in df.columns: # init the bar legend
        plt.bar(x=-5, height=0, color=colors[col-1], label=col)
    for i,feature in enumerate(df.index):
        isort = np.argsort(df.loc[feature].abs())[::-1] # make smallest bars drawn on top
        for j,col in enumerate(df.columns[isort]):
            widths = [0.8, 0.7, 0.6, 0.5]
            # widths = [0.8, 0.6, 0.4, 0.2]
            plt.bar(x=i, height=df.loc[feature,col], width=widths[j], color=colors[col-1], zorder=5)
    plt.xticks(range(len(df.index)), df.index, rotation=45, ha="left")
    plt.xlim([-0.5, len(df.index)-0.5])
    yrange = max(np.abs(plt.ylim()))
    plt.ylim([-yrange, yrange])
    plt.ylabel("$z$-score")
    plt.gca().xaxis.set_minor_locator(AutoMinorLocator(2))
    plt.grid(which="minor")
    plt.grid(axis="y")
    if ticks_top:
        plt.gca().xaxis.set_label_position("top")
        plt.gca().xaxis.tick_top()
    # plt.gca().xaxis.set_ticks_position('none')
    plt.gca().tick_params(which="minor", length=0)
    plt.axhline(y=0, color="black", zorder=10, ls="dotted", lw=1)
    plt.legend(title="Prototype", loc="lower left",
            framealpha=1.0, borderaxespad=0.0, edgecolor="black", fancybox=False,
            ncols=2)

def plot_cluster_validity():
    df = pd.read_csv("figures/tuned_cluster_validity.csv")
    df = df.rename(columns={"Unnamed: 0":"model"})
    df = df.drop(columns=["silhouette_w_index"])
    df = df.iloc[:10,:]
    df.loc[1.5] = ["-------------", 0, 0, 0, 0, 0]
    df = df.sort_index()
    df = pd.melt(df, id_vars=["model"])
    df.value = df.value.astype(float)
    tt = list(TARGET_NAME_MAPPING.values())
    df.model = df.model.replace({**{"MTSLVQ":"MESA-LVQ", "STSLVQ":"SurvivalLVQ mean"}, **{f"STSLVQ_{i}":f"{tt[i]} SurvivalLVQ" for i in range(8)}})
    df.variable = df.variable.map({
        "negated_ball_hall_index":"$-$Ball-Hall",
        "calinski_harabasz_index":"Calinski-Harabasz", 
        "negated_davies_bouldin_index":"$-$Davies-Bouldin", 
        "silhouette_index":"Silhouette",
        "generalised_dunn_index":"Generalised Dunn"})

    fig, ax = plt.subplots(ncols=5, figsize=(10,5), sharey=True)
    for i,metric in enumerate(df.variable.unique()):
        # plt.plot(df.loc[df.variable == metric])
        plt.sca(ax[i])
        sns.barplot(df.loc[df.variable == metric], y="model", x="value", facecolor="grey")
        plt.xlabel(metric)
        plt.ylabel("")

def plot_performance_metrics():
    df = pd.read_csv("figures/tuned_results_test.csv")
    # Hardcode CoxPH results
    df.loc[ 8] = ["harrell_c"   , "CoxPH", 0.57927454,0.64066102,0.70091234,0.67165944,0.65640009,0.66550709,0.64552165,0.68243845]
    df.loc[ 9] = ["unos_c"      , "CoxPH", 0.58104982,0.62699506,0.66615968,0.65891820,0.64063223,0.65607454,0.65293135,0.68162439]
    df.loc[10] = ["integr_auroc", "CoxPH", 0.61315437,0.65980211,0.71985634,0.69684831,0.67018855,0.67782687,0.66220551,0.71849841]
    df.loc[11] = ["integr_brier", "CoxPH", 0.18215312,0.19148632,0.16708830,0.16957594,0.17829584,0.17746888,0.19042020,0.17839357]
    # Unstack the dataframe and change varnames for figure
    df = pd.melt(df, id_vars=["metric","model"])
    df.variable = df.variable.map(TARGET_NAME_MAPPING)
    df.model = df.model.map({"MTSLVQ":"MESA-LVQ", "STSLVQ":"SurvivalLVQ", "CoxPH":"CoxPH"})
    metric_mapping = {"harrell_c":"C (Harrell)", "unos_c":"C (Uno)", "integr_brier": "IBS", "integr_auroc":"IAUROC"}
    df.metric = df.metric.map(metric_mapping)

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10,5), sharex=True, sharey=True)
    ax = ax.flatten()
    fig.subplots_adjust(wspace=0.05, hspace=0.2)
    for i,metric in enumerate(metric_mapping.values()):
        plt.sca(ax[i])
        sns.barplot(df.loc[df.metric == metric], hue="model", x="variable", y="value")
        plt.xlabel("")
        plt.ylabel("")
        plt.title(metric)
        plt.ylim([0,1])
        ax[i].get_legend().remove()
    fig.legend(*ax[0].get_legend_handles_labels(), loc="center", bbox_to_anchor=(0.5,0.5), title="Model", framealpha=1.0, borderaxespad=0.0, edgecolor="black", fancybox=False)
