import numpy as np
import pandas as pd

def define_progression_events(df, pid="USUBJID", time="QSDY", edss="QSORRES"):
    """Converts the given dataframe of EDSS visits to a survival analysis one,
    with as event "time to significant EDSS worsening".
    
    @param df  : A multiple-instance pandas dataframe (visits per patient).
    @param pid : Column in edss grouping the different patients together.
    @param time: Column in edss representing the patient visit times.
    @param edss: Column in edss representing the outcome measure.
    @return: A dataframe with one row per patient and columns "cens" and "time":
        - "cens" = censoring status (False for censored, True for observed)
        - "time" = time to event 
    """
    df = pd.DataFrame(df)
    df = df.sort_values(by=[pid, time]) # Primary sort on ID, secondary on date
    df_surv = pd.DataFrame(
            index=df[pid].unique(), columns=["cens","time"]
        ).astype({"cens":"boolean", "time":float})
    for patient in df_surv.index:
        df_patient = df.loc[df[pid]==patient, :]
        out = __define_event(df_patient[time], df_patient[edss])
        df_surv.loc[patient, "cens"] = out[0]
        df_surv.loc[patient, "time"] = out[1]
    return df_surv

def __define_event(time, edss):
    """Defines the time to progression label for a single patient.

    @param time: Pandas Series of patient visit times (sorted low to high).
    @param edss: Pandas Series of outcome measurements.
    @return (censoring status, event time)
    """
    # Only define time to worsening if we can extract baseline EDSS
    # and at least one visit after baseline
    if (not any(time <= 0)) or (not any(time > 0)):
        return pd.NA, pd.NA
    # Define the baseline EDSS as the max EDSS observed at study start
    edss_t0 = np.max(edss.loc[time <= 0])
    # Define the events as the significantly higher scores
    events = is_significant(edss_t0, edss)
    if any(events):
        # Observed: Time to first progression event
        return (True, time.iloc[np.argmax(events)])
    else:
        # Censored: None of the future EDSS scores are significantly higher
        return (False, time.iloc[-1])

def is_significant(base, new):
    """
    Defines whether the EDSS difference between `base` and `new` is significant:
    - True if new-base >= 1.5 and 0 = base
    - True if new-base >= 1.0 and 0 < base < 6
    - True if new-base >= 0.5 and     base > 5.5
    - False otherwise

    @param base: Number indicating current EDSS score.
    @param new : Number or numpy array indicating some future EDSS score(s).
    @return: Boolean.
    """
    delta = new-base
    if base == 0:
        return delta >= 1.5
    elif base <= 5.5:
        return delta >= 1
    else:
        return delta >= 0.5

if __name__ == "__main__":
    # Unit test
    df_time = pd.Series([0,1,2,3,4,5,6,7,8])
    df_edss = pd.Series([3,3,1,1,2,3,2,3,3])
    expected_cens     = [0,0,1,1,1,0,1,0,0]
    expected_time     = [8,7,2,1,1,3,1,1,0]
    for i in range(len(df_time)-1):
        cens, time = __define_event(df_time.iloc[i:] - df_time.iloc[i], df_edss.iloc[i:])
        assert cens == expected_cens[i], f"{cens} <-> {expected_cens[i]}"
        assert time == expected_time[i], f"{time} <-> {expected_time[i]}"
