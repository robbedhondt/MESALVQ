import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from .time_to_progression import define_progression_events
from .constants import PATH_DATA, EVENT_NAMES
from sklearn.model_selection import train_test_split
from sksurv.util import Surv
from .slvq import SkewTransformer

class Utils:
    @staticmethod
    def load_raw(fname):
        return pd.read_sas(os.path.join(PATH_DATA, fname+".xpt"), format="xport", encoding="utf-8")

    @staticmethod
    def load(fname):
        df = Utils.load_raw(fname)
        df = df.set_index("USUBJID")
        df = Utils.drop_constant_columns(df)
        df = Utils.drop_all_missing_columns(df)
        return df

    @staticmethod
    def drop_constant_columns(df):
        return df.loc[:, (df != df.iloc[0]).any()]
    
    @staticmethod
    def drop_all_missing_columns(df):
        return df.dropna(how="all", axis="columns")

    @staticmethod
    def subselect_pre_studystart(df, verbose=False):
        """
        Automatically subselects the given dataframe to the visits that are pre-
        study start. Priority goes to the xxDY variable, indicating the day the
        test was taken: if it is there, a threshold of xxDY <= 1 is taken for
        visits to include/exclude. If it's not there, we fill in blanks based
        on the same rule for other DY variables (e.g. VISITDY). Finally, for the
        remaining 
        """
        printv = lambda t: print(t) if verbose else ""
        printv(f"Start subselecting, {df.shape[0]} entries to fill")
        # Start off with a clean mask
        mask = pd.Series(pd.NA, index=df.index)
        # Identify the column referring to the day test was taken (xxDY)
        dfDY = df.columns[(df.columns.str.endswith("DY")) & (df.columns.str.len() == 4)]
        if len(dfDY) > 1:
            print(f"WARNING: unexpected, multiple xxDY columns: {df.columns[dfDY]}")
        if len(dfDY) >= 1:
            mask.loc[(df[dfDY] <= 1).all(axis=1)] = True
            mask.loc[(df[dfDY] >  1).all(axis=1)] = False
        printv(f"> Filled {(~mask.isna()).sum()} entries (using {dfDY})")
        # Use any other DY columns to fill this further (e.g., VISITDY - planned day of visit)
        allDY = df.columns[df.columns.str.endswith("DY")]
        mask.loc[mask.isna() & (df[allDY] <= 1).any(axis=1)] = True
        mask.loc[mask.isna() & (df[allDY] >  1).any(axis=1)] = False
        printv(f"> Filled {(~mask.isna()).sum()} entries (using {allDY})")
        # Use the string VISIT column to further enlarge the mask
        if "VISIT" in df.columns:
            mask = mask.fillna( df.VISIT.str.contains("SCREENING|BASELINE|DAY 1|PRIOR TO RANDOMIZATION") )
            printv(f"> Filled {(~mask.isna()).sum()} entries (using 'VISIT')")
        # Remaining visits do not have any information available to screen, so are dropped
        mask = mask.fillna(False)
        printv(f"> Filled {(~mask.isna()).sum()} entries (set remaining to False)")
        return df.loc[mask]
    
    @staticmethod
    def drop_notdone_or_too_old(df):
        # Drop the visits that did not complete successfully (?)
        statcol = df.columns[df.columns.str.endswith("STAT")].values.squeeze()
        df = df.loc[df[statcol] != "NOT DONE"]
        # Drop the visits that are too far in the past (over 6 months)
        dycol = df.columns[(df.columns.str.endswith("DY")) & (df.columns.str.len() == 4)].values.squeeze()
        df = df.loc[~(df[dycol] < -30*6)]
        return df
    
    @staticmethod
    def convert_to_scores(df):
        pass

def load_baseline_dm():
    """
    Load and clean the MSOAC data file of demographical characteristics.
    """
    dm = Utils.load("dm")
    dm = dm.drop(columns=["SUBJID","AGEU","ACTARM","ACTARMCD"])
    # More advanced reformatting
    def map_race(s):
        if   s == "WHITE": return True
        elif s == ""     : return pd.NA
        else             : return False
    def map_country_to_continent(s):
        if s in ["USA", "CAN", "MEX"]:
            return "North America"
        elif s in ["PER", "COL", "CHL"]:
            return "South America"
        elif s in ["NZL", "AUS"]:
            return "Australia"
        elif s in ["GEO", "ISR", "IND"]:
            return "Asia"
        elif s in ['POL', 'UKR', 'CZE', 'RUS', 'SRB', 'DEU', 'GBR', 'NLD', 'BGR', 
                'HUN', 'ROU', 'GRC', 'FRA', 'BEL', 'SWE', 'EST', 'ESP', 'CHE', 
                'HRV', 'TUR', 'LVA', 'FIN', 'IRL', 'DNK']:
            return "Europe"
        else:
            return pd.NA
    dm.SEX = dm.SEX.astype("category")
    dm[f"IS_{dm.SEX.cat.categories[1]}"] = dm.SEX.cat.codes
    dm["IS_HISPLAT"] = dm["ETHNIC"].map({"HISPANIC OR LATINO":True, 
        "NOT HISPANIC OR LATINO":False, "":pd.NA})
    dm["IS_WHITE"] = dm["RACE"].apply(map_race)
    dm["CONTINENT"] = dm["COUNTRY"].apply(map_country_to_continent)
    dm["LATITUDE"] = dm["COUNTRY"].map({ # Map country to average latitude, 
        # motivation: https://www.nature.com/articles/nrneurol.2016.181
        'USA': 38, 'POL': 52, 'CAN': 60, 'UKR': 49, 'CZE': 50, 'IND': 20, 'RUS': 60,
        'SRB': 44, 'DEU': 51, 'GBR': 54, 'NLD': 52, 'BGR': 43, 'HUN': 47, 'ROU': 46,
        'GRC': 39, 'FRA': 46, 'NZL': 41, 'BEL': 50, 'SWE': 62, 'MEX': 23, 'EST': 59,
        'ESP': 40, 'PER': 10, 'GEO': 42, 'AUS': 27, 'ISR': 31, 'CHE': 47, 'HRV': 45,
        'TUR': 39, 'COL':  4, 'LVA': 57, 'FIN': 64, 'IRL': 53, 'DNK': 56, 'CHL': 30,
    })
    dm = dm.drop(columns=["SEX","ETHNIC","RACE","COUNTRY"])

    # # Further filling based on supplementary information... TODO
    # # Actually ethnicity has no impact on MS progression so probably not worth
    # # to pursue further...
    # sdm = Utils.load("suppdm")
    # dm.loc[sdm.index[sdm.QVAL == "HISPANIC"], "IS_HISPLAT"] = True
    # ...
    return dm

def load_baseline_fa():
    fa = Utils.load("fa")
    # Merge the two types of evaluation intervals
    fa["FAINTERVAL"] = fa["FAEVLINT"] + fa["FAEVINTX"]
    # Pivot: 1 column per evaluation interval
    nrelap = pd.pivot(
        fa.loc[fa.FATESTCD == "NUMRLPS"].reset_index(), 
        values="FASTRESN", index="USUBJID", columns="FAINTERVAL"
        ).add_prefix("NUMRLPS ")
    acurelap = fa.loc[fa.FATESTCD == "ACUTRLPS"].FASTRESC.map({"Y":1, "N":0}).rename("ACUTRLPS")
    return pd.concat((nrelap, acurelap), axis=1)

def load_baseline_mh():
    mh = Utils.load("mh")
    smh = Utils.load("suppmh")

    # Extract disease duration
    mhdiag = mh.loc[(mh.MHDECOD == "DIAGNOSIS OF MULTIPLE SCLEROSIS") | 
                    (mh.MHCAT.isin(["PRIMARY DIAGNOSIS", "DIAGNOSIS"]))]
    mhdiag["IDVARVAL"] = mhdiag.MHSEQ.astype(int).astype(str)
    mhdiag = mhdiag.reset_index().set_index(["USUBJID","IDVARVAL"])
    smh  =  smh.reset_index().set_index(["USUBJID","IDVARVAL"])
    joint_index = mhdiag.index.intersection(smh.index)
    diag = smh.loc[joint_index]
    diag.QVAL = diag.QVAL.astype(float)
    diag.loc[diag.QNAM.isin(["STSTUDMO","STUDMO"]), "QVAL"] /= 12
    diag.loc[diag.QNAM == "YRSDIAG", "QVAL"] *= -1
    diag = diag.groupby("USUBJID").QVAL.min() # sometimes multiple entries e.g. when conversion to SPMS --> we just want the first one
    diag.loc[diag > 0] -= 1 # (see how timing is recorded in MSOAC)
    diag.name = "DISEASE DURATION"
    # # Weird distribution, a lot of values == 1...
    # sum(diag >= -1)
    # mhdiag.loc[joint_index].loc[diag.index[diag > 0]]
    # smh.loc[joint_index].loc[diag.index[diag > 0]]

    # Extract disease subtype and 4 key medical history factors
    search_terms = {
        "MS_COURSE_RR": ["RRMS","RELAPSING-REMITTING"],
        "MS_COURSE_PP": ["PPMS","PRIMARY-PROGRESSIVE","PROGRESSIVE RELAPSING","PRIMARY PROGRESSIVE"], # NOTE counting progressive relapsing as PPMS here as it's the most similar (and PRMS is rare)
        "MS_COURSE_SP": ["SPMS","SECONDARY-PROGRESSIVE","SECONDARY PROGRESSIVE"],
        "MH_CARDIO"   : ["CARDIOVASCULAR"],
        "MH_URINARY"  : ["BLADDER","URINARY","URINATION"],
        "MH_MUSCKELET": ["MUSCULOSKELETAL"],
        "MH_FATIGUE"  : ["FATIGUE"],
    }
    term_cols = [
        "MHTERM", "MHLLT", "MHDECOD", "MHCAT", "MHSCAT", "MHHLGT", "MHBODSYS", "MHSOC"
    ]
    terms_agg = mh[term_cols].agg('///'.join, axis=1)
    terms_agg = terms_agg.str.upper() # Make sure everything is capitalized
    found = {}
    for name, terms in search_terms.items():
        found[name] = terms_agg.str.contains("|".join(terms))
    found = pd.concat(found, axis=1)
    found = found.groupby("USUBJID").max() # aggregate over patients
    found.MS_COURSE_RR[found.MS_COURSE_SP] = False # If SPMS, then not RRMS anymore

    # Return concatenated results
    return pd.concat((diag, found), axis=1)

def load_baseline_ft():
    """
    Load and clean the functional tests taken before study start.
    """
    ft = Utils.load("ft")
    ft.VISITDY = ft.VISITDY.replace("", "NaN").astype(float)
    ft = Utils.subselect_pre_studystart(ft)
    ft = Utils.drop_notdone_or_too_old(ft)

    # Aggregate all functional test results from before study start
    def aggregate(fttest, ftscat=""):
        df = ft.loc[(ft.FTTEST == fttest) & (ft.FTSCAT == ftscat)]
        if all(df.FTSTRESN.isna()): # binary variable, remap and return max (since "Y" is usually uncommon here)
            return df.FTSTRESC.map({"N":0, "Y":1}).groupby("USUBJID").max()
        else: # continuous variable, return average
            return df.FTSTRESN.groupby("USUBJID").mean()
    scores = {}
    for test in ft.FTTEST.unique():
        for cat in ft.loc[ft.FTTEST == test, "FTSCAT"].unique():
            scores[f"{test} {cat}".strip()] = aggregate(test, cat)
    scores = pd.concat(scores, axis=1)
    return scores

def load_baseline_oe():
    oe = Utils.load("oe")
    oe = Utils.subselect_pre_studystart(oe)
    oe = Utils.drop_notdone_or_too_old(oe)

    # Convert visual acuity into categories (normal/abnormal)
    oe.loc[oe.OETESTCD == "VISACU"  , "OESTRESN"] = oe.loc[oe.OETESTCD == "VISACU"  , "OESTRESC"].astype("category").cat.codes
    # Drop low vision test (only 8 observations, nothing interesting)
    oe = oe.loc[oe.OETESTCD != "LOVISTST"]
    # Convert Snellen into a numerical score
    oe.loc[oe.OETESTCD == "SNELLCOR", "OESTRESN"] = oe.loc[oe.OETESTCD == "SNELLCOR", "OESTRESC"].apply(
        lambda frac: int(frac.split("/")[0]) / int(frac.split("/")[1])
    )
    # Normal Snellen needs some faulty entry fixes and remapping
    oe.loc[oe.OETESTCD == "SNELLEQ", "OESTRESN"] = oe.loc[oe.OETESTCD == "SNELLEQ", "OESTRESC"].map({
        # Assuming everything is in the 6/xx format, since we have no other info to
        # base this on... following snippet is helpful to identify the cats:
        # `oe.loc[oe.OETESTCD == "SNELLEQ", ["OESTRESC","OESTRESN"]].value_counts(dropna=False, sort=False)`
        "0.2"   : 9.5 , # This was probably a typo, very extreme score... assumed it was formatted as logMAR and converted accordingly, logMAR = np.log10(1/SES) see https://spie.org/publications/spie-publication-resources/optipedia-free-optics-information/fg04_p19-20_visual_acuity
        # "37.5"  : 37.5, # Very high but this person also has pretty bad scores on "number of letters correct" so seems consistent at least
        "6/12"  : 12.0 ,
        "6/15"  : 15.0 ,
        "6/4.8" : 4.8 ,
        "6/48"  : 4.8 , # Assumed "48" is typo since these people actually have good scores on "number of letters correct"
        "6/6"   : 6.0 ,
        "6/7.5" : 7.5 ,
        "6/9.6" : 9.6 ,
        "60"    : 6.0, # Assumed "60" is a type since good scores on "NUMLCOR"
        # NOTE: all these I didn't change, they seemed reasonable numbers for 6/xx
        # "12"    : 12.0,
        # "15"    : 15.0,
        # "19.2"  : 19.2,
        # "24"    : 24.0,
        # "30"    : 30.0,
        # "4.8"   : 4.8 ,
        # "6"     : 6.0 ,
        # "7.5"   : 7.5 ,
        # "9.6"   : 9.6 ,
    }).astype(float)
    # Combine different subcategorizations together for consistency
    oe.OESCAT = oe.OESCAT + " | " + oe.OELAT + " | " + oe.OEMETHOD

    # Aggregate all functional test results from before study start
    def aggregate(oetest, oescat=""):
        df = oe.loc[(oe.OETEST == oetest) & (oe.OESCAT == oescat)]
        if all(df.OESTRESN.isna()): # binary variable, remap and return max (since "Y" is usually uncommon here)
            # NOTE: for the one time we wanted to use max, we can also just take mean and then np.ceil it
            return df.OESTRESC.astype("category").cat.codes.groupby("USUBJID").mean()
        else: # continuous variable, return average
            return df.OESTRESN.groupby("USUBJID").mean()
    scores = {}
    for test in oe.OETEST.unique():
        for cat in oe.loc[oe.OETEST == test, "OESCAT"].unique():
            scores[f"{test} {cat}".strip()] = aggregate(test, cat)
    scores = pd.concat(scores, axis=1)
    return scores

def load_baseline_qs():
    """Processing is similar to `ft`, but no category subdivision + some custom preprocessing"""
    qs = Utils.load("qs")
    qs = Utils.subselect_pre_studystart(qs)
    qs = Utils.drop_notdone_or_too_old(qs)

    # Drop problematic tests
    qs = qs.loc[qs.QSTEST != "KFSS1-Weakness Interferes With Testing"] # not useful, only "CHECKED" or missing
    qs = qs.loc[qs.QSTEST != "KFSS1-Other Functions Specify"] # only measured for 28 patients. 45 measurements split over 11 categories, not translatable to numeric features
    # Clean the numeric outcome variable
    qs.loc[qs.QSSTRESN == 99, "QSSTRESN"] = np.nan # placeholder nan (only some values)
    qs.QSSTRESN = qs.QSSTRESN.round(10) # 0 was mapped to 5.397605e-79
    qs = qs.loc[~qs.QSSTRESN.isna()] # Obviously not useful if the test has no value

    # MERGING RAND36 AND SF12 INTO A SINGLE COHERENT MEASURE
    # Normalize RAND36 and SF12 questions to a [0,1] scale
    locproms = qs.QSTEST.str.startswith("R36") | qs.QSTEST.str.startswith("SF12")
    qs_proms = qs.loc[locproms].copy()
    to_reverse = [ # reverse the order for some questions such that higher=better
        # cfr https://www.rand.org/health-care/surveys_tools/mos/36-item-short-form/survey-instrument.html
        "R360101", "R360102", "R360120", "R360121", "R360122", "R360123", 
        "R360126", "R360127", "R360130", "R360134", "R360136",
        # cfr https://journals.lww.com/lww-medicalcare/_layouts/15/oaks.journals/ImageView.aspx?k=lww-medicalcare:1996:03000:00003&i=F2-3&year=1996&issue=03000&article=00003&type=Fulltext
        "SF12101", "SF12105", "SF12106A", "SF12106B"
    ]
    scale_minmax = lambda arr: (arr - arr.min()) / (arr.max() - arr.min())
    for testcd in qs_proms.QSTESTCD.unique():
        multiplier = [1,-1][testcd in to_reverse]
        qs_proms["QSSTRESN"] = scale_minmax(multiplier * qs_proms["QSSTRESN"])
    # Harmonize the categories: translate SF12 cat into RAND36 cat
    map_sf_to_rand = {
        # # The "health change" is unique to RAND36 and is not included in SF12
        # ''                    : 'HEALTH CHANGE'                             , 
        'GENERAL HEALTH'      : 'GENERAL HEALTH'                            ,
        'PHYSICAL FUNCTIONING': 'PHYSICAL FUNCTIONING'                      ,
        'MENTAL HEALTH'       : 'EMOTIONAL WELL-BEING'                      ,
        'ROLE PHYSICAL'       : 'ROLE LIMITATIONS DUE TO PHYSICAL HEALTH'   ,
        'VITALITY'            : 'ENERGY/FATIGUE'                            ,
        'ROLE EMOTIONAL'      : 'ROLE LIMITATIONS DUE TO EMOTIONAL PROBLEMS',
        'BODILY PAIN'         : 'PAIN'                                      ,
        'SOCIAL FUNCTIONING'  : 'SOCIAL FUNCTIONING'                        ,
    }
    qs_proms.QSSCAT = qs_proms.QSSCAT.replace(map_sf_to_rand)
    # Remap the questions to their category
    catmap = qs_proms[["QSTEST","QSSCAT"]].value_counts()
    assert catmap.index.get_level_values(0).nunique() == catmap.shape[0] # all questions should be unique
    catmap = {k:v for k,v in catmap.index}
    qs_proms["QSTEST"] = qs_proms["QSTEST"].replace(catmap)
    qs_proms["QSTEST"] = "PROMS R36+SF12 " + qs_proms["QSTEST"] # Prefix by a clear ID
    # Add the constructed PROMS df back into the original df
    qs = pd.concat((qs, qs_proms), axis=0)

    # Aggregate test results per patient
    # (currently mean although these variables are usually categorical,
    #  but most patients only have 1-2 baseline tests done anyway)
    scores = {}
    for test in qs.QSTEST.unique():
        scores[test] = qs.loc[qs.QSTEST == test, "QSSTRESN"].groupby("USUBJID").mean()
    scores = pd.concat(scores, axis=1)

    # Add indicator variable for merged PROMS variable
    loc_rand36 = ~scores.loc[:, scores.columns.str.startswith("R36")].isna().min(axis=1)
    loc_sf12   = ~scores.loc[:, scores.columns.str.startswith("SF12")].isna().min(axis=1)
    scores["PROMS questionnaire"] = np.nan              # neither
    scores.loc[loc_rand36, "PROMS questionnaire"] = 1.0 # RAND36
    scores.loc[loc_sf12  , "PROMS questionnaire"] = 0.0 # SF12
    return scores

def load_baseline_sc():
    sc = Utils.load("sc")
    sc = sc.groupby("USUBJID").SCSTRESC.apply(lambda t: t.mode().values[0]) # sometimes more than one mode, just take the first one
    sc = sc.astype("category").cat.codes
    sc = sc.rename("DOMINANT_HAND")
    return sc

def load_and_merge_baselines():
    df_list = [
        load_baseline_dm(),
        load_baseline_fa(),
        load_baseline_ft(),
        load_baseline_mh(),
        load_baseline_oe(),
        load_baseline_qs(),
        load_baseline_sc(),
    ]
    return pd.concat(df_list, axis=1).sort_index()

def clean_qs(qs=None, verbose=True):
    """
    Load and clean the MSOAC data file of questionnaires.
    """
    if qs is None:
        qs = Utils.load_raw("qs")
    # Convert the positive times to actual time since study start
    qs.loc[qs.QSDY >  0, "QSDY"] -= 1 # (1 day offset, see how timing is recorded in MSOAC)

    # Filling in missing dates based on other information
    def printv(text):
        if verbose:
            print(f"{qs.QSDY.isna().sum():6d} missing time stamps ({text})")
    printv("original")
    # (1) Use VISITDY (= Planned Study Day of Visit) to fill holes
    qs.QSDY = qs.QSDY.fillna(qs.VISITDY)
    printv("fill based on VISITDY (planned study day of visit)")
    # (2) Fill holes when VISIT == screening or baseline
    baseline = pd.Series(np.nan, index=qs.index)
    # baseline.loc[qs.VISIT.str.contains("BASELINE" )] =   0.0
    # baseline.loc[qs.VISIT.str.contains("SCREENING")] = -30.0 #+ np.random.uniform(-15, 15, baseline.shape)
    for keyword in ["BASELINE", "SCREENING", "DAY 1", "PRIOR TO RANDOMIZATION"]:
        loc = qs.VISIT.str.contains(keyword)
        baseline.loc[loc] = qs.QSDY.loc[loc].mean() # fill with the mean of the values that ARE observed
    qs.QSDY = qs.QSDY.fillna(baseline) # This skips the already observed QSDY
    printv("fill based on VISIT (BASELINE / SCREENING / DAY 1 / PRIOR TO RAND)")
    # (3) Propagate "M(ON)TH %d" in the VISIT column 
    #     to a value of %d / 12 in the QSDY column (with some added noise)
    month = qs.VISIT.str.extract(r"M(?:ON)?TH (\d{1,2})")[0].astype(float) 
    # month += np.random.uniform(-10/30, 10/30, month.shape)
    # month /= 12 # Convert month to year
    # month += 1/6*np.random.standard_normal(month.shape) # 99% CI is [-3,3] / 6
    month *= 30.436875 # Convert month to days
    # Add random noise to make it non-unique
    # month += 5*np.random.standard_normal(month.shape) # 99% CI is [-15,15] days
    qs.QSDY = qs.QSDY.fillna(month) # Fill missing values with imputed
    printv("fill based on VISIT (keyword 'm[on]th')")
    
    qs = qs.sort_values(by=["USUBJID", "QSTEST", "QSDY"]).reset_index(drop=True)
    return qs

def clean_kfss_and_edss(qs=None, dropna_qsdy=True, drop_inconsistent=True, sub_columns=True, verbose=True):
    """
    Load and clean the KFSS and EDSS test results.
    """
    if qs is None:
        qs = clean_qs()
    def printv(text):
        if verbose:
            print(f"{qs.shape[0]:6d} visits x {qs.shape[1]:2d} features " +
                  f"for {qs.USUBJID.nunique():4d} patients ({text})")
    printv(f"=== start processing qs ===")
    # Drop other questionnaire tests (e.g. RAND-36, SF-12)
    qs = qs.loc[qs.QSCAT.isin(["KFSS","EDSS"])]
    printv("subselect EDSS & KFSS categories")
    # Limit to the big categories without free-text fields
    qs = qs.loc[qs.QSTEST.isin(EVENT_NAMES)]
    printv("drop small free-text categories")
    # Fix data entry mistakes (117 times empty string, 2 times "99", 1 time "3.6")
    qs.QSSTRESC = qs.QSSTRESC.replace({"":"NaN", "3.6":"3.5", "99":"NaN"})
    # Convert the actual score to floats
    qs.QSSTRESC = qs.QSSTRESC.astype(float)
    # Drop visits without measurement
    qs = qs.loc[qs.QSSTAT != "NOT DONE"] # test is not performed (n = 212)
    printv("drop visits marked as 'NOT DONE'")
    qs = qs.loc[~qs.QSSTRESC.isna()] # value is still missing (n = 2)
    printv("drop visits with missing test result")
    # Drop constant columns and columns with only missing values
    qs = Utils.drop_constant_columns(qs)
    printv("drop constant columns")
    qs = Utils.drop_all_missing_columns(qs)
    printv("drop columns with only missing values")
    # Drop when date is missing (n = 13,677 or 6.4% of the visits)
    if dropna_qsdy:
        qs = qs.dropna(subset="QSDY")
        printv("drop visits where date is missing")
    # Drop duplicated rows (now that we have no more missing values in QSDY)
    qs = qs.drop_duplicates(subset=["USUBJID", "QSTEST", "QSSTRESC", "QSDY"])
    printv("drop duplicated rows")
    # Duplicate time points that remain have a different test result ==> drop both
    if drop_inconsistent:
        qs = qs.drop_duplicates(subset=["USUBJID", "QSTEST", "QSDY"], keep=False)
        printv("drop matching rows (same day and patient) with inconsistent test result")
    # Subselect some columns of interest
    if sub_columns:
        qs = qs[["USUBJID", "QSTEST", "QSSTRESC", "QSDY"]]
        printv("subselect columns of interest")
    printv(f"=== finished processing qs ===")
    return qs

def build_time_to_worsening(qs=None, verbose=True, kfss_always_significant=True):
    if qs is None:
        qs = clean_kfss_and_edss(verbose=verbose)
    if kfss_always_significant:
        # NOTE: small hack to ensure that KFSS differences are always significant
        for test in qs.QSTEST.unique():
            if "KFSS" in test:
                qs.loc[qs.QSTEST == test, "QSSTRESC"] *= 10
    # Construct the events per test
    cens = pd.DataFrame(index=qs.USUBJID.unique(), columns=EVENT_NAMES, dtype="boolean")
    time = pd.DataFrame(index=qs.USUBJID.unique(), columns=EVENT_NAMES, dtype=float)
    for test in qs.QSTEST.unique():
        df_surv = define_progression_events(qs.loc[qs.QSTEST == test], edss="QSSTRESC")
        cens.loc[df_surv.index, test] = df_surv["cens"]
        time.loc[df_surv.index, test] = df_surv["time"]
    # Drop some rows based on event existence
    cens = cens.dropna(axis="rows", how="all")
    time = time.dropna(axis="rows", how="all")
    assert cens.shape[0] == time.shape[0]
    if verbose: print(f"{qs.USUBJID.nunique()} --> {cens.shape[0]} patients " + 
            "(drop patients with insufficient data to define any event)")
    cens = cens.dropna(axis="rows", how="any")
    time = time.dropna(axis="rows", how="any")
    assert cens.shape[0] == time.shape[0]
    if verbose: print(f"     --> {cens.shape[0]} patients " + 
            "(drop patients with insufficient data to define all 8 events)")
    return cens, time

def load_kfss_and_edss():
    fpath = os.path.join(PATH_DATA, "processed", "qs_kfss_edss.csv")
    if not os.path.exists(fpath):
        qs = clean_kfss_and_edss()
        qs.to_csv(fpath)
    qs = pd.read_csv(fpath, index_col=0) # to make sure return value is always consistent!
    return qs

def load_time_to_worsening(**kwargs):
    fpath_cens = os.path.join(PATH_DATA, "processed", "cens.csv")
    fpath_time = os.path.join(PATH_DATA, "processed", "time.csv")
    if not (os.path.exists(fpath_cens) and os.path.exists(fpath_time)):
        cens, time = build_time_to_worsening(**kwargs)
        cens.to_csv(fpath_cens)
        time.to_csv(fpath_time)
    cens = pd.read_csv(fpath_cens, index_col=0) # to make sure return is always consistent!
    time = pd.read_csv(fpath_time, index_col=0) # to make sure return is always consistent!
    return cens, time

def load_input_data():
    fpath = os.path.join(PATH_DATA, "processed", "input.csv")
    if not os.path.exists(fpath):
        X = load_and_merge_baselines()
        X.to_csv(fpath)
    X = pd.read_csv(fpath, index_col=0) # to make sure return value is always consistent!
    return X

def load_xy(thresh_dropna_cols=0.5, normalize=True):
    cens, time = load_time_to_worsening()
    X = load_input_data()
    X = X.dropna(axis="columns", thresh=thresh_dropna_cols*X.shape[0])
    X = X.loc[cens.index]
    if normalize:
        X = pd.DataFrame(
            SkewTransformer().fit_transform(X), index=X.index, columns=X.columns
        )
    y = []
    for key in cens.columns:
        y.append(Surv().from_arrays(cens[key], time[key]))
    y = np.vstack(y).T
    return X, y

def multi_surv_from_arrays(cens, time):
    y = []
    for j in range(cens.shape[1]):
        y.append(Surv().from_arrays(np.array(cens)[:,j], np.array(time)[:,j]))
    y = np.vstack(y).T
    return y

def split_train_test(X, y):
    # Split into train-test, stratified by having any of the events at all 
    # (true for 82% of the patients)
    X_train, X_test, y_train, y_test = train_test_split(
        np.array(X).astype(float), y, test_size=0.2, 
        random_state=42, stratify=np.max(y["event"], axis=1)
    )
    return X_train, X_test, y_train, y_test

def print_patient_characteristics():
    cens, time = load_time_to_worsening()
    X = load_input_data()
    X = X.dropna(axis="columns", thresh=0.5*X.shape[0])
    X = X.loc[cens.index]

    numvar = lambda f: f"{f.mean():.2f} $\pm$ {f.std():.2f}"
    binvar = lambda f, cats: f"{100*f.mean():.2f}\% {cats[0]}, {100*(1-f.mean()):.2f}\% {cats[1]}"

    print(f"Age & {numvar(X.AGE)} years \\\\")
    print(f"Sex & {binvar(X.IS_M, ['male','female'])} \\\\")
    print(f"Race & {binvar(X.IS_WHITE, ['white','other (aggregated)'])} \\\\")
    # vc = X["NUMRLPS -P1Y"].value_counts().sort_index()
    # vc = ", ".join([f"{vc.index[i]:.0f}:{vc.values[i]}" for i in range(len(vc))])
    vc = numvar(X["NUMRLPS -P1Y"])
    print(f"\#relapses past year & {vc} \\\\")
    # ================
    # FUNCTIONAL TESTS
    f = X["NHPT01-Time to Complete 9-Hole Peg Test NON-DOMINANT HAND"]
    print(f"9HPT - time - non-dominant & {numvar(f)} seconds \\\\")
    f = X["NHPT01-Time to Complete 9-Hole Peg Test DOMINANT HAND"]
    print(f"9HPT - time - dominant & {numvar(f)} seconds \\\\")
    f = X["NHPT01-More Than Two Attempts NON-DOMINANT HAND"]
    print(f"9HPT - >2 - non-dominant & {binvar(f, ['yes','no'])} \\\\")
    f = X["NHPT01-More Than Two Attempts DOMINANT HAND"]
    print(f"9HPT - >2 - dominant & {binvar(f, ['yes','no'])} \\\\")
    f = X["T25FW1-Time to Complete 25-Foot Walk"]
    print(f"T25FW - time & {numvar(f)} seconds \\\\")
    f = X["T25FW1-More Than Two Attempts"]
    print(f"T25FW - >2 & {binvar(f, ['yes','no'])} \\\\")
    f = X["PASAT1-Total Correct 3 SECONDS"]
    print(f"PASAT - total - 3s & {numvar(f)} \\\\")
    # ================
    # 
    print(f"Disease duration & {numvar(X['DISEASE DURATION'])} \\\\")
    vc = X.loc[:, X.columns.str.startswith("MS_COURSE")] # RR, PP, SP
    vc = vc.idxmax(axis=1).apply(lambda s: s[-2:]).value_counts()
    vc = ", ".join([f"{vc.index[i]}:{vc.values[i]}" for i in range(len(vc))])
    print(f"MS course & {vc} \\\\")
    print(f"History: cardiovascular & {binvar(X['MH_CARDIO'], ['yes','no'])} \\\\")
    print(f"History: urinary & {binvar(X['MH_URINARY'], ['yes','no'])} \\\\")
    print(f"History: musculoskeletal & {binvar(X['MH_MUSCKELET'], ['yes','no'])} \\\\")
    print(f"History: fatigue & {binvar(X['MH_FATIGUE'], ['yes','no'])} \\\\")
    # ================
    # QUESTIONNAIRES
    print(f"EDSS & {numvar(X['EDSS01-Expanded Disability Score'])} \\\\")
    print(f"KFSS pyramidal & {numvar(X['KFSS1-Pyramidal Functions'])} \\\\")
    print(f"KFSS cerebellar & {numvar(X['KFSS1-Cerebellar Functions'])} \\\\")
    print(f"KFSS brainstem & {numvar(X['KFSS1-Brain Stem Functions'])} \\\\")
    print(f"KFSS sensory & {numvar(X['KFSS1-Sensory Functions'])} \\\\")
    print(f"KFSS bowel and bladder & {numvar(X['KFSS1-Bowel and Bladder Functions'])} \\\\")
    print(f"KFSS visual & {numvar(X['KFSS1-Visual or Optic Functions'])} \\\\")
    print(f"KFSS cerebral & {numvar(X['KFSS1-Cerebral or Mental Functions'])} \\\\")
    print(f"PROMS emotional well-being & {numvar(X['PROMS R36+SF12 EMOTIONAL WELL-BEING'])} \\\\")
    print(f"PROMS energy / fatigue & {numvar(X['PROMS R36+SF12 ENERGY/FATIGUE'])} \\\\")
    print(f"PROMS general health & {numvar(X['PROMS R36+SF12 GENERAL HEALTH'])} \\\\")
    print(f"PROMS pain & {numvar(X['PROMS R36+SF12 PAIN'])} \\\\")
    print(f"PROMS physical functioning & {numvar(X['PROMS R36+SF12 PHYSICAL FUNCTIONING'])} \\\\")
    print(f"PROMS role limitations emotional & {numvar(X['PROMS R36+SF12 ROLE LIMITATIONS DUE TO EMOTIONAL PROBLEMS'])} \\\\")
    print(f"PROMS role limitations physical & {numvar(X['PROMS R36+SF12 ROLE LIMITATIONS DUE TO PHYSICAL HEALTH'])} \\\\")
    print(f"PROMS social functioning & {numvar(X['PROMS R36+SF12 SOCIAL FUNCTIONING'])} \\\\")
    print(f"Dominant hand & {binvar(X['DOMINANT_HAND'], ['right','left'])} \\\\")

if __name__ == "__main__":
    load_kfss_and_edss()
    load_time_to_worsening()
    load_input_data()
