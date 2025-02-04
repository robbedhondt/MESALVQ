import os
import numpy as np
import matplotlib.pyplot as plt

# Where is the MSOAC data located?
PATH_DATA = os.path.join(os.path.dirname(__file__), "..", "data") 
# Where to store figures?
PATH_FIGS = os.path.join(os.path.dirname(__file__), "..", "figures")

# Figure parameters
FIGWIDTH = 4.8041 # latex \textwidth or \linewidth in inches
COLORS = plt.get_cmap("rocket")(np.linspace(0.1,0.9, 4)) # 4 tableau colors
LINESTYLES = ["solid",(0,(1,1)),"dashed","dashdot", # 4 linestyles
        (0,(3,3,1,3,1,3)), # 1 additional linestyle for local Kaplan-Meier
        # (0, (5,10)), 
        ] 

# Which targets do we consider for prediction?
EVENT_NAMES = [
    "EDSS01-Expanded Disability Score",
    "KFSS1-Sensory Functions",
    "KFSS1-Brain Stem Functions",
    "KFSS1-Bowel and Bladder Functions",
    "KFSS1-Pyramidal Functions",
    "KFSS1-Cerebral or Mental Functions",
    "KFSS1-Visual or Optic Functions",
    "KFSS1-Cerebellar Functions",
    # "KFSS1-Other Functions",
]

# For plots
TARGET_NAME_MAPPING = {
    "EDSS01-Expanded Disability Score"  : "EDSS",
    "KFSS1-Sensory Functions"           : "SENS",
    "KFSS1-Brain Stem Functions"        : "BRAI",
    "KFSS1-Bowel and Bladder Functions" : "SPHI",
    "KFSS1-Pyramidal Functions"         : "PYRA",
    "KFSS1-Cerebral or Mental Functions": "CBRA",
    "KFSS1-Visual or Optic Functions"   : "VISU",
    "KFSS1-Cerebellar Functions"        : "CBEL",
}
FEATURE_NAME_MAPPING = {
    'AGE': "Age", 
    'IS_M': "Sex (is male?)", 
    'IS_WHITE': "Race (is white?)", 
    'NUMRLPS -P1Y': "#relapses past year",
    'NHPT01-Time to Complete 9-Hole Peg Test NON-DOMINANT HAND': "9HPT - time - non-dominant",
    'NHPT01-Time to Complete 9-Hole Peg Test DOMINANT HAND': "9HPT - time - dominant",
    'T25FW1-Time to Complete 25-Foot Walk': "T25FW - time",
    'NHPT01-More Than Two Attempts DOMINANT HAND': "9HPT - >2 - non-dominant",
    'NHPT01-More Than Two Attempts NON-DOMINANT HAND': "9HPT - >2 - dominant",
    'PASAT1-Total Correct 3 SECONDS': "PASAT - total - 3s", 
    'T25FW1-More Than Two Attempts': "T25FW - >2",
    'DISEASE DURATION': "Disease duration", 
    'MS_COURSE_RR': "MS course (is RR?)", 
    'MS_COURSE_PP': "MS course (is PP?)", 
    'MS_COURSE_SP': "MS course (is SP?)",
    'MH_CARDIO': "History: cardiovascular", 
    'MH_URINARY': "History: urinary", 
    'MH_MUSCKELET': "History: musculoskeletal", 
    'MH_FATIGUE': "History: fatigue",
    'EDSS01-Expanded Disability Score': "EDSS", 
    'KFSS1-Pyramidal Functions': "KFSS pyramidal",
    'KFSS1-Cerebellar Functions': "KFSS cerebellar", 
    'KFSS1-Brain Stem Functions': "KFSS brainstem",
    'KFSS1-Sensory Functions': "KFSS sensory", 
    'KFSS1-Bowel and Bladder Functions': "KFSS bowel and bladder",
    'KFSS1-Visual or Optic Functions': "KFSS visual", 
    'KFSS1-Cerebral or Mental Functions': "KFSS cerebral",
    'PROMS R36+SF12 EMOTIONAL WELL-BEING': "PROMS emotional well-being", 
    'PROMS R36+SF12 ENERGY/FATIGUE': "PROMS energy / fatigue",
    'PROMS R36+SF12 GENERAL HEALTH': "PROMS general health", 
    'PROMS R36+SF12 PAIN': "PROMS pain",
    'PROMS R36+SF12 PHYSICAL FUNCTIONING': "PROMS physical functioning",
    'PROMS R36+SF12 ROLE LIMITATIONS DUE TO EMOTIONAL PROBLEMS': "PROMS role limitations emotional",
    'PROMS R36+SF12 ROLE LIMITATIONS DUE TO PHYSICAL HEALTH': "PROMS role limitations physical",
    'PROMS R36+SF12 SOCIAL FUNCTIONING': "PROMS social functioning", 
    'DOMINANT_HAND': "Dominant hand (is right?)",
}
