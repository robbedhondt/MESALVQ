# Since survivallvq itself is not a module, we add the folder to the Python
# path
import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "survivallvq"))

# Now import the 2 functions of interest
from SkewTransformer import SkewTransformer
from Models.SurvivalLVQ import SurvivalLVQ
