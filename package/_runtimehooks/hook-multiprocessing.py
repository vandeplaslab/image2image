"""multi-processing hooks"""

import os
import sys
from multiprocessing import freeze_support, set_start_method

os.environ["JOBLIB_MULTIPROCESSING"] = "0"

freeze_support()
if sys.platform == "darwin":
    os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
    set_start_method("spawn", True)
