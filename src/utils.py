
import os
import random
import numpy as np
import tensorflow as tf
from statsmodels.stats.diagnostic import acorr_ljungbox

def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.config.experimental.enable_op_determinism()

def ljung_box_summary(data, title, lags=52):
    lb_results = acorr_ljungbox(data, lags=lags, return_df=True)
    print(f"Ljung-Box test for {title}\n{lb_results}")
