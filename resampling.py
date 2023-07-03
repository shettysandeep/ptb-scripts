"""
Code snippet for stratified over/under-sampling
# Over & under sampling stratified by State
# Simple set up

"""
# Install imbalanced-learn package
# `conda install -c conda-forge imbalanced-learn`

import pandas as pd
import imblearn  # you should have this installed
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

ptb = pd.read_csv("sample_dataset.csv")
state_group = ptb.groupby("state_code")

ros = RandomOverSampler(random_state=0)
rus = RandomOverSampler(random_state=0)


def resample_fun(grp_obj, resamplr):  # resample = rus or ros
    y = grp_obj["pre_term_birth_ind"]
    X = grp_obj
    X_resampled, y_resampled = resamplr.fit_resample(X, y)
    dat_tab = X_resampled.merge(y_resampled)
    return dat_tab


dat_tab = state_group.apply(resample_fun, ros)
