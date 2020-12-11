from covid_xprize.nixtamalai import models
from covid_xprize.nixtamalai import helpers
import pandas as pd
import numpy as np


def test_Features_fit():
    data = helpers.preprocess()
    m = models.Features().fit(data)
    assert isinstance(m, models.Features)


def test_Features_traning_set():
    data = helpers.get_OxCGRT()
    helpers.preprocess_npi(data)
    helpers.preprocess_newcases(data)
    m = models.Features().fit(data)
    X, y = m.training_set()
    assert isinstance(X, pd.DataFrame) and isinstance(y, np.ndarray)


def test_Features_transform():
    data = helpers.get_OxCGRT()
    helpers.preprocess_npi(data)
    helpers.preprocess_newcases(data)
    m = models.Features().fit(data)
    for X in m.transform(data):
        m.update_prediction(-10)
