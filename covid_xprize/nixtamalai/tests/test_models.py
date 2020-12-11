from covid_xprize.nixtamalai import models
from covid_xprize.nixtamalai import helpers
import pandas as pd
import numpy as np


def test_AR():
    data = helpers.preprocess(k=0)
    models.AR()


def test_Features_fit():
    data = helpers.preprocess(k=0)
    m = models.Features().fit(data)
    assert isinstance(m, models.Features)


def test_Features_traning_set():
    data = helpers.preprocess(k=0)
    m = models.Features().fit(data)
    X, y = m.training_set()
    assert isinstance(X, pd.DataFrame) and isinstance(y, np.ndarray) 