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


def test_AR():
    from microtc.utils import save_model
    data = helpers.get_OxCGRT()
    helpers.preprocess_npi(data)
    helpers.preprocess_newcases(data)
    m = models.Features().fit(data)
    X, y = m.training_set()
    ar = models.AR().fit(X, y)
    save_model([m, ar], "AR.model")
    for X in m.transform(data):
        hy = ar.predict(X)[0]
        assert np.isfinite(hy)
        m.update_prediction(hy)


def test_Lars():
    from microtc.utils import save_model
    data = helpers.get_OxCGRT()
    helpers.preprocess_npi(data)
    helpers.preprocess_newcases(data)
    m = models.Features().fit(data)
    X, y = m.training_set()
    ar = models.Lars().fit(X, y)
    save_model([m, ar], "Lars.model")