from covid_xprize.nixtamalai import models
from covid_xprize.nixtamalai import helpers
import pandas as pd
import numpy as np


def test_Features_fit():
    data = helpers.preprocess_full()
    m = models.Features().fit(data)
    assert isinstance(m, models.Features)


def test_Features_traning_set():
    data = helpers.preprocess_full()
    m = models.Features().fit(data)
    X, y = m.training_set()
    assert isinstance(X, pd.DataFrame) and isinstance(y, np.ndarray)


def test_Features_transform():
    data = helpers.preprocess_full()
    m = models.Features().fit(data)
    for X in m.transform(data):
        m.update_prediction(-10)


def test_AR():
    from microtc.utils import save_model
    data = helpers.preprocess_full()
    m = models.Features().fit(data)
    X, y = m.training_set()
    ar = models.AR().fit(X, y)
    save_model([m, ar], "AR.model")
    for X in m.transform(data):
        hy = m.update_prediction(ar.predict(X))
        assert np.isfinite(hy)


def test_Lars():
    from microtc.utils import save_model
    data = helpers.preprocess_full()
    m = models.Features().fit(data)
    X, y = m.training_set()
    ar = models.Lars().fit(X, y)
    save_model([m, ar], "Lars.model")


def test_Lasso():
    from microtc.utils import save_model
    data = helpers.get_OxCGRT()
    data = (data.pipe(helpers.preprocess_npi)
                .pipe(helpers.preprocess_newcases)
    )    
    m = models.Features().fit(data)
    X, y = m.training_set()
    ar = models.Lasso().fit(X, y)
    save_model([m, ar], "Lasso.model")


def test_evomsa():
    from microtc.utils import save_model
    from EvoMSA import base
    data = helpers.preprocess_full()
    m = models.Features().fit(data)
    X, y = m.training_set()
    evo = base.EvoMSA(TR=False, stacked_method=models.AR,
                      classifier=False,
                      models=[[models.Identity, models.AR],
                              [models.Identity, models.Lars],
                              # [models.Identity, models.SVR],
                              [models.Identity, models.Lasso]]).fit(X, y)
    save_model([m, evo], "evomsa.model")


def test_transform_population():
    data = helpers.preprocess_full()
    m = models.FeaturesN().fit(data)
    X, y = m.training_set()
    for X in m.transform(data):
        hy = m.update_prediction(np.array([10]))
        assert hy != 10
        break


def test_AR_N():
    from microtc.utils import save_model
    data = helpers.preprocess_full()
    m = models.FeaturesN().fit(data)
    X, y = m.training_set()
    ar = models.AR().fit(X, y)
    save_model([m, ar], "ARN.model")



def test_evomsaN():
    from microtc.utils import save_model
    from EvoMSA import base
    data = helpers.preprocess_full()
    m = models.FeaturesN().fit(data)
    X, y = m.training_set()
    evo = base.EvoMSA(TR=False, stacked_method=models.AR,
                      classifier=False,
                      models=[[models.Identity, models.AR],
                              [models.Identity, models.Lars],
                              [models.Identity, models.Lasso]]).fit(X, y)
    save_model([m, evo], "evomsaN.model")


def test_kmeans():
    from microtc.utils import save_model
    from EvoMSA import base
    data = helpers.preprocess_full()
    m = models.FeaturesN().fit(data)
    X, y = m.training_set()
    evo = base.EvoMSA(TR=False, stacked_method=models.AR,
                      classifier=False,
                      models=[[models.KMeans, models.ARG],
                              [models.KMeans, models.LarsG],
                              [models.KMeans, models.LassoG]]).fit(X, y)
    save_model([m, evo], "kmeans.model")


def test_transform_gap():
    from collections import defaultdict
    data = helpers.preprocess_full()
    m = models.FeaturesN().fit(data)
    X, y = m.training_set()
    output = defaultdict(list)
    for X in m.transform(data, start="2020-12-04", end="2020-12-05"):
        hy = m.update_prediction(np.array([10]))
        output[X.iloc[0]["GeoID"]].append(hy)

    for key, value in output.items():
        assert len(value) > 2
