import pandas as pd
import numpy as np
from .helpers import NPI_COLS, ID_COLS, CASES_COL
from .helpers import preprocess_npi
from collections import OrderedDict
from typing import Union


class Features(object):
    """Transform DataFrame into features to used in the 
    regressors"""

    def __init__(self, lags: int=30, date: str="2020-11-12"):
        self._lags = lags
        self._data = None
        self._stop_training = np.datetime64(date)

    def update_prediction(self, hy: float) -> None:
        self._last_hy = hy

    @property
    def lags(self):
        return self._lags

    def get_data(self, exo, date, output, key, lag, y=True):
        lags = self._lags
        f = OrderedDict(dict(GeoID=key))
        f.update(Date=date[lag-1])
        #TODO: pegar datos estáticos para el país key (las predicciones queden al final)
        f.update([("e%s" % i, v) for i, v in enumerate(exo[lag - lags:lag].flatten())])
        f.update([("l%s" % i, v) for i, v in enumerate(output[lag - lags:lag].flatten())])
        if y:
            f.update(dict(y=output[lag]))
        return f

    def fit(self, data: pd.DataFrame) -> "Features":
        data = data.loc[data.Date <= self._stop_training, ["Date", "GeoID"] + NPI_COLS + CASES_COL]
        lags = self._lags
        D = []
        for key, value in data.groupby("GeoID"):
            exo = value.loc[:, NPI_COLS].to_numpy()
            date = value.loc[:, "Date"].to_numpy()
            output = value.loc[:, CASES_COL[0]].to_numpy()
            if exo.shape[0] <= lags:
                continue
            for lag in range(lags, exo.shape[0]):
                D.append(self.get_data(exo, date, output, key, lag))
            D.append(self.get_data(exo, date, output, key, lag + 1, y=False))
        self._data = pd.DataFrame(D)
        return self

    def training_set(self):
        #TODO: convertir a tasa * 100,000
        data = self._data.dropna()
        y = data.loc[:, "y"].to_numpy()
        data.drop(columns=["Date", "y"], inplace=True)
        return data, y

    def transform(self, data: pd.DataFrame, start: str="2020-11-13", end: str="2020-12-05"):
        start = np.datetime64(start)
        end = np.datetime64(end)
        data = data[(data.Date >= start) & (data.Date <= end)]
        cnt = (end - start).astype(int)
        max_date = self._data.Date.max()
        if start > max_date:
            start = max_date
        for key, value in self._data.groupby("GeoID"):
            X = value.loc[value.Date == start]
            _ = X.drop(columns=["Date", "y"])
            yield _
            columns = _.columns
            _ = _.to_numpy()
            X = _[0, 1:-self.lags]
            output = _[0, -self.lags:].tolist()
            X.shape = (self.lags, int(X.shape[0] / self.lags))
            X = X.tolist()
            gips_df = data.loc[data.GeoID == key]
            gips_np = gips_df.loc[:, NPI_COLS].to_numpy()
            for i in range(cnt):
                X.append(gips_np[i].tolist())
                del X[0]
                output.append(self._last_hy)
                del output[0]
                _ = np.concatenate(([key], np.array(X).flatten(), output))
                #TODO: pegar datos estáticos para el país key (las predicciones queden al final)
                yield pd.DataFrame([_], columns=columns)


class AR(object):
    def __init__(self):
        from sklearn.linear_model import LinearRegression
        self._model = LinearRegression()

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: np.ndarray) -> "AR":
        if isinstance(X, pd.DataFrame):
            X = X.drop(columns="GeoID").to_numpy()
        self._model.fit(X, y)
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            X = X.drop(columns="GeoID").to_numpy()
        hy = self._model.predict(X)
        hy[hy < 0 ] = 0
        hy[~ np.isfinite(hy)] = np.exp(12)
        return hy

    def decision_function(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        return self.predict(X)


class Lars(AR):
    def __init__(self):
        from sklearn.linear_model import LarsCV
        self._model = LarsCV()


class SVR(AR):
    def __init__(self):
        from sklearn.svm import SVR
        self._model = SVR()


class Lasso(AR):
    def __init__(self):
        from sklearn.linear_model import Lasso
        self._model = Lasso(alpha=0.1,
                            precompute=True,
                            max_iter=10000,
                            positive=True,
                            selection='random')    


class Identity(object):
    def fit(self, X):
        return self

    def transform(self, X):
        return X.drop(columns="GeoID").to_numpy()


