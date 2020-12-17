import pandas as pd
import numpy as np
from .helpers import NPI_COLS, ID_COLS, CASES_COL, STATIC_COLS
from .helpers import preprocess_npi
from collections import OrderedDict
from typing import Union


class Features(object):
    """Transform DataFrame into features to used in the 
    regressors"""

    def __init__(self, lags: int=30, date: str="2020-11-12", static_cols: list=STATIC_COLS):
        self._lags = lags
        self._data = None
        self._stop_training = np.datetime64(date)
        self._exo_cols = None
        self._lag_cols = None
        self._rename_static = None
        self.static_cols = static_cols

    def update_prediction(self, hy: np.ndarray) -> float:
        hy[hy < 0 ] = 0
        hy[~ np.isfinite(hy)] = np.exp(12)
        hy = hy[0]
        self._last_hy = hy
        return hy

    @property
    def lags(self):
        return self._lags

    @property
    def static_cols(self):
        return self._static_cols

    @static_cols.setter
    def static_cols(self, s):
        if not set(s).issubset(set(STATIC_COLS)): 
            raise Exception("Static Columns must be a subset of STATIC_COLS")
        self._static_cols = s

    def get_data(self, exo, date, output, key, lag, y=True):
        lags = self._lags
        f = OrderedDict(dict(GeoID=key))
        f.update(Date=date[lag-1])
        f.update([("e%s" % i, v) for i, v in enumerate(exo[lag - lags:lag].flatten())])
        f.update([("l%s" % i, v) for i, v in enumerate(output[lag - lags:lag].flatten())])
        if y:
            f.update(dict(y=output[lag]))
        return f

    def fit(self, data: pd.DataFrame) -> "Features":
        data = data.loc[data.Date <= self._stop_training, 
                       ["Date", "GeoID"] + NPI_COLS + CASES_COL + self._static_cols]
        static = (data
                  .loc[:, ["Date", "GeoID"] + self._static_cols]
                  .groupby('GeoID')
                  .first()
                  .reset_index()
                  .drop('Date', axis=1))
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
        # merge static data here
        d = pd.DataFrame(D)
        self._lag_cols = [c for c in d.columns if c.startswith('l')]
        self._exo_cols = [c for c in d.columns if c.startswith('e')]
        self._rename_static ={c: "s%s" % i for i,c in enumerate(self._static_cols)}
        d = (d.merge(static, on="GeoID", how='inner')
             .reindex(['GeoID', 'Date'] + self._static_cols + 
                      self._exo_cols + self._lag_cols +['y'], axis=1)
             .rename(columns=self._rename_static)
             )
        self._data = d
        return self

    def training_set(self):
        data = self._data.dropna()
        y = data.loc[:, "y"].to_numpy()
        data.drop(columns=["Date", "y"], inplace=True)
        return data, y

    def transform(self, data: pd.DataFrame, start: str="2020-11-13", end: str="2020-12-05"):
        start = np.datetime64(start)
        end = np.datetime64(end)
        max_date = self._data.Date.max()
        if start > max_date:
            start = max_date
        # start es el último día del entrenamiento
        data = data[(data.Date > start) & (data.Date <= end)]
        static = (data
                .loc[:, ["Date", "GeoID"] + self._static_cols]
                .groupby('GeoID')
                .first()
                .reset_index()
                .drop('Date', axis=1))
        exo_cols = [c for c in self._data.columns if c.startswith('e')]
        lag_cols = [c for c in self._data.columns if c.startswith('l')]
        exo_lags = self._data[["GeoID", "Date"] + exo_cols + lag_cols]
        for key in exo_lags.GeoID.unique():
            self._last_key = key
            X = exo_lags.loc[(self._data.Date == start) & (self._data.GeoID == key)]
            _ = X.drop(columns=["Date"])
            d = (_.merge(static, on="GeoID")
                .reindex(['GeoID'] + self._static_cols + 
                        self._exo_cols + self._lag_cols, axis=1)
                .rename(columns=self._rename_static)
                )
            yield d
            columns = _.columns
            _ = _.to_numpy()
            X = _[0, 1:-self.lags]
            output = _[0, -self.lags:].tolist()
            X.shape = (self.lags, int(X.shape[0] / self.lags))
            X = X.tolist()
            gips_df = data.loc[data.GeoID == key]
            gips_np = gips_df.loc[:, NPI_COLS].to_numpy()
            for i in range(gips_np.shape[0]):
                X.append(gips_np[i].tolist())
                del X[0]
                output.append(self._last_hy)
                del output[0]
                _ = np.concatenate(([key], np.array(X).flatten(), output))
                d = pd.DataFrame([_], columns=columns)
                d = (d.merge(static, on="GeoID")
                    .reindex(['GeoID'] + self._static_cols + 
                            self._exo_cols + self._lag_cols, axis=1)
                    .rename(columns=self._rename_static)
                    )
                #print(d)
                yield d


class FeaturesN(Features):
    def __init__(self, *args, **kwargs):
        super(FeaturesN, self).__init__(*args, **kwargs)
        self.population = None

    def get_data(self, exo, date, output, key, lag, y=True):
        _ = 100000 * np.array(output) / self.population[key] 
        return super(FeaturesN, self).get_data(exo, date, _, key, lag, y=y)

    def fit(self, data: pd.DataFrame) -> "Features":
        self.population = {k:v for k, v in data.groupby("GeoID").Population.last().items()}
        return super(FeaturesN, self).fit(data)

    def update_prediction(self, hy: np.ndarray) -> float:
        hy = super(FeaturesN, self).update_prediction(hy)
        pop = self.population.get(self._last_key, 1)
        hy = pop * hy / 100000
        return hy


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


class KMeans(object):
    def fit(self, X):
        from sklearn.cluster import KMeans
        r = [(k, v.l29.to_numpy()[-28:]) for k, v in X.groupby("GeoID")]
        data = np.array([x[1] for x in r])
        kmeans = KMeans(n_clusters=2).fit(data)
        self.group = {k: v for (k, _), v  in zip(r, kmeans.predict(data))}
        return self

    def transform(self, X):
        _ = np.atleast_2d([self.group.get(x, 0) for x in X.GeoID]).T
        return np.concatenate((_, X.drop(columns="GeoID").to_numpy()), axis=1)


class ARG(object):
    def model(self):
        from sklearn.linear_model import LinearRegression
        return LinearRegression()

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ARG":
        models = []
        for kl in np.unique(X[:, 0]):
            m = X[:, 0] == kl
            _ = self.model().fit(X[m, 1:], y[m])
            models.append(_)
        self.models = models
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        hy = []
        for x in X:
            m = self.models[int(x[0])]
            _ = m.predict(np.atleast_2d(x[1:]))
            hy.append(_)
        return np.array(hy)

    def decision_function(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        return self.predict(X)


class LarsG(ARG):
    def model(self):
        from sklearn.linear_model import LarsCV
        return LarsCV()


class LassoG(ARG):
    def model(self):
        from sklearn.linear_model import Lasso
        return Lasso(alpha=0.1,
                     precompute=True,
                     max_iter=10000,
                     positive=True,
                     selection='random')          