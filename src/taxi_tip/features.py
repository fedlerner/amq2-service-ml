"""
Feature engineering del modelo de propinas de Yellow Taxi.

Está armado como un ``Pipeline`` de sklearn que se registra entero en MLflow. Las reglas son
las de AMq1.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, TargetEncoder
from sklearn.utils.validation import check_is_fitted

# Columnas que deja el ETL, en este orden
RAW_FEATURES = ['trip_distance', 'duration_min', 'fare_amount', 'recargos',
                'PULocationID', 'DOLocationID', 'passenger_count', 'hour', 'dow',
                'RatecodeID', 'VendorID']
TARGET = 'tip_amount'

WINSOR_COLS = ['trip_distance', 'duration_min', 'fare_amount', 'recargos']
ZONE_COLS = ['PULocationID', 'DOLocationID']
CATEGORICAL_COLS = ['VendorID', 'RatecodeID']

# Definiciones de AMq1: aeropuerto por tarifa JFK (2) o Newark (3), hora pico 7-9 y 17-19
AIRPORT_RATECODES = [2, 3]
PEAK_HOURS = [7, 8, 9, 17, 18, 19]
RANDOM_STATE = 42


class TaxiFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Winsoriza las variables continuas y crea las features derivadas de AMq1.

    En ``fit`` aprende, solo con train, los límites de winsorización y la mediana de
    ``fare_per_mile``. En ``transform``:

    - recorta ``trip_distance``, ``duration_min``, ``fare_amount`` y ``recargos`` a
      [max(p1, 0), p99];
    - agrega ``speed_mph``, ``fare_per_mile``, ``es_aeropuerto`` y ``es_hora_pico``;
    - reemplaza ``hour`` y ``dow`` por su codificación cíclica (seno y coseno).

    Las zonas y las categóricas pasan sin tocar; las codifica el paso siguiente.

    :param lower_quantile: Cuantil inferior de la winsorización.
    :param upper_quantile: Cuantil superior de la winsorización.
    """

    def __init__(self, lower_quantile: float = 0.01, upper_quantile: float = 0.99):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def fit(self, X: pd.DataFrame, y=None):
        """
        Aprende los límites de winsorización y la mediana de ``fare_per_mile``.

        :param X: DataFrame con las columnas de ``RAW_FEATURES``.
        :param y: No se usa.
        :return: El transformer ajustado.
        """
        X = self._as_float(X)
        self.lower_bounds_ = {c: max(float(X[c].quantile(self.lower_quantile)), 0.0)
                              for c in WINSOR_COLS}
        self.upper_bounds_ = {c: float(X[c].quantile(self.upper_quantile))
                              for c in WINSOR_COLS}
        clipped = self._clip(X)
        fare_per_mile = clipped['fare_amount'] / clipped['trip_distance'].replace(0, np.nan)
        self.fare_per_mile_median_ = float(fare_per_mile.median())
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica la winsorización y agrega las features derivadas.

        :param X: DataFrame con las columnas de ``RAW_FEATURES``.
        :return: DataFrame con las columnas de ``get_feature_names_out()``.
        """
        check_is_fitted(self, 'fare_per_mile_median_')
        X = self._clip(self._as_float(X))

        X['speed_mph'] = X['trip_distance'] / (X['duration_min'] / 60)
        X['fare_per_mile'] = (X['fare_amount'] / X['trip_distance'].replace(0, np.nan)
                              ).fillna(self.fare_per_mile_median_)
        X['es_aeropuerto'] = X['RatecodeID'].isin(AIRPORT_RATECODES).astype(float)
        X['es_hora_pico'] = X['hour'].isin(PEAK_HOURS).astype(float)

        X['hour_sin'] = np.sin(2 * np.pi * X['hour'] / 24)
        X['hour_cos'] = np.cos(2 * np.pi * X['hour'] / 24)
        X['dow_sin'] = np.sin(2 * np.pi * X['dow'] / 7)
        X['dow_cos'] = np.cos(2 * np.pi * X['dow'] / 7)

        return X.drop(columns=['hour', 'dow'])

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """Nombres de las columnas que devuelve ``transform``."""
        base = [c for c in RAW_FEATURES if c not in ('hour', 'dow')]
        derived = ['speed_mph', 'fare_per_mile', 'es_aeropuerto', 'es_hora_pico',
                   'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos']
        return np.array(base + derived, dtype=object)

    @staticmethod
    def _as_float(X: pd.DataFrame) -> pd.DataFrame:
        """Ordena las columnas como ``RAW_FEATURES`` y las pasa a float."""
        # El ETL guarda enteros; el modelo trabaja todo en float
        return pd.DataFrame(X, columns=RAW_FEATURES).astype('float64')

    def _clip(self, X: pd.DataFrame) -> pd.DataFrame:
        """Recorta las columnas de ``WINSOR_COLS`` a los límites aprendidos en ``fit``."""
        X = X.copy()
        for c in WINSOR_COLS:
            X[c] = X[c].clip(lower=self.lower_bounds_[c], upper=self.upper_bounds_[c])
        return X


def build_pipeline(**model_params) -> Pipeline:
    """
    Arma el pipeline completo: feature engineering, codificación y modelo.

    - Zonas (``PULocationID``, ``DOLocationID``): target encoding. ``TargetEncoder`` usa
      validación cruzada interna al ajustar, para que al codificar train no se filtre su
      propio target.
    - ``VendorID`` y ``RatecodeID``: one-hot; una categoría nueva en producción queda en ceros.
    - Modelo: ``HistGradientBoostingRegressor`` con pérdida absoluta, como en AMq1.

    No se escala: los árboles no dependen de la escala de las variables.

    :param model_params: Hiperparámetros del ``HistGradientBoostingRegressor``.
    :return: Pipeline sin ajustar.
    """
    encoder = ColumnTransformer(
        transformers=[
            ('zonas', TargetEncoder(target_type='continuous', random_state=RANDOM_STATE),
             ZONE_COLS),
            ('categoricas', OneHotEncoder(handle_unknown='ignore', sparse_output=False),
             CATEGORICAL_COLS),
        ],
        remainder='passthrough',
        verbose_feature_names_out=False,
    )
    model = HistGradientBoostingRegressor(loss='absolute_error', random_state=RANDOM_STATE,
                                          **model_params)
    return Pipeline([
        ('features', TaxiFeatureEngineer()),
        ('encoding', encoder),
        ('model', model),
    ])
