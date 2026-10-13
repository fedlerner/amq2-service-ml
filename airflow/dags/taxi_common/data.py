"""
Descarga y limpieza de los datos de Yellow Taxi de la TLC.

Lo usan el DAG de ETL (para armar train/test) y el de predicción en lote (para preparar el mes
a predecir), para no tener dos versiones de la limpieza. Las reglas son las de AMq1.
"""
import tempfile

import awswrangler as wr
import pandas as pd
import requests

FEATURES = ['trip_distance', 'duration_min', 'fare_amount', 'recargos',
            'PULocationID', 'DOLocationID', 'passenger_count', 'hour', 'dow',
            'RatecodeID', 'VendorID']
TARGET = 'tip_amount'

INT_COLS = ['PULocationID', 'DOLocationID', 'passenger_count', 'hour', 'dow',
            'RatecodeID', 'VendorID']
RECARGOS_COLS = ['extra', 'mta_tax', 'tolls_amount', 'improvement_surcharge',
                 'congestion_surcharge', 'Airport_fee', 'cbd_congestion_fee']
TLC_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{month}.parquet"


def raw_path(month: str) -> str:
    """Path en S3 del parquet crudo de un mes (YYYY-MM)."""
    return f"s3://data/raw/yellow_tripdata_{month}.parquet"


def ensure_raw_month(month: str) -> str:
    """
    Deja el parquet crudo del mes en ``s3://data/raw/``.

    Si ya existe no hace nada; si no, lo baja de la TLC y lo sube.

    :param month: Mes a procesar, en formato YYYY-MM.
    :return: Path en S3 del parquet crudo.
    """
    path = raw_path(month)
    if wr.s3.does_object_exist(path):
        print(f"{path} ya existe, no se descarga")
        return path

    url = TLC_URL.format(month=month)
    print(f"Descargando {url}")
    with tempfile.NamedTemporaryFile(suffix=".parquet") as tmp:
        with requests.get(url, stream=True, timeout=300) as response:
            response.raise_for_status()
            for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                tmp.write(chunk)
        tmp.flush()
        wr.s3.upload(local_file=tmp.name, path=path)
    return path


def clean_month(path: str, month: str) -> pd.DataFrame:
    """
    Limpia un mes crudo con las mismas reglas que usamos en AMq1.

    - Universo: solo pagos con tarjeta (``payment_type == 1``), porque la propina en efectivo
      no se registra.
    - ``recargos``: suma de todo lo que se paga además de la tarifa.
    - Saca viajes imposibles: duración fuera de (0, 180] min, distancia fuera de (0, 100]
      millas, tarifa < 2.50, propina negativa, pickup fuera del mes, velocidad > 100 mph y
      tarifa estándar incoherente con la distancia.
    - Se queda con las 11 features del modelo más el target. ``total_amount`` queda afuera
      porque incluye la propina.

    :param path: Path en S3 del parquet crudo.
    :param month: Mes procesado (YYYY-MM), para filtrar pickups de otros meses.
    :return: DataFrame con ``FEATURES`` + ``TARGET``, sin muestrear.
    """
    df = wr.s3.read_parquet(path)
    # En algunos años la TLC publicó la columna como airport_fee, en minúscula
    df = df.rename(columns={'airport_fee': 'Airport_fee'})
    n_raw = len(df)

    df = df[df['payment_type'] == 1].copy()

    # Algunos meses viejos no traen todas las columnas de recargos
    df['recargos'] = df[[c for c in RECARGOS_COLS if c in df.columns]].fillna(0).sum(axis=1)
    df['duration_min'] = (
        df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']
    ).dt.total_seconds() / 60

    month_start = pd.Timestamp(f"{month}-01")
    month_end = month_start + pd.offsets.MonthBegin(1)

    df = df[
        (df['duration_min'] > 0) & (df['duration_min'] <= 180) &
        (df['trip_distance'] > 0) & (df['trip_distance'] <= 100) &
        (df['fare_amount'] >= 2.50) &
        (df['tip_amount'] >= 0) &
        (df['tpep_pickup_datetime'] >= month_start) &
        (df['tpep_pickup_datetime'] < month_end)
    ]

    speed_mph = df['trip_distance'] / (df['duration_min'] / 60)
    df = df[speed_mph <= 100]

    # Solo la tarifa estándar (RatecodeID == 1) sigue la relación tarifa-distancia
    incoherente = (df['RatecodeID'] == 1) & (df['fare_amount'] < 2.50 * df['trip_distance'])
    df = df[~incoherente]

    df['hour'] = df['tpep_pickup_datetime'].dt.hour
    df['dow'] = df['tpep_pickup_datetime'].dt.dayofweek

    df = df[FEATURES + [TARGET]].dropna().reset_index(drop=True)
    # Hay meses donde passenger_count y RatecodeID vienen como double; los dejamos enteros
    df[INT_COLS] = df[INT_COLS].astype('int64')
    print(f"Filas crudas: {n_raw:,} | filas limpias: {len(df):,}")
    return df
