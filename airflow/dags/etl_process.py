import datetime

from airflow.decorators import dag, task
from airflow.models.param import Param

markdown_text = """
### ETL de Yellow Taxi NYC (propinas)

Prepara los datos para el modelo que estima `tip_amount` en viajes pagados con tarjeta.

1. **extract_raw_data**: si el mes no está en `s3://data/raw/`, lo baja de la TLC y lo sube.
2. **clean_data**: aplica las reglas de limpieza de AMq1, calcula `duration_min`, `recargos`,
   `hour` y `dow`, se queda con las 11 features del modelo y toma una muestra.
3. **split_dataset**: separa train/test, guarda los parquet en `s3://data/taxi/<mes>/` y
   registra el run en MLflow.

La descarga y la limpieza están en `taxi_common/data.py`, compartido con el DAG de
predicción en lote. El feature engineering (winsorización, target encoding, dummies, etc.) no
se hace acá, está dentro del `Pipeline` que se registra en MLflow (`src/taxi_tip/features.py`).

El mes se elige con el parámetro `month` (formato `YYYY-MM`) al disparar el DAG.
"""

default_args = {
    'owner': "grupo_taxi",
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': datetime.timedelta(minutes=5),
}


@dag(
    dag_id="process_etl_taxi_data",
    description="ETL del dataset de Yellow Taxi NYC para predecir tip_amount",
    doc_md=markdown_text,
    default_args=default_args,
    schedule=None,
    dagrun_timeout=datetime.timedelta(minutes=45),
    catchup=False,
    # Dos corridas del mismo mes escribirían los mismos archivos
    max_active_runs=1,
    params={"month": Param("2026-03", type="string", pattern=r"^\d{4}-(0[1-9]|1[0-2])$",
                           description="Mes de la TLC a procesar (YYYY-MM)")},
    tags=["ETL", "taxi"],
)
def process_etl_taxi_data():

    @task(task_id="extract_raw_data")
    def extract_raw_data(month: str) -> str:
        """
        Deja el parquet crudo del mes en ``s3://data/raw/`` (lo baja de la TLC si falta).

        :param month: Mes a procesar, en formato YYYY-MM.
        :return: Path en S3 del parquet crudo.
        """
        from taxi_common.data import ensure_raw_month

        return ensure_raw_month(month)

    @task(task_id="clean_data")
    def clean_data(raw_path: str, month: str) -> str:
        """
        Limpia el mes con las reglas de AMq1 y toma una muestra para entrenar.

        El tamaño de la muestra sale de la variable ``taxi_sample_size``.

        :param raw_path: Path en S3 del parquet crudo.
        :param month: Mes procesado (YYYY-MM).
        :return: Path en S3 del parquet limpio (muestreado).
        """
        import awswrangler as wr
        from airflow.models import Variable
        from taxi_common.data import clean_month, dataset_path

        df = clean_month(raw_path, month)

        sample_size = int(Variable.get("taxi_sample_size", default_var=300000))
        random_state = int(Variable.get("taxi_random_state", default_var=42))
        if len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)
        print(f"Muestra para entrenar: {len(df):,} filas")

        clean_path = dataset_path(month, "clean.parquet")
        wr.s3.to_parquet(df, clean_path)
        return clean_path

    @task(task_id="split_dataset")
    def split_dataset(clean_path: str, month: str) -> None:
        """
        Separa el dataset limpio en train/test y lo deja listo para entrenar.

        Guarda ``train.parquet``, ``test.parquet`` y un JSON con la descripción del dataset
        (``data_info.json``) en ``s3://data/taxi/<mes>/``, y registra el run en el experimento
        ``Taxi Tip ETL`` de MLflow.

        :param clean_path: Path en S3 del parquet limpio.
        :param month: Mes procesado (YYYY-MM).
        """
        import json

        import awswrangler as wr
        import boto3
        import mlflow
        from airflow.models import Variable
        from sklearn.model_selection import train_test_split
        from taxi_common.data import TARGET, dataset_path

        df = wr.s3.read_parquet(clean_path)

        test_size = float(Variable.get("taxi_test_size", default_var=0.2))
        random_state = int(Variable.get("taxi_random_state", default_var=42))

        train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)

        wr.s3.to_parquet(train_df, dataset_path(month, "train.parquet"), index=False)
        wr.s3.to_parquet(test_df, dataset_path(month, "test.parquet"), index=False)

        data_info = {
            "month": month,
            "target": TARGET,
            "features": [c for c in df.columns if c != TARGET],
            "dtypes": {c: str(t) for c, t in df.dtypes.items()},
            "rows_train": len(train_df),
            "rows_test": len(test_df),
        }
        boto3.client("s3").put_object(
            Bucket="data",
            Key=f"taxi/{month}/data_info.json",
            Body=json.dumps(data_info, indent=2).encode("utf-8"),
        )

        mlflow.set_tracking_uri("http://mlflow:5000")
        experiment = mlflow.set_experiment("Taxi Tip ETL")
        with mlflow.start_run(run_name=f"etl_taxi_{month}", experiment_id=experiment.experiment_id):
            mlflow.log_params({
                "month": month,
                "rows_train": len(train_df),
                "rows_test": len(test_df),
                "test_size": test_size,
                "random_state": random_state,
                "target": TARGET,
            })
            mlflow.log_dict(data_info, "data_info.json")

    month = "{{ params.month }}"
    raw_path = extract_raw_data(month)
    clean_path = clean_data(raw_path, month)
    split_dataset(clean_path, month)


dag = process_etl_taxi_data()
