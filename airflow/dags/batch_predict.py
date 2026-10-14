import datetime

from airflow.decorators import dag, task
from airflow.models.param import Param
from airflow.providers.standard.operators.empty import EmptyOperator
from airflow.providers.standard.operators.trigger_dagrun import TriggerDagRunOperator

markdown_text = """
### Predicción en lote de propinas

Toma un mes de la TLC, predice la propina de todos sus viajes con el modelo `champion` y
deja los resultados en Postgres.

1. **extract_raw_data**: si el mes no está en `s3://data/raw/`, lo baja de la TLC.
2. **predict**: limpia el mes con las mismas reglas que el ETL (`taxi_common/data.py`), carga
   `taxi_tip_model@champion` desde MLflow y predice. Guarda el resultado en
   `s3://data/taxi/predictions/<mes>.parquet`.
3. **store_predictions**: carga las predicciones en la tabla `predicciones_propina` de la base
   `taxi` (conexión `taxi_db`). Si el mes ya estaba, lo reemplaza.
4. **log_metrics**: como la propina real viene en los datos, mide el error del mes y lo
   registra en el experimento `Taxi Tip Monitoreo` de MLflow. Si el MAE del mes es más de
   `taxi_mae_alert_ratio` veces (1,2 por defecto) el MAE de test del champion, marca el run con
   `alerta=degradacion`.
5. **check_degradation**: cuando terminan los pasos 3 y 4, si hubo degradación
   **trigger_retrain** dispara `retrain_taxi_tip_model` con este mismo mes; si no, termina en
   **no_retrain**.

El mes se elige con el parámetro `month` (formato `YYYY-MM`) al disparar el DAG.
"""

default_args = {
    'owner': "grupo_taxi",
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': datetime.timedelta(minutes=5),
}

MODEL_NAME = "taxi_tip_model"
MODEL_ALIAS = "champion"
TABLE = "predicciones_propina"


@dag(
    dag_id="batch_predict_taxi_tips",
    description="Predicción en lote de propinas de un mes con el modelo champion",
    doc_md=markdown_text,
    default_args=default_args,
    schedule=None,
    dagrun_timeout=datetime.timedelta(minutes=45),
    catchup=False,
    max_active_runs=1,
    params={"month": Param("2026-04", type="string", pattern=r"^\d{4}-(0[1-9]|1[0-2])$",
                           description="Mes de la TLC a predecir (YYYY-MM)")},
    tags=["batch", "taxi"],
)
def batch_predict_taxi_tips():

    @task(task_id="extract_raw_data")
    def extract_raw_data(month: str) -> str:
        """
        Deja el parquet crudo del mes en ``s3://data/raw/`` (lo baja de la TLC si falta).

        :param month: Mes a predecir, en formato YYYY-MM.
        :return: Path en S3 del parquet crudo.
        """
        from taxi_common.data import ensure_raw_month

        return ensure_raw_month(month)

    @task(task_id="predict")
    def predict(raw_path: str, month: str) -> dict:
        """
        Limpia el mes completo y predice la propina de cada viaje con el champion.

        :param raw_path: Path en S3 del parquet crudo.
        :param month: Mes a predecir (YYYY-MM).
        :return: Path del parquet con predicciones y versión del modelo usada.
        """
        import awswrangler as wr
        import mlflow
        import numpy as np
        from taxi_common.data import FEATURES, clean_month
        from taxi_common.registry import champion_version

        df = clean_month(raw_path, month)

        version = champion_version(mlflow.MlflowClient(), MODEL_NAME, MODEL_ALIAS)
        model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{version}")

        # Predecimos en bloques: el pipeline hace varias copias de los datos y un mes entero de
        # una sola vez ocupa varios GB
        chunk_size = 500_000
        preds = [model.predict(df[FEATURES].iloc[start:start + chunk_size].astype('float64'))
                 for start in range(0, len(df), chunk_size)]

        # La cruda es la que se usa para medir el MAE, igual que el mae_test del registro;
        # a Postgres va la recortada y redondeada, que es lo que sirve la API
        df['tip_amount_pred_raw'] = np.concatenate(preds)
        df['tip_amount_pred'] = df['tip_amount_pred_raw'].clip(lower=0).round(2)
        df['month'] = month
        df['model_version'] = int(version)

        path = f"s3://data/taxi/predictions/{month}.parquet"
        wr.s3.to_parquet(df, path, index=False)
        print(f"{len(df):,} viajes predichos con {MODEL_NAME} v{version}")
        return {"path": path, "model_version": int(version)}

    @task(task_id="store_predictions")
    def store_predictions(result: dict, month: str) -> None:
        """
        Carga las predicciones del mes en ``predicciones_propina`` (base ``taxi``).

        Crea la tabla si no existe y borra antes las filas del mes, así correr dos veces el
        mismo mes no duplica datos. Inserta con ``COPY`` en bloques, que es mucho más rápido
        que ``INSERT`` para millones de filas.

        :param result: Salida de ``predict``.
        :param month: Mes predicho (YYYY-MM).
        """
        import io

        import awswrangler as wr
        from airflow.providers.postgres.hooks.postgres import PostgresHook
        from taxi_common.data import FEATURES, TARGET

        # Mismo orden que las columnas del COPY de abajo
        columns = ['month', *FEATURES, TARGET, 'tip_amount_pred', 'model_version']
        df = wr.s3.read_parquet(result["path"], columns=columns)[columns]

        create_sql = f"""
            CREATE TABLE IF NOT EXISTS {TABLE} (
                id               BIGSERIAL PRIMARY KEY,
                month            CHAR(7)          NOT NULL,
                trip_distance    DOUBLE PRECISION NOT NULL,
                duration_min     DOUBLE PRECISION NOT NULL,
                fare_amount      DOUBLE PRECISION NOT NULL,
                recargos         DOUBLE PRECISION NOT NULL,
                pu_location_id   INTEGER          NOT NULL,
                do_location_id   INTEGER          NOT NULL,
                passenger_count  INTEGER          NOT NULL,
                hour             SMALLINT         NOT NULL,
                dow              SMALLINT         NOT NULL,
                ratecode_id      INTEGER          NOT NULL,
                vendor_id        INTEGER          NOT NULL,
                tip_amount       DOUBLE PRECISION NOT NULL,
                tip_amount_pred  DOUBLE PRECISION NOT NULL,
                model_version    INTEGER          NOT NULL,
                predicted_at     TIMESTAMPTZ      NOT NULL DEFAULT now()
            );
            CREATE INDEX IF NOT EXISTS {TABLE}_month_idx ON {TABLE} (month);
        """
        copy_sql = f"""
            COPY {TABLE} (month, trip_distance, duration_min, fare_amount, recargos,
                          pu_location_id, do_location_id, passenger_count, hour, dow,
                          ratecode_id, vendor_id, tip_amount, tip_amount_pred, model_version)
            FROM STDIN WITH (FORMAT csv)
        """

        conn = PostgresHook(postgres_conn_id="taxi_db").get_conn()
        try:
            with conn.cursor() as cursor:
                cursor.execute(create_sql)
                cursor.execute(f"DELETE FROM {TABLE} WHERE month = %s", (month,))
                chunk_size = 500_000
                for start in range(0, len(df), chunk_size):
                    buffer = io.StringIO()
                    df.iloc[start:start + chunk_size].to_csv(buffer, index=False, header=False)
                    buffer.seek(0)
                    cursor.copy_expert(copy_sql, buffer)
            conn.commit()
        finally:
            conn.close()
        print(f"{len(df):,} filas cargadas en {TABLE} para {month}")

    @task(task_id="log_metrics")
    def log_metrics(result: dict, month: str) -> bool:
        """
        Mide el error del mes contra la propina real y lo registra en MLflow.

        Usa la predicción cruda del modelo, la misma con la que se calculó ``mae_test``, y
        compara el MAE con ese tag de la versión champion. Si es más de
        ``taxi_mae_alert_ratio`` veces ese valor, marca el run con ``alerta=degradacion``.
        Si la versión no tiene el tag, no hay contra qué comparar: lo avisa y no dispara nada.

        :param result: Salida de ``predict``.
        :param month: Mes predicho (YYYY-MM).
        :return: ``True`` si el modelo se degradó.
        """
        import math

        import awswrangler as wr
        import mlflow
        from airflow.models import Variable
        from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

        df = wr.s3.read_parquet(result["path"], columns=['tip_amount', 'tip_amount_pred_raw'])
        y_true, y_pred = df['tip_amount'], df['tip_amount_pred_raw']
        metrics = {
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': root_mean_squared_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'tip_real_media': float(y_true.mean()),
            'tip_pred_media': float(y_pred.mean()),
        }

        client = mlflow.MlflowClient()
        version = client.get_model_version(MODEL_NAME, str(result["model_version"]))
        mae_test = float(version.tags.get('mae_test', 'nan'))
        alert_ratio = float(Variable.get("taxi_mae_alert_ratio", default_var=1.2))
        degraded = metrics['mae'] > alert_ratio * mae_test
        if math.isnan(mae_test):
            print(f"La versión {result['model_version']} no tiene tag mae_test: no se evalúa la alerta")

        experiment = mlflow.set_experiment("Taxi Tip Monitoreo")
        run_name = f"batch_{month}"
        # Igual que en Postgres, una corrida nueva del mes reemplaza a la anterior
        for old_run in client.search_runs([experiment.experiment_id],
                                          filter_string=f"tags.mlflow.runName = '{run_name}'"):
            client.delete_run(old_run.info.run_id)
        with mlflow.start_run(run_name=run_name, experiment_id=experiment.experiment_id):
            mlflow.log_params({'month': month, 'model_version': result["model_version"],
                               'rows': len(df), 'mae_test_champion': mae_test,
                               'alert_ratio': alert_ratio})
            mlflow.log_metrics(metrics)
            mlflow.set_tag('alerta', 'sin_referencia' if math.isnan(mae_test)
                           else 'degradacion' if degraded else 'ok')

        print(f"MAE {month}: {metrics['mae']:.4f} (test del champion: {mae_test:.4f})"
              f" -> {'DEGRADACIÓN' if degraded else 'ok'}")
        return bool(degraded)

    @task.branch(task_id="check_degradation")
    def check_degradation(degraded: bool) -> str:
        """Elige si hay que reentrenar según el resultado de ``log_metrics``."""
        return "trigger_retrain" if degraded else "no_retrain"

    month = "{{ params.month }}"
    raw_path = extract_raw_data(month)
    result = predict(raw_path, month)
    stored = store_predictions(result, month)
    degraded = log_metrics(result, month)

    trigger_retrain = TriggerDagRunOperator(
        task_id="trigger_retrain",
        trigger_dag_id="retrain_taxi_tip_model",
        conf={"month": month},
    )
    check = check_degradation(degraded)
    stored >> check
    check >> [trigger_retrain, EmptyOperator(task_id="no_retrain")]


dag = batch_predict_taxi_tips()
