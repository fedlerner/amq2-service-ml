import datetime

from airflow.decorators import dag, task
from airflow.exceptions import AirflowFailException
from airflow.models.param import Param
from airflow.providers.standard.operators.trigger_dagrun import TriggerDagRunOperator

markdown_text = """
### Reentrenamiento champion / challenger

Reentrena el modelo de propinas con un mes nuevo y lo promueve solo si mejora.

1. **run_etl**: dispara `process_etl_taxi_data` para el mes elegido y espera a que termine
   (deja `train.parquet` / `test.parquet` del mes en `s3://data/taxi/`).
2. **train_challenger**: clona el pipeline del `champion` (mismos hiperparámetros), lo entrena
   con el train del mes, lo registra en MLflow y le pone el alias `challenger`.
3. **evaluate_champion_challenger**: evalúa los dos modelos con el test del mes nuevo, contra
   la propina real.
4. Si el challenger tiene menor MAE, **promote_challenger** lo pasa a `champion` (el anterior
   queda como `old_champion` para poder volver atrás); si no, **keep_champion** le saca el alias.

La API levanta el champion nuevo sola. El DAG de predicción en lote dispara este DAG cuando
detecta degradación. Necesita que `process_etl_taxi_data` esté activo.
"""

default_args = {
    'owner': "grupo_taxi",
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': datetime.timedelta(minutes=5),
}

MODEL_NAME = "taxi_tip_model"


@dag(
    dag_id="retrain_taxi_tip_model",
    description="Reentrena el modelo de propinas con un mes nuevo (champion / challenger)",
    doc_md=markdown_text,
    default_args=default_args,
    schedule=None,
    dagrun_timeout=datetime.timedelta(minutes=60),
    catchup=False,
    # Usa los mismos train/test de S3 que el ETL: dos corridas a la vez se pisarían
    max_active_runs=1,
    params={"month": Param("2026-04", type="string", pattern=r"^\d{4}-(0[1-9]|1[0-2])$",
                           description="Mes de la TLC con el que reentrenar (YYYY-MM)")},
    tags=["retrain", "taxi"],
)
def retrain_taxi_tip_model():

    run_etl = TriggerDagRunOperator(
        task_id="run_etl",
        trigger_dag_id="process_etl_taxi_data",
        conf={"month": "{{ params.month }}"},
        wait_for_completion=True,
        poke_interval=15,
        allowed_states=["success"],
        failed_states=["failed"],
        # Espera en el triggerer en vez de ocupar un slot del worker
        deferrable=True,
    )

    @task(task_id="train_challenger")
    def train_challenger(month: str) -> int:
        """
        Entrena un challenger con los hiperparámetros del champion y lo registra.

        Al cargar el champion, MLflow agrega al path el paquete ``taxi_tip`` que guardamos con
        ``code_paths``, y lo reusamos para loguear el challenger.

        :param month: Mes con el que se entrena (YYYY-MM).
        :return: Versión registrada del challenger.
        """
        import json
        import os

        import awswrangler as wr
        import boto3
        import mlflow
        from mlflow.models import infer_signature
        from sklearn.base import clone
        from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
        from taxi_common.data import FEATURES, TARGET
        from taxi_common.registry import champion_version as champion_version_of

        data_info = json.loads(
            boto3.client("s3").get_object(Bucket="data", Key="taxi/data_info.json")["Body"].read()
        )
        if data_info["month"] != month:
            raise AirflowFailException(f"El train en S3 es de {data_info['month']}, no de {month}")

        train = wr.s3.read_parquet("s3://data/taxi/train.parquet")
        test = wr.s3.read_parquet("s3://data/taxi/test.parquet")
        X_train = train[FEATURES].astype("float64")
        X_test, y_test = test[FEATURES].astype("float64"), test[TARGET]
        # Mismo recorte del target que en la notebook: p99,5 del train
        tip_cap = float(train[TARGET].quantile(0.995))
        y_train = train[TARGET].clip(lower=0, upper=tip_cap)

        mlflow.set_tracking_uri("http://mlflow:5000")
        client = mlflow.MlflowClient()
        champion_version = champion_version_of(client, MODEL_NAME)
        champion = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{champion_version}")
        import taxi_tip  # lo deja importable el code_paths del champion
        from taxi_tip.mlflow_utils import skops_trusted_types

        challenger = clone(champion).fit(X_train, y_train)
        y_pred = challenger.predict(X_test)
        metrics = {"test_sin_recorte_mae": mean_absolute_error(y_test, y_pred),
                   "test_sin_recorte_rmse": root_mean_squared_error(y_test, y_pred),
                   "test_sin_recorte_r2": r2_score(y_test, y_pred)}

        experiment = mlflow.set_experiment("Taxi Tip")
        model_params = challenger.named_steps["model"].get_params()
        with mlflow.start_run(run_name=f"retrain_{month}", experiment_id=experiment.experiment_id,
                              tags={"modelo": "HistGradientBoostingRegressor",
                                    "proyecto": "taxi_tip", "origen": "retrain_dag"}) as run:
            mlflow.log_params({"etl_month": month, "rows_train": len(X_train),
                               "rows_test": len(X_test), "tip_cap": tip_cap,
                               "champion_version": champion_version, **model_params})
            mlflow.log_metrics(metrics)
            model_info = mlflow.sklearn.log_model(
                challenger,
                name="model",
                signature=infer_signature(X_train, challenger.predict(X_train.head(100))),
                input_example=X_train.head(3),
                code_paths=[os.path.dirname(taxi_tip.__file__)],
                skops_trusted_types=skops_trusted_types(challenger),
            )

        tags = {"modelo": "HistGradientBoostingRegressor", "etl_month": month,
                "mae_test": str(metrics["test_sin_recorte_mae"]), "origen": "retrain_dag"}
        version = client.create_model_version(
            name=MODEL_NAME, source=model_info.model_uri, run_id=run.info.run_id, tags=tags,
            description=f"Reentrenado con {month}, hiperparámetros de la v{champion_version}",
        )
        client.set_registered_model_alias(MODEL_NAME, "challenger", version.version)
        print(f"Challenger v{version.version}: MAE {metrics['test_sin_recorte_mae']:.4f}")
        return int(version.version)

    @task.branch(task_id="evaluate_champion_challenger")
    def evaluate_champion_challenger(challenger_version: int, month: str) -> str:
        """
        Evalúa champion y challenger sobre el mismo test y decide si promover.

        :param challenger_version: Versión registrada del challenger.
        :param month: Mes del reentrenamiento (YYYY-MM).
        :return: ``promote_challenger`` si el challenger tiene menor MAE; si no,
                 ``keep_champion``.
        """
        import json

        import awswrangler as wr
        import boto3
        import mlflow
        from sklearn.metrics import mean_absolute_error
        from taxi_common.data import FEATURES, TARGET

        data_info = json.loads(
            boto3.client("s3").get_object(Bucket="data", Key="taxi/data_info.json")["Body"].read()
        )
        if data_info["month"] != month:
            raise AirflowFailException(f"El test en S3 es de {data_info['month']}, no de {month}")

        test = wr.s3.read_parquet("s3://data/taxi/test.parquet")
        X_test, y_test = test[FEATURES].astype("float64"), test[TARGET]

        mlflow.set_tracking_uri("http://mlflow:5000")
        client = mlflow.MlflowClient()
        champion_version = client.get_model_version_by_alias(MODEL_NAME, "champion").version
        mae = {}
        for alias, version in (("champion", champion_version), ("challenger", challenger_version)):
            model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{version}")
            mae[alias] = mean_absolute_error(y_test, model.predict(X_test))

        client.set_model_version_tag(MODEL_NAME, str(challenger_version),
                                     "mae_champion_mismo_test", str(mae["champion"]))
        # promote_challenger lo usa para no pisar un champion que no se comparó
        client.set_model_version_tag(MODEL_NAME, str(challenger_version),
                                     "champion_comparado", str(champion_version))
        print(f"MAE en el test del mes: champion v{champion_version} {mae['champion']:.4f} | "
              f"challenger v{challenger_version} {mae['challenger']:.4f}")
        return "promote_challenger" if mae["challenger"] < mae["champion"] else "keep_champion"

    @task(task_id="promote_challenger")
    def promote_challenger(challenger_version: int) -> None:
        """
        Pasa a champion la versión evaluada y deja la anterior como ``old_champion``.

        Usa la versión que llega por XCom y no el alias ``challenger``, por si alguien registró
        otro challenger (por ejemplo, desde la notebook) mientras corría el DAG. Si el champion
        cambió después de la evaluación, no promueve: el challenger nunca se comparó con él.

        :param challenger_version: Versión registrada del challenger.
        """
        import mlflow

        mlflow.set_tracking_uri("http://mlflow:5000")
        client = mlflow.MlflowClient()
        old = client.get_model_version_by_alias(MODEL_NAME, "champion").version
        new = str(challenger_version)
        compared = client.get_model_version(MODEL_NAME, new).tags.get("champion_comparado")
        if compared != old:
            raise AirflowFailException(
                f"El champion pasó a ser la v{old} después de evaluar (se comparó contra la "
                f"v{compared}): no se promueve la v{new}, hay que volver a correr el DAG")

        client.set_registered_model_alias(MODEL_NAME, "old_champion", old)
        if client.get_registered_model(MODEL_NAME).aliases.get("challenger") == new:
            client.delete_registered_model_alias(MODEL_NAME, "challenger")
        client.set_registered_model_alias(MODEL_NAME, "champion", new)
        print(f"Nuevo champion: v{new} (la v{old} queda como old_champion)")

    @task(task_id="keep_champion")
    def keep_champion(challenger_version: int) -> None:
        """
        Descarta el challenger: le saca el alias y deja la versión marcada en el registro.

        :param challenger_version: Versión registrada del challenger.
        """
        import mlflow

        mlflow.set_tracking_uri("http://mlflow:5000")
        client = mlflow.MlflowClient()
        version = str(challenger_version)
        aliases = client.get_registered_model(MODEL_NAME).aliases
        if aliases.get("challenger") == version:
            client.delete_registered_model_alias(MODEL_NAME, "challenger")
        client.set_model_version_tag(MODEL_NAME, version, "estado", "descartado")
        print(f"El champion se mantiene; la v{version} queda descartada")

    month = "{{ params.month }}"
    challenger_version = train_challenger(month)
    run_etl >> challenger_version
    evaluate_champion_challenger(challenger_version, month) >> [
        promote_challenger(challenger_version), keep_champion(challenger_version)]


dag = retrain_taxi_tip_model()
