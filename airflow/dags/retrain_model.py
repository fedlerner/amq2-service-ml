import datetime

from airflow.decorators import dag, task
from airflow.exceptions import AirflowFailException
from airflow.models.param import Param
from airflow.providers.standard.operators.trigger_dagrun import TriggerDagRunOperator

markdown_text = """
### Reentrenamiento champion / challenger

Reentrena el modelo de propinas con un mes nuevo y lo promueve solo si mejora.

1. **run_etl**: dispara `process_etl_taxi_data` para el mes elegido y espera a que termine
   (deja `train.parquet` / `test.parquet` del mes en `s3://data/taxi/<mes>/`).
2. **train_challenger**: clona el pipeline del `champion` (mismos hiperparámetros), lo entrena
   con el train del mes y lo registra en MLflow.
3. **evaluate_champion_challenger**: evalúa los dos modelos con el test del mes nuevo, contra
   la propina real, y le pone el alias `challenger` a la versión nueva.
4. Si el challenger baja el MAE al menos `taxi_min_improvement` (1 % por defecto),
   **promote_challenger** lo pasa a `champion` (el anterior queda como `old_champion` para poder
   volver atrás); si no, **keep_champion** le saca el alias.

El challenger se entrena solo con el mes nuevo y no se vuelven a buscar hiperparámetros: la idea
es adaptar el modelo a los datos recientes, no rediseñarlo. Para volver a buscar hiperparámetros
está `train_taxi_tip_model` (o la notebook del experimento).

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
    # Dos reentrenamientos a la vez se pisarían los alias del registro
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
        # Espera en el worker: en Airflow 3.0.2 la espera diferida (deferrable=True) a veces no
        # se entera de que el ETL terminó y queda colgada hasta el timeout
        deferrable=False,
    )

    # Sin reintentos: si falla después de registrar la versión, un reintento registraría otra
    @task(task_id="train_challenger", retries=0)
    def train_challenger(month: str) -> int:
        """
        Entrena un challenger con los hiperparámetros del champion y lo registra (todavía sin
        alias: se lo pone la evaluación).

        ``taxi_tip`` se importa antes de cargar el champion: al cargar un modelo, MLflow pone
        primero en el path el ``taxi_tip`` que se guardó con él, que puede ser más viejo que el de
        ``src/``.

        :param month: Mes con el que se entrena (YYYY-MM).
        :return: Versión registrada del challenger.
        """
        import awswrangler as wr
        import mlflow
        from sklearn.base import clone
        from taxi_common.data import FEATURES, TARGET, dataset_path
        from taxi_common.registry import champion_version as champion_version_of
        from taxi_tip.mlflow_utils import create_version, log_pipeline
        from taxi_tip.training import metrics as compute_metrics
        from taxi_tip.training import tip_cap as compute_tip_cap

        train = wr.s3.read_parquet(dataset_path(month, "train.parquet"))
        test = wr.s3.read_parquet(dataset_path(month, "test.parquet"))
        X_train = train[FEATURES].astype("float64")
        X_test, y_test = test[FEATURES].astype("float64"), test[TARGET]
        # Mismo recorte del target que en la notebook: p99,5 del train
        tip_cap = compute_tip_cap(train[TARGET])
        y_train = train[TARGET].clip(lower=0, upper=tip_cap)

        client = mlflow.MlflowClient()
        champion_version = champion_version_of(client, MODEL_NAME)
        champion = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{champion_version}")

        challenger = clone(champion).fit(X_train, y_train)
        metrics = {f"test_sin_recorte_{k}": v
                   for k, v in compute_metrics(y_test, challenger.predict(X_test)).items()}

        experiment = mlflow.set_experiment("Taxi Tip")
        model_params = challenger.named_steps["model"].get_params()
        with mlflow.start_run(run_name=f"retrain_{month}", experiment_id=experiment.experiment_id,
                              tags={"modelo": "HistGradientBoostingRegressor",
                                    "proyecto": "taxi_tip", "origen": "retrain_dag"}) as run:
            mlflow.log_params({"etl_month": month, "rows_train": len(X_train),
                               "rows_test": len(X_test), "tip_cap": tip_cap,
                               "champion_version": champion_version, **model_params})
            mlflow.log_metrics(metrics)
            model_info = log_pipeline(challenger, X_train)

        tags = {"modelo": "HistGradientBoostingRegressor", "etl_month": month,
                "mae_test": metrics["test_sin_recorte_mae"], "origen": "retrain_dag"}
        version = create_version(
            client, model_info.model_uri, run.info.run_id, tags,
            description=f"Reentrenado con {month}, hiperparámetros de la v{champion_version}",
        )
        print(f"Challenger v{version.version}: MAE {metrics['test_sin_recorte_mae']:.4f}")
        return int(version.version)

    @task.branch(task_id="evaluate_champion_challenger")
    def evaluate_champion_challenger(challenger_version: int, month: str) -> str:
        """
        Evalúa champion y challenger sobre el mismo test y decide si promover.

        Para promover, el challenger tiene que bajar el MAE del champion al menos
        ``taxi_min_improvement`` (1 % por defecto). Con un solo test, diferencias más chicas
        pueden ser ruido, y sin ese margen el champion podría cambiar de un mes a otro sin que el
        modelo mejore de verdad.

        :param challenger_version: Versión registrada del challenger.
        :param month: Mes del reentrenamiento (YYYY-MM).
        :return: ``promote_challenger`` si el challenger mejora el MAE en al menos el margen; si
                 no, ``keep_champion``.
        """
        import awswrangler as wr
        import mlflow
        from airflow.models import Variable
        from sklearn.metrics import mean_absolute_error
        from taxi_common.data import FEATURES, TARGET, dataset_path
        from taxi_common.registry import champion_version as champion_version_of

        test = wr.s3.read_parquet(dataset_path(month, "test.parquet"))
        X_test, y_test = test[FEATURES].astype("float64"), test[TARGET]

        client = mlflow.MlflowClient()
        champion_version = champion_version_of(client, MODEL_NAME)
        mae = {}
        for alias, version in (("champion", champion_version), ("challenger", challenger_version)):
            model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{version}")
            mae[alias] = mean_absolute_error(y_test, model.predict(X_test))

        client.set_model_version_tag(MODEL_NAME, str(challenger_version),
                                     "mae_champion_mismo_test", str(mae["champion"]))
        # promote_challenger lo usa para no pisar un champion que no se comparó
        client.set_model_version_tag(MODEL_NAME, str(challenger_version),
                                     "champion_comparado", str(champion_version))
        # Recién acá queda como challenger: si algo falla antes, el alias no apunta a una
        # versión sin evaluar
        client.set_registered_model_alias(MODEL_NAME, "challenger", str(challenger_version))
        min_improvement = float(Variable.get("taxi_min_improvement", default_var=0.01))
        threshold = mae["champion"] * (1 - min_improvement)
        print(f"MAE en el test del mes: champion v{champion_version} {mae['champion']:.4f} | "
              f"challenger v{challenger_version} {mae['challenger']:.4f} "
              f"(para promover tiene que ser menor a {threshold:.4f})")
        return "promote_challenger" if mae["challenger"] < threshold else "keep_champion"

    @task(task_id="promote_challenger")
    def promote_challenger(challenger_version: int) -> None:
        """
        Pasa a champion la versión evaluada y deja la anterior como ``old_champion``.

        Usa la versión que llega por XCom y no el alias ``challenger``, por si alguien registró
        otro challenger (por ejemplo, desde la notebook) mientras corría el DAG. Si el champion
        cambió después de la evaluación, no promueve (el challenger nunca se comparó con él): le
        saca el alias, lo marca como ``no_promovido`` y falla.

        :param challenger_version: Versión registrada del challenger.
        """
        import mlflow
        from taxi_common.registry import champion_version, drop_alias_if_points_to

        client = mlflow.MlflowClient()
        old = champion_version(client, MODEL_NAME)
        new = str(challenger_version)
        if old == new:
            # Un intento anterior ya lo promovió y falló después: no hay nada más que hacer
            drop_alias_if_points_to(client, MODEL_NAME, "challenger", new)
            print(f"La v{new} ya es champion (la promovió un intento anterior)")
            return
        compared = client.get_model_version(MODEL_NAME, new).tags.get("champion_comparado")
        if compared != old:
            drop_alias_if_points_to(client, MODEL_NAME, "challenger", new)
            client.set_model_version_tag(MODEL_NAME, new, "estado", "no_promovido")
            raise AirflowFailException(
                f"El champion pasó a ser la v{old} después de evaluar (se comparó contra la "
                f"v{compared}): no se promueve la v{new}, hay que volver a correr el DAG")

        client.set_registered_model_alias(MODEL_NAME, "old_champion", old)
        drop_alias_if_points_to(client, MODEL_NAME, "challenger", new)
        client.set_registered_model_alias(MODEL_NAME, "champion", new)
        print(f"Nuevo champion: v{new} (la v{old} queda como old_champion)")

    @task(task_id="keep_champion")
    def keep_champion(challenger_version: int) -> None:
        """
        Descarta el challenger: le saca el alias y deja la versión marcada en el registro.

        :param challenger_version: Versión registrada del challenger.
        """
        import mlflow
        from taxi_common.registry import drop_alias_if_points_to

        client = mlflow.MlflowClient()
        version = str(challenger_version)
        drop_alias_if_points_to(client, MODEL_NAME, "challenger", version)
        client.set_model_version_tag(MODEL_NAME, version, "estado", "descartado")
        print(f"El champion se mantiene; la v{version} queda descartada")

    month = "{{ params.month }}"
    challenger_version = train_challenger(month)
    run_etl >> challenger_version
    evaluate_champion_challenger(challenger_version, month) >> [
        promote_challenger(challenger_version), keep_champion(challenger_version)]


dag = retrain_taxi_tip_model()
