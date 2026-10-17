import datetime

from airflow.decorators import dag, task
from airflow.models.param import Param
from airflow.providers.standard.operators.trigger_dagrun import TriggerDagRunOperator

markdown_text = """
### Entrenamiento del modelo de propinas

Corre el ETL de un mes, busca hiperparámetros con Optuna, entrena el modelo y lo registra en
MLflow. Es lo mismo que hace la notebook del experimento, pero orquestado: con un solo trigger
se pasa de no tener datos a tener un `champion` que sirve la API.

1. **run_etl**: dispara `process_etl_taxi_data` para el mes elegido y espera a que termine.
2. **search_hyperparameters**: Optuna con `taxi_n_trials` trials (15 por defecto), cada uno
   evaluado con CV de 3 folds sobre train y registrado como run anidado del experimento
   `Taxi Tip`.
3. **train_and_register**: entrena el pipeline con los mejores hiperparámetros sobre todo el
   train, lo evalúa en test (contra dos baselines) y registra una versión de `taxi_tip_model`.
4. **promote**: si no hay champion, la versión nueva pasa a `champion`. Si hay, solo lo
   reemplaza si baja su MAE en este mismo test al menos `taxi_min_improvement` (1 %); si no,
   queda como `challenger`.

La búsqueda, los baselines y la regla de promoción están en `src/taxi_tip/training.py`,
compartido con la notebook. A diferencia de `retrain_taxi_tip_model`, que reusa los
hiperparámetros del champion, acá se vuelven a buscar.
"""

default_args = {
    'owner': "grupo_taxi",
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': datetime.timedelta(minutes=5),
}


@dag(
    dag_id="train_taxi_tip_model",
    description="ETL + búsqueda de hiperparámetros + registro del modelo de propinas",
    doc_md=markdown_text,
    default_args=default_args,
    schedule=None,
    dagrun_timeout=datetime.timedelta(minutes=90),
    catchup=False,
    # Dos entrenamientos a la vez se pisarían los alias del registro
    max_active_runs=1,
    params={"month": Param("2026-03", type="string", pattern=r"^\d{4}-(0[1-9]|1[0-2])$",
                           description="Mes de la TLC con el que entrenar (YYYY-MM)")},
    tags=["train", "taxi"],
)
def train_taxi_tip_model():

    run_etl = TriggerDagRunOperator(
        task_id="run_etl",
        trigger_dag_id="process_etl_taxi_data",
        conf={"month": "{{ params.month }}"},
        wait_for_completion=True,
        poke_interval=15,
        allowed_states=["success"],
        failed_states=["failed"],
        # Igual que en el reentrenamiento: la espera diferida a veces queda colgada en 3.0.2
        deferrable=False,
    )

    @task(task_id="search_hyperparameters")
    def search_hyperparameters(month: str) -> dict:
        """
        Busca hiperparámetros con Optuna y deja todo en un run padre del experimento.

        :param month: Mes con el que se entrena (YYYY-MM).
        :return: Id del run padre y mejores hiperparámetros.
        """
        import datetime as dt

        import awswrangler as wr
        import mlflow
        import optuna
        from airflow.models import Variable
        from taxi_common.data import FEATURES, TARGET, dataset_path
        from taxi_tip.mlflow_utils import get_or_create_experiment
        from taxi_tip.training import compute_baselines, search_hyperparameters, tip_cap

        train = wr.s3.read_parquet(dataset_path(month, "train.parquet"))
        test = wr.s3.read_parquet(dataset_path(month, "test.parquet"))
        X_train, X_test = train[FEATURES].astype("float64"), test[FEATURES].astype("float64")
        cap = tip_cap(train[TARGET])
        y_train = train[TARGET].clip(lower=0, upper=cap)
        y_test = test[TARGET].clip(lower=0, upper=cap)

        n_trials = int(Variable.get("taxi_n_trials", default_var=15))
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        experiment_id = get_or_create_experiment("Taxi Tip")
        run_name = "optuna_hgb_" + dt.datetime.now().strftime("%Y-%m-%d_%H-%M")
        with mlflow.start_run(experiment_id=experiment_id, run_name=run_name,
                              tags={"modelo": "HistGradientBoostingRegressor",
                                    "proyecto": "taxi_tip", "origen": "train_dag"}) as parent:
            mlflow.log_params({"etl_month": month, "rows_train": len(X_train),
                               "rows_test": len(X_test), "n_trials": n_trials, "tip_cap": cap})
            baselines, _ = compute_baselines(X_train, y_train, X_test, y_test)
            for name, values in baselines.items():
                mlflow.log_metrics({f"baseline_{name}_{k}": v for k, v in values.items()})

            study = search_hyperparameters(X_train, y_train, experiment_id, n_trials=n_trials)
            mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
            mlflow.log_metric("best_cv_mae", study.best_value)

        print(f"Mejores hiperparámetros: {study.best_params} (MAE de CV {study.best_value:.4f})")
        # XCom serializa a JSON: los valores de Optuna ya son int / float de Python
        return {"run_id": parent.info.run_id, "params": dict(study.best_params)}

    # Sin reintentos: si falla después de registrar la versión, un reintento registraría otra
    @task(task_id="train_and_register", retries=0)
    def train_and_register(search: dict, month: str) -> dict:
        """
        Entrena con los mejores hiperparámetros, evalúa en test y registra la versión.

        El MAE que queda en el tag ``mae_test`` se calcula contra la propina sin recortar, igual
        que en la notebook y en la predicción en lote.

        :param search: Salida de ``search_hyperparameters``.
        :param month: Mes con el que se entrena (YYYY-MM).
        :return: Versión registrada y su MAE de test sin recorte.
        """
        import awswrangler as wr
        import mlflow
        from taxi_common.data import FEATURES, TARGET, dataset_path
        from taxi_tip.features import build_pipeline
        from taxi_tip.mlflow_utils import create_version, log_pipeline
        from taxi_tip.training import metrics, tip_cap

        train = wr.s3.read_parquet(dataset_path(month, "train.parquet"))
        test = wr.s3.read_parquet(dataset_path(month, "test.parquet"))
        X_train, X_test = train[FEATURES].astype("float64"), test[FEATURES].astype("float64")
        cap = tip_cap(train[TARGET])
        y_train = train[TARGET].clip(lower=0, upper=cap)
        y_test_raw = test[TARGET]

        model = build_pipeline(**search["params"]).fit(X_train, y_train)
        y_pred = model.predict(X_test)
        test_metrics = metrics(y_test_raw.clip(lower=0, upper=cap), y_pred)
        test_metrics_raw = metrics(y_test_raw, y_pred)

        with mlflow.start_run(run_id=search["run_id"]):
            mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
            mlflow.log_metrics({f"test_sin_recorte_{k}": v for k, v in test_metrics_raw.items()})
            model_info = log_pipeline(model, X_train)

        tags = {**search["params"], "modelo": "HistGradientBoostingRegressor",
                "etl_month": month, "mae_test": test_metrics_raw["mae"], "origen": "train_dag"}
        version = create_version(mlflow.MlflowClient(), model_info.model_uri, search["run_id"],
                                 tags, description=f"Optuna desde Airflow, entrenado con {month}")
        print(f"Versión {version.version}: MAE de test {test_metrics_raw['mae']:.4f} "
              f"({test_metrics['mae']:.4f} con el target recortado)")
        return {"version": int(version.version), "mae": float(test_metrics_raw["mae"])}

    @task(task_id="promote")
    def promote(registered: dict, month: str) -> None:
        """
        Pasa la versión a ``champion`` si no hay uno o si lo mejora en el test del mes.

        :param registered: Salida de ``train_and_register``.
        :param month: Mes con el que se entrenó (YYYY-MM).
        """
        import awswrangler as wr
        import mlflow
        from airflow.models import Variable
        from taxi_common.data import FEATURES, TARGET, dataset_path
        from taxi_tip.training import promote_if_better

        test = wr.s3.read_parquet(dataset_path(month, "test.parquet"))
        min_improvement = float(Variable.get("taxi_min_improvement", default_var=0.01))
        print(promote_if_better(mlflow.MlflowClient(), registered["version"], registered["mae"],
                                test[FEATURES].astype("float64"), test[TARGET],
                                min_improvement=min_improvement))

    month = "{{ params.month }}"
    search = search_hyperparameters(month)
    run_etl >> search
    promote(train_and_register(search, month), month)


dag = train_taxi_tip_model()
