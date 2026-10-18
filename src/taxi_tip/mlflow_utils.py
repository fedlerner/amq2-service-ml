"""Funciones de apoyo para registrar el modelo de propinas en MLflow."""
from pathlib import Path

import mlflow
import skops.io as sio
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature

MODEL_NAME = 'taxi_tip_model'
MODEL_DESCRIPTION = 'Estima la propina (USD) de un viaje de Yellow Taxi NYC pagado con tarjeta.'


def get_or_create_experiment(experiment_name: str) -> str:
    """
    Devuelve el id del experimento y lo crea si no existe.

    :param experiment_name: Nombre del experimento en MLflow.
    :return: Id del experimento.
    """
    if experiment := mlflow.get_experiment_by_name(experiment_name):
        return experiment.experiment_id
    return mlflow.create_experiment(experiment_name)


def skops_trusted_types(model) -> list[str]:
    """
    Lista los tipos que skops no carga por defecto para este modelo.

    MLflow 3 guarda los modelos de sklearn con skops, que solo carga tipos marcados como
    confiables. Además de nuestro ``TaxiFeatureEngineer``, aparecen tipos internos de
    sklearn (los árboles del HistGradientBoosting, la pérdida, etc.). El modelo es nuestro,
    así que los marcamos todos.

    :param model: Pipeline ya ajustado.
    :return: Nombres completos de los tipos a confiar.
    """
    return sio.get_untrusted_types(data=sio.dumps(model))


def log_pipeline(model, X_train):
    """
    Loguea el pipeline en el run activo.

    Le pasa a MLflow ``code_paths`` con la carpeta de ``taxi_tip``, así el código del
    transformer queda dentro del artefacto y la API y Airflow lo pueden cargar sin instalar
    nada, y ``skops_trusted_types``, porque MLflow 3 guarda con skops. La firma y el
    ``input_example`` salen de las 11 columnas del ETL.

    :param model: Pipeline ya ajustado.
    :param X_train: Features de train, para la firma y el ejemplo.
    :return: ``ModelInfo`` del modelo logueado.
    """
    return mlflow.sklearn.log_model(
        model,
        name='model',
        signature=infer_signature(X_train, model.predict(X_train.head(100))),
        input_example=X_train.head(3),
        code_paths=[str(Path(__file__).parent)],
        skops_trusted_types=skops_trusted_types(model),
    )


def create_version(client: MlflowClient, model_uri: str, run_id: str, tags: dict,
                   description: str, model_name: str = MODEL_NAME):
    """
    Registra una versión nueva del modelo, sin alias.

    Si el modelo todavía no existe en el registro, lo crea. Cualquier otro error de MLflow
    (servidor caído, timeout) se propaga.

    :param client: Cliente de MLflow ya configurado.
    :param model_uri: URI del modelo logueado (``ModelInfo.model_uri``).
    :param run_id: Run que generó el modelo.
    :param tags: Tags de la versión (todos se guardan como string).
    :param description: Descripción de la versión.
    :param model_name: Nombre del modelo registrado.
    :return: La ``ModelVersion`` creada.
    """
    try:
        client.get_registered_model(model_name)
    except MlflowException as error:
        if error.error_code != 'RESOURCE_DOES_NOT_EXIST':
            raise
        client.create_registered_model(name=model_name, description=MODEL_DESCRIPTION)
    return client.create_model_version(
        name=model_name, source=model_uri, run_id=run_id,
        tags={k: str(v) for k, v in tags.items()}, description=description,
    )


def load_champion(model_name: str = MODEL_NAME):
    """
    Carga el champion actual del registro.

    Devuelve ``None`` solo si el modelo no está registrado o no tiene alias ``champion``.
    Cualquier otro error de MLflow (red, servidor caído) se propaga: si no, la notebook
    tomaría un error pasajero como "no hay champion" y promovería sin comparar.

    :param model_name: Nombre del modelo registrado.
    :return: Tupla ``(versión, modelo)``, o ``None`` si todavía no hay champion.
    """
    try:
        aliases = MlflowClient().get_registered_model(model_name).aliases
    except MlflowException as error:
        if error.error_code == 'RESOURCE_DOES_NOT_EXIST':
            return None
        raise
    if 'champion' not in aliases:
        return None
    version = aliases['champion']
    return int(version), mlflow.sklearn.load_model(f'models:/{model_name}/{version}')
