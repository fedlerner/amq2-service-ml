"""Funciones de apoyo para registrar el modelo de propinas en MLflow."""
import mlflow
import skops.io as sio
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

MODEL_NAME = 'taxi_tip_model'


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
