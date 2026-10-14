"""Consultas al registro de MLflow que comparten los DAGs."""
from airflow.exceptions import AirflowFailException
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException


def champion_version(client: MlflowClient, model_name: str, alias: str = "champion") -> str:
    """
    Devuelve la versión con el alias pedido.

    Si el modelo no está registrado o no tiene ese alias, falla sin reintentar y explica qué
    falta. Cualquier otro error de MLflow (red, servidor caído) se propaga, así Airflow
    reintenta la tarea.

    :param client: Cliente de MLflow ya configurado.
    :param model_name: Nombre del modelo registrado.
    :param alias: Alias a buscar.
    :return: Número de versión, como string.
    """
    try:
        aliases = client.get_registered_model(model_name).aliases
    except MlflowException as error:
        if error.error_code != "RESOURCE_DOES_NOT_EXIST":
            raise
        aliases = {}
    if alias not in aliases:
        raise AirflowFailException(
            f"No hay {alias} de {model_name}: primero hay que correr la notebook del experimento")
    return aliases[alias]


def drop_alias_if_points_to(client: MlflowClient, model_name: str, alias: str,
                            version: str) -> None:
    """
    Saca un alias solo si todavía apunta a esa versión.

    Si mientras tanto alguien lo movió a otra versión (por ejemplo, la notebook registró otro
    challenger), lo deja como está.

    :param client: Cliente de MLflow ya configurado.
    :param model_name: Nombre del modelo registrado.
    :param alias: Alias a sacar.
    :param version: Versión a la que tiene que apuntar para sacarlo.
    """
    if client.get_registered_model(model_name).aliases.get(alias) == str(version):
        client.delete_registered_model_alias(model_name, alias)
