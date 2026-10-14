"""
API REST que sirve el modelo de propinas de Yellow Taxi NYC.

Carga ``taxi_tip_model@champion`` desde MLflow. Como el modelo registrado es el pipeline
completo, la API recibe directamente las 11 columnas del ETL.

Cada ``MODEL_REFRESH_SECONDS`` segundos se fija en segundo plano si cambió el champion y, si
cambió, lo recarga, aunque no lleguen requests.
"""
import asyncio
import logging
import os
import threading
from contextlib import asynccontextmanager
from typing import Literal

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from mlflow import MlflowClient
from pydantic import BaseModel, Field, model_validator

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "taxi_tip_model")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "champion")
MODEL_REFRESH_SECONDS = float(os.getenv("MODEL_REFRESH_SECONDS", "60"))

# Mismo orden de columnas que el ETL y que la firma del modelo en MLflow
FEATURES = ["trip_distance", "duration_min", "fare_amount", "recargos", "PULocationID",
            "DOLocationID", "passenger_count", "hour", "dow", "RatecodeID", "VendorID"]

logger = logging.getLogger("uvicorn.error")


class ModelState:
    """
    Modelo cargado en memoria y la versión del registro de la que salió.

    Guarda los dos juntos en una tupla, así cada request lee un par consistente aunque otro
    thread esté cargando un champion nuevo.
    """

    def __init__(self):
        self.current: tuple[object, int] | None = None
        self._lock = threading.Lock()

    def refresh(self) -> None:
        """
        Carga el champion si todavía no hay modelo o si cambió la versión en el registro.

        Si ya hay otro chequeo en curso, no hace nada. Si MLflow no responde o no hay champion,
        deja el modelo que estaba (o ninguno) y lo anota en el log.
        """
        if not self._lock.acquire(blocking=False):
            return
        try:
            latest = int(MlflowClient().get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS).version)
            if self.current is None or self.current[1] != latest:
                model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{latest}")
                self.current = (model, latest)
                logger.info("Cargado %s versión %s (%s)", MODEL_NAME, latest, MODEL_ALIAS)
        except Exception as error:  # noqa: BLE001 - la API tiene que seguir viva igual
            logger.warning("No se pudo actualizar el modelo desde MLflow: %s", error)
        finally:
            self._lock.release()


state = ModelState()


async def refresh_periodically() -> None:
    """Revisa el registro cada ``MODEL_REFRESH_SECONDS`` segundos, en un thread aparte."""
    while True:
        await asyncio.sleep(MODEL_REFRESH_SECONDS)
        await asyncio.to_thread(state.refresh)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carga el champion al arrancar y deja corriendo el chequeo periódico."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    await asyncio.to_thread(state.refresh)
    refresher = asyncio.create_task(refresh_periodically())
    yield
    refresher.cancel()


class TripFeatures(BaseModel):
    """
    Datos de un viaje pagado con tarjeta: las mismas 11 columnas que deja el ETL.

    Los rangos de distancia, duración, tarifa mínima, velocidad y tarifa por milla son los
    filtros de limpieza del ETL (viajes fuera de ahí no se usaron para entrenar). Los demás
    topes son controles de sentido común para cortar datos mal cargados.
    """

    trip_distance: float = Field(gt=0, le=100, description="Distancia del viaje en millas.")
    duration_min: float = Field(gt=0, le=180, description="Duración del viaje en minutos.")
    fare_amount: float = Field(ge=2.5, le=1000, description="Tarifa del taxímetro en USD.")
    recargos: float = Field(
        ge=0, le=500,
        description="Suma de extras, impuestos, peajes, recargo de mejora, congestión, tasa "
                    "de aeropuerto y recargo CBD en USD (todo lo que se paga además de la tarifa).",
    )
    PULocationID: int = Field(ge=1, le=265, description="Zona TLC de origen (1 a 265).")
    DOLocationID: int = Field(ge=1, le=265, description="Zona TLC de destino (1 a 265).")
    passenger_count: int = Field(ge=0, le=9, description="Cantidad de pasajeros declarada.")
    hour: int = Field(ge=0, le=23, description="Hora del inicio del viaje (0 a 23).")
    dow: int = Field(ge=0, le=6, description="Día de la semana del inicio (0 = lunes, 6 = domingo).")
    RatecodeID: Literal[1, 2, 3, 4, 5, 6, 99] = Field(
        description="Tarifa: 1 estándar, 2 JFK, 3 Newark, 4 Nassau/Westchester, 5 negociada, "
                    "6 viaje grupal, 99 desconocida.",
    )
    VendorID: Literal[1, 2, 6, 7] = Field(
        description="Proveedor del registro: 1 Creative Mobile Technologies, 2 Curb Mobility, "
                    "6 Myle Technologies, 7 Helix.",
    )

    @model_validator(mode="after")
    def check_consistency(self):
        """
        Rechaza los viajes que el ETL descarta por inconsistentes.

        - Velocidad media mayor a 100 mph.
        - Tarifa estándar (``RatecodeID`` 1) menor a USD 2,50 por milla.
        """
        if self.trip_distance / (self.duration_min / 60) > 100:
            raise ValueError("La velocidad media supera 100 mph: revisar distancia y duración.")
        if self.RatecodeID == 1 and self.fare_amount < 2.5 * self.trip_distance:
            raise ValueError("Con tarifa estándar (RatecodeID 1) la tarifa no puede ser menor a "
                             "USD 2,50 por milla: revisar tarifa y distancia.")
        return self

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "trip_distance": 3.2, "duration_min": 15.0, "fare_amount": 18.4,
                "recargos": 5.75, "PULocationID": 161, "DOLocationID": 236,
                "passenger_count": 1, "hour": 18, "dow": 2, "RatecodeID": 1, "VendorID": 2,
            }]
        }
    }


class PredictRequest(BaseModel):
    """Cuerpo de ``POST /predict``."""

    features: TripFeatures


class PredictBatchRequest(BaseModel):
    """Cuerpo de ``POST /predict/batch``: hasta 1000 viajes."""

    trips: list[TripFeatures] = Field(min_length=1, max_length=1000,
                                      description="Viajes a predecir (entre 1 y 1000).")


class TipPrediction(BaseModel):
    """Propina estimada para un viaje."""

    tip_amount: float = Field(description="Propina estimada en USD.")
    model_version: int = Field(description="Versión de taxi_tip_model que hizo la predicción.")

    model_config = {"json_schema_extra": {"examples": [{"tip_amount": 4.81, "model_version": 1}]}}


class TipBatchPrediction(BaseModel):
    """Propinas estimadas, en el mismo orden que los viajes recibidos."""

    tip_amounts: list[float] = Field(description="Propinas estimadas en USD.")
    model_version: int = Field(description="Versión de taxi_tip_model que hizo la predicción.")

    model_config = {"json_schema_extra": {
        "examples": [{"tip_amounts": [4.81, 14.79], "model_version": 1}]}}


class APIInfo(BaseModel):
    """Información del servicio y del modelo cargado."""

    name: str = Field(description="Nombre del servicio.")
    description: str = Field(description="Qué estima el servicio.")
    model_name: str = Field(description="Modelo del registro de MLflow que se sirve.")
    model_alias: str = Field(description="Alias del registro que se sigue (champion).")
    model_version: int | None = Field(description="None si todavía no se pudo cargar un modelo.")


class ErrorMessage(BaseModel):
    """Error devuelto por la API."""

    detail: str


app = FastAPI(
    title="Taxi Tip API",
    description="Estima la propina (USD) de viajes de Yellow Taxi NYC pagados con tarjeta. "
                "Sirve el modelo champion registrado en MLflow. Mientras no haya un champion "
                "(antes de correr la notebook del experimento) los endpoints de predicción "
                "responden 503.",
    version="1.0.0",
    lifespan=lifespan,
)

PREDICT_RESPONSES = {
    422: {"description": "Algún campo fuera de rango, o un viaje que el ETL descarta por "
                         "inconsistente (más de 100 mph, o tarifa estándar menor a USD 2,50 "
                         "por milla)."},
    503: {"model": ErrorMessage, "description": "Todavía no hay un champion en MLflow."},
}


def _predict(trips: list[TripFeatures]) -> tuple[list[float], int]:
    """
    Corre el modelo sobre una lista de viajes.

    :param trips: Viajes validados.
    :return: Propinas en USD (redondeadas a centavos y nunca negativas) y la versión usada.
    """
    current = state.current
    if current is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"No hay un modelo '{MODEL_NAME}@{MODEL_ALIAS}' disponible en MLflow.",
        )
    model, version = current
    X = pd.DataFrame([t.model_dump() for t in trips], columns=FEATURES).astype("float64")
    return [round(max(float(p), 0.0), 2) for p in model.predict(X)], version


@app.get("/", response_model=APIInfo)
def read_root() -> APIInfo:
    """Devuelve el nombre del servicio y qué versión del modelo está sirviendo."""
    return APIInfo(
        name="Taxi Tip API",
        description="Propina estimada para viajes de Yellow Taxi NYC pagados con tarjeta.",
        model_name=MODEL_NAME,
        model_alias=MODEL_ALIAS,
        model_version=state.current[1] if state.current else None,
    )


@app.post("/predict", response_model=TipPrediction, responses=PREDICT_RESPONSES)
def predict(request: PredictRequest) -> TipPrediction:
    """Estima la propina de un viaje."""
    tips, version = _predict([request.features])
    return TipPrediction(tip_amount=tips[0], model_version=version)


@app.post("/predict/batch", response_model=TipBatchPrediction, responses=PREDICT_RESPONSES)
def predict_batch(request: PredictBatchRequest) -> TipBatchPrediction:
    """Estima la propina de hasta 1000 viajes en una sola llamada."""
    tips, version = _predict(request.trips)
    return TipBatchPrediction(tip_amounts=tips, model_version=version)
