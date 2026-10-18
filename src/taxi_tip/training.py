"""
Entrenamiento del modelo de propinas: baselines, búsqueda de hiperparámetros y promoción.

Lo usan la notebook del experimento y el DAG ``train_taxi_tip_model``, para que los dos
entrenen y elijan el champion de la misma forma.
"""
import mlflow
import numpy as np
import optuna
import pandas as pd
from mlflow import MlflowClient
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold

from taxi_tip.features import RANDOM_STATE, build_pipeline
from taxi_tip.mlflow_utils import MODEL_NAME, load_champion

# Igual que en AMq1: se recorta la cola extrema de la propina (errores tipo USD 397)
TIP_CAP_QUANTILE = 0.995
# Mejora mínima del MAE para reemplazar al champion (la misma que usa el reentrenamiento)
MIN_IMPROVEMENT = 0.01


def metrics(y_true, y_pred) -> dict:
    """MAE, RMSE y R² de una predicción."""
    return {'mae': mean_absolute_error(y_true, y_pred),
            'rmse': root_mean_squared_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred)}


def tip_cap(y_train: pd.Series) -> float:
    """Valor en el que se recorta el target: el p99,5 del train."""
    return float(y_train.quantile(TIP_CAP_QUANTILE))


def compute_baselines(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame,
                      y_test: pd.Series) -> tuple[dict, float]:
    """
    Evalúa en test las dos referencias simples contra las que comparamos el modelo.

    - **Mediana**: siempre predice la mediana del train (el óptimo constante para MAE).
    - **% del monto**: el porcentaje mediano de propina del train (entre los viajes con propina)
      sobre ``fare_amount + recargos``; con tarifa 99 (desconocida) predice 0.

    :return: Métricas de cada baseline y el porcentaje de propina usado.
    """
    median = DummyRegressor(strategy='median').fit(X_train, y_train).predict(X_test)

    amount_train = X_train['fare_amount'] + X_train['recargos']
    amount_test = X_test['fare_amount'] + X_test['recargos']
    with_tip = y_train > 0
    tip_pct = float(np.median(y_train[with_tip] / amount_train[with_tip]))
    pct = np.where(X_test['RatecodeID'] == 99, 0.0, tip_pct * amount_test)

    return {'mediana': metrics(y_test, median), 'pct_monto': metrics(y_test, pct)}, tip_pct


def suggest_params(trial: optuna.Trial) -> dict:
    """Espacio de búsqueda de AMq1 para el ``HistGradientBoostingRegressor``."""
    return dict(
        max_depth=trial.suggest_int('max_depth', 4, 8),
        learning_rate=trial.suggest_float('learning_rate', 0.03, 0.30, log=True),
        max_iter=trial.suggest_int('max_iter', 100, 200, step=50),
        min_samples_leaf=trial.suggest_int('min_samples_leaf', 10, 200, log=True),
        l2_regularization=trial.suggest_float('l2_regularization', 1e-3, 10.0, log=True),
        max_leaf_nodes=trial.suggest_int('max_leaf_nodes', 15, 40),
    )


def search_hyperparameters(X_train: pd.DataFrame, y_train: pd.Series, experiment_id: str,
                           n_trials: int = 15, n_folds: int = 3) -> optuna.Study:
    """
    Busca hiperparámetros con Optuna, minimizando el MAE de validación cruzada sobre train.

    Hay que llamarla con el run padre activo: cada trial queda como run anidado. Con el
    ``MedianPruner``, si después de dos folds un trial viene peor que la mediana de los
    anteriores, se corta.

    :param X_train: Features de train (las 11 columnas del ETL).
    :param y_train: Target de train, ya recortado.
    :param experiment_id: Experimento de MLflow donde se registran los trials.
    :param n_trials: Cantidad de trials.
    :param n_folds: Folds de la validación cruzada.
    :return: El estudio de Optuna terminado.
    """
    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial)
        with mlflow.start_run(experiment_id=experiment_id, run_name=f'trial_{trial.number}',
                              nested=True):
            mlflow.log_params(params)
            folds = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
            maes = []
            for i, (idx_fit, idx_val) in enumerate(folds.split(X_train)):
                pipe = build_pipeline(**params)
                pipe.fit(X_train.iloc[idx_fit], y_train.iloc[idx_fit])
                maes.append(mean_absolute_error(y_train.iloc[idx_val],
                                                pipe.predict(X_train.iloc[idx_val])))
                trial.report(float(np.mean(maes)), i)
                if trial.should_prune():
                    mlflow.set_tag('estado', 'podado')
                    mlflow.log_metric('cv_mae', float(np.mean(maes)))
                    raise optuna.TrialPruned()
            cv_mae = float(np.mean(maes))
            mlflow.set_tag('estado', 'completo')
            mlflow.log_metric('cv_mae', cv_mae)
        return cv_mae

    def log_progress(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        # Corre con el run padre activo (el del trial ya se cerró): deja el historial de la
        # búsqueda como métricas por paso, que MLflow grafica solo
        if trial.state == optuna.trial.TrialState.COMPLETE:
            mlflow.log_metric('optuna_cv_mae', trial.value, step=trial.number)
            mlflow.log_metric('optuna_mejor_cv_mae', study.best_value, step=trial.number)

    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
    )
    study.optimize(objective, n_trials=n_trials, callbacks=[log_progress])
    return study


def log_training_report(model, X_test: pd.DataFrame, y_test: pd.Series,
                        sample_size: int = 10_000) -> None:
    """
    Registra en el run activo las curvas de entrenamiento y los gráficos de evaluación.

    - Curvas: el ``HistGradientBoostingRegressor`` separa un 10 % del train para validar en
      cada iteración (early stopping). El MAE de train y de esa validación por iteración
      queda como métricas por paso (``curva_mae_train`` y ``curva_mae_validacion``), que
      MLflow grafica y permite comparar entre runs, y también como imagen.
    - Gráficos sobre una muestra del test: predicción vs. real, distribución de los residuos e
      importancia por permutación de las 11 columnas de entrada (cuánto empeora el MAE al
      mezclar cada una).

    :param model: Pipeline ya ajustado.
    :param X_test: Features de test.
    :param y_test: Target de test, recortado igual que el de train.
    :param sample_size: Viajes de test que se usan para los gráficos.
    """
    import matplotlib.pyplot as plt
    from mlflow.entities import Metric
    from sklearn.inspection import permutation_importance

    run_id = mlflow.active_run().info.run_id
    hgb = model.named_steps['model']
    if hasattr(hgb, 'validation_score_') and len(hgb.validation_score_):
        # sklearn guarda los scores como -pérdida (más alto es mejor): los pasamos a MAE
        curves = {'curva_mae_train': -np.asarray(hgb.train_score_),
                  'curva_mae_validacion': -np.asarray(hgb.validation_score_)}
        timestamp = int(pd.Timestamp.now().timestamp() * 1000)
        MlflowClient().log_batch(run_id, metrics=[
            Metric(key, float(value), timestamp, step)
            for key, values in curves.items() for step, value in enumerate(values)])

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(curves['curva_mae_train'], label='Train')
        ax.plot(curves['curva_mae_validacion'], label='Validación (10 % del train)')
        ax.set_xlabel('Iteración (árboles)')
        ax.set_ylabel('MAE (USD)')
        ax.set_title('Curva de entrenamiento')
        ax.legend()
        fig.tight_layout()
        mlflow.log_figure(fig, 'graficos/curva_entrenamiento.png')
        plt.close(fig)

    sample = X_test.sample(min(sample_size, len(X_test)), random_state=RANDOM_STATE)
    y_sample = y_test.loc[sample.index]
    pred = model.predict(sample)
    cap = float(max(y_sample.max(), pred.max()))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_sample, pred, s=4, alpha=0.3)
    ax.plot([0, cap], [0, cap], color='black', linewidth=1)
    ax.set_xlabel('Propina real (USD)')
    ax.set_ylabel('Propina predicha (USD)')
    ax.set_title('Predicción vs. real (muestra de test)')
    fig.tight_layout()
    mlflow.log_figure(fig, 'graficos/prediccion_vs_real.png')
    plt.close(fig)

    residuals = y_sample - pred
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(residuals.clip(-10, 10), bins=80)
    ax.axvline(0, color='black', linewidth=1)
    ax.set_xlabel('Real - predicha (USD, recortado a ±10)')
    ax.set_ylabel('Viajes')
    ax.set_title('Distribución de los residuos (muestra de test)')
    fig.tight_layout()
    mlflow.log_figure(fig, 'graficos/residuos.png')
    plt.close(fig)

    importance = permutation_importance(model, sample, y_sample,
                                        scoring='neg_mean_absolute_error', n_repeats=3,
                                        random_state=RANDOM_STATE)
    importance = pd.Series(importance.importances_mean, index=sample.columns).sort_values()
    fig, ax = plt.subplots(figsize=(7, 5))
    importance.plot.barh(ax=ax)
    ax.set_xlabel('Aumento del MAE (USD)')
    ax.set_title('Importancia por permutación')
    fig.tight_layout()
    mlflow.log_figure(fig, 'graficos/importancia_permutacion.png')
    plt.close(fig)


def promote_if_better(client: MlflowClient, version: str, mae: float, X_test: pd.DataFrame,
                      y_test: pd.Series, min_improvement: float = MIN_IMPROVEMENT,
                      model_name: str = MODEL_NAME) -> str:
    """
    Decide si una versión recién registrada pasa a ``champion``.

    - Si no hay champion, pasa directo.
    - Si hay, se evalúa al champion con este mismo test. Si la versión nueva baja su MAE al
      menos ``min_improvement``, pasa a ``champion`` y el anterior queda como
      ``old_champion`` (un challenger viejo que hubiera quedado se descarta). Si no, queda
      como ``challenger``.

    :param client: Cliente de MLflow ya configurado.
    :param version: Versión registrada a evaluar.
    :param mae: MAE de esa versión en ``X_test`` / ``y_test``.
    :param X_test: Features de test.
    :param y_test: Propina real de test, sin recortar (igual que ``mae_test``).
    :param min_improvement: Mejora relativa mínima del MAE para reemplazar al champion.
    :param model_name: Nombre del modelo registrado.
    :return: Mensaje con la decisión.
    """
    version = str(version)
    champion = load_champion(model_name)
    if champion is None:
        client.set_registered_model_alias(model_name, 'champion', version)
        return f'No había champion: la versión {version} pasa a champion (MAE {mae:.4f})'

    champion_version, champion_model = champion
    if str(champion_version) == version:
        # Un intento anterior ya la promovió
        return f'La versión {version} ya es champion'
    champion_mae = mean_absolute_error(y_test, champion_model.predict(X_test))
    if mae < champion_mae * (1 - min_improvement):
        client.set_registered_model_alias(model_name, 'old_champion', str(champion_version))
        if 'challenger' in client.get_registered_model(model_name).aliases:
            client.delete_registered_model_alias(model_name, 'challenger')
        client.set_registered_model_alias(model_name, 'champion', version)
        return (f'Versión {version} es el nuevo champion: MAE {mae:.4f} vs {champion_mae:.4f} '
                f'de la v{champion_version} (queda como old_champion)')
    client.set_registered_model_alias(model_name, 'challenger', version)
    return (f'Versión {version} queda como challenger: MAE {mae:.4f} vs {champion_mae:.4f} '
            f'de la v{champion_version} (no mejora al menos {100 * min_improvement:g} %)')
