FROM apache/airflow:3.0.2
COPY requirements.txt /requirements.txt

USER root
RUN apt-get update \
  && apt-get install -y --no-install-recommends \
         git \
  && apt-get autoremove -yqq --purge \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

USER airflow
COPY requirements.txt /requirements.txt

RUN pip install --upgrade pip
RUN pip install --no-cache-dir apache-airflow[virtualenv]==3.0.2 -r /requirements.txt

# Los contenedores corren con AIRFLOW_UID (no siempre 50000) y grupo 0. El cache que crea el
# build en /tmp/.cache queda de airflow, así que lo abrimos al grupo para que uv pueda armar
# los virtualenvs de las tareas @task.virtualenv. Nuestros DAGs no las usan, pero los de la
# cátedra sí, y sin esto fallan con un AIRFLOW_UID distinto de 50000.
USER root
RUN mkdir -p /tmp/.cache && chmod -R g+rwX /tmp/.cache
USER airflow
