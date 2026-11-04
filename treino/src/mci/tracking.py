"""Registro dos treinos no MLflow (#22).

Só entram no registro números, configuração, hash dos dados e pesos. Desenho, rótulo e
identificador de paciente são dado clínico, por isso não se usa autolog nem log_input,
que podem guardar amostras da entrada.
"""
from __future__ import annotations

import hashlib
import os
import platform
import subprocess
import warnings
from contextlib import contextmanager
from functools import wraps
from importlib import metadata
from pathlib import Path
from typing import Any, Callable, Iterator

# O MLflow envia telemetria de uso por padrão, e nada do treino deve sair da máquina sem
# decisão do time. A variável precisa estar definida antes do import.
os.environ["MLFLOW_DISABLE_TELEMETRY"] = "true"

import mlflow  # noqa: E402
import numpy as np  # noqa: E402
from mlflow.exceptions import MlflowException  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402

# Dentro de outputs/, que o .gitignore já cobre, para os registros não irem para o Git.
LOCAL_STORE = Path("outputs/mlflow")
HASH_CHUNK_BYTES = 1024 * 1024

ORIGIN_TAG = "registro"
ORIGIN_TRAINING = "treino"
ORIGIN_IMPORTED = "importado"
GIT_COMMIT_TAG = "mlflow.source.git.commit"
GIT_DIRTY_TAG = "git.dirty"
WEIGHTS_HASH_TAG = "weights_sha256"
UNKNOWN = "desconhecido"


def local_store_uri() -> str:
    return f"sqlite:///{(LOCAL_STORE / 'mlflow.db').resolve().as_posix()}"


def local_artifact_uri() -> str:
    return (LOCAL_STORE / "artifacts").resolve().as_uri()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        while chunk := source.read(HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def flatten_config(cfg: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat = {}
    for key, value in cfg.items():
        if isinstance(value, dict):
            flat.update(flatten_config(value, f"{prefix}{key}."))
        else:
            flat[f"{prefix}{key}"] = value
    return flat


def _use_tracking_store() -> str:
    uri = os.environ.get("MLFLOW_TRACKING_URI") or local_store_uri()
    if uri == local_store_uri():
        LOCAL_STORE.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(uri)
    return uri


def _activate_experiment(name: str) -> None:
    uri = _use_tracking_store()
    client = MlflowClient()
    # Sem local explícito, o MLflow grava os arquivos em ./mlruns. Com um servidor, quem decide é ele.
    if uri == local_store_uri() and client.get_experiment_by_name(name) is None:
        client.create_experiment(name, artifact_location=f"{local_artifact_uri()}/{name}")
    mlflow.set_experiment(name)


def _git_is_dirty() -> str:
    try:
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=Path(__file__).parent, capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return UNKNOWN
    return str(bool(status.stdout.strip())).lower()


def _environment_snapshot() -> str:
    packages = sorted(f"{dist.metadata['Name']}=={dist.version}" for dist in metadata.distributions())
    return "\n".join([f"python=={platform.python_version()}", *packages])


def _log_provenance(cfg: dict[str, Any], imported: bool) -> None:
    mlflow.log_params(flatten_config(cfg))
    mlflow.log_param("data.label_csv_sha256", file_sha256(cfg["data"]["label_csv"]))
    mlflow.log_dict(cfg, "config.yaml")
    if imported:
        # O commit e o ambiente de hoje não são os do treino, que aconteceu antes do registro.
        mlflow.set_tags({ORIGIN_TAG: ORIGIN_IMPORTED, GIT_COMMIT_TAG: UNKNOWN})
        return
    mlflow.set_tags({ORIGIN_TAG: ORIGIN_TRAINING, GIT_DIRTY_TAG: _git_is_dirty()})
    mlflow.log_text(_environment_snapshot(), "environment.txt")


@contextmanager
def experiment_run(cfg: dict[str, Any], seeds: list[int], imported: bool = False) -> Iterator[None]:
    _activate_experiment(cfg["experiment_name"])
    with mlflow.start_run(run_name=f"{len(seeds)}-seeds"):
        _log_provenance(cfg, imported)
        mlflow.log_param("seeds", seeds)
        yield


@contextmanager
def seed_run(cfg: dict[str, Any], seed: int, imported: bool = False) -> Iterator[None]:
    is_nested = mlflow.active_run() is not None
    if not is_nested:
        _activate_experiment(cfg["experiment_name"])
    # O uso de memória da GPU importa porque, ao estourar, o treino fica 35x mais lento sem erro
    # (TRAINING.md, seção 3). Num treino importado não há o que medir.
    with mlflow.start_run(run_name=f"seed-{seed}", nested=is_nested, log_system_metrics=not imported):
        _log_provenance(cfg, imported)
        mlflow.log_param("seed", seed)
        yield


def _only_inside_run(log: Callable[..., None]) -> Callable[..., None]:
    # O smoke_test chama run_single_split sem run aberto, e o MLflow abriria um por conta própria.
    @wraps(log)
    def log_if_run_is_open(*args: Any, **kwargs: Any) -> None:
        if mlflow.active_run() is not None:
            log(*args, **kwargs)

    return log_if_run_is_open


def current_run_id() -> str | None:
    run = mlflow.active_run()
    return run.info.run_id if run else None


@_only_inside_run
def log_device(device_name: str) -> None:
    mlflow.set_tag("device", device_name)


@_only_inside_run
def log_epoch(history_entry: dict[str, float]) -> None:
    metrics = {name: value for name, value in history_entry.items() if name != "epoch"}
    try:
        mlflow.log_metrics(metrics, step=history_entry["epoch"])
    except MlflowException as error:
        # A série também fica no history.json, então uma falha aqui não justifica perder o treino.
        warnings.warn(f"Época {history_entry['epoch']} sem registro no MLflow: {error}", stacklevel=2)


@_only_inside_run
def log_result(result: dict[str, Any], save_dir: str | Path, log_weights: bool = True) -> None:
    save_dir = Path(save_dir)
    weights = save_dir / "model.pt"
    mlflow.log_metrics({f"test_{name}": value for name, value in result["test_metrics"].items()})
    mlflow.log_params({name: value for name, value in result.items() if name.startswith("n_")})
    mlflow.set_tag(WEIGHTS_HASH_TAG, file_sha256(weights))
    mlflow.log_artifact(str(save_dir / "history.json"))
    mlflow.log_artifact(str(save_dir / "result.json"))
    if log_weights:
        mlflow.log_artifact(str(weights), "weights")


@_only_inside_run
def log_summary(per_seed_metrics: list[dict[str, float]]) -> None:
    for name in per_seed_metrics[0]:
        values = [metrics[name] for metrics in per_seed_metrics]
        mlflow.log_metrics({f"mean_test_{name}": float(np.mean(values)), f"std_test_{name}": float(np.std(values))})


def runs_with_weights(weights: str | Path) -> list[Any]:
    _use_tracking_store()
    return MlflowClient().search_runs(
        [experiment.experiment_id for experiment in MlflowClient().search_experiments()],
        filter_string=f"tags.{WEIGHTS_HASH_TAG} = '{file_sha256(weights)}'",
    )
