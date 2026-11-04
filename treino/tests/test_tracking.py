import json
import math
import os
import uuid
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest
import torch
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from PIL import Image

import registrar_treinos_existentes as registro
from mci import run as run_module
from mci import tracking
from mci.run import run_single_split

PATIENT_ID = "paciente-ficticio-0042"
TEST_METRICS = {"accuracy": 0.75, "f1": 0.5, "auc": 0.8, "loss": 0.4}
RESULT = {"seed": 1, "n_train": 10, "n_test": 3, "test_metrics": TEST_METRICS}
HISTORY = [{"epoch": 0, "train_loss": 0.9, "val_auc": 0.6}, {"epoch": 1, "train_loss": 0.7, "val_auc": 0.7}]


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.head = torch.nn.Linear(3, 2)

    def forward(self, images):
        return self.head(sum(image.mean(dim=(2, 3)) for image in images))


@pytest.fixture(scope="session")
def tracking_uri(tmp_path_factory):
    return f"sqlite:///{(tmp_path_factory.mktemp('mlflow') / 'mlflow.db').as_posix()}"


@pytest.fixture
def cfg(tmp_path, monkeypatch, tracking_uri):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "label.csv").write_text(f"patient_id,MoCA_score\n{PATIENT_ID},20\n")
    return {
        "experiment_name": uuid.uuid4().hex,
        "output_dir": "outputs/teste",
        "data": {"label_csv": "label.csv", "tasks": ["clock"]},
        "training": {"epochs": 2, "lr": 1.0e-5},
    }


def write_training_files(save_dir: Path, result: dict = RESULT, weights: bytes = b"pesos") -> Path:
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "model.pt").write_bytes(weights)
    (save_dir / "history.json").write_text(json.dumps(HISTORY))
    (save_dir / "result.json").write_text(json.dumps(result))
    return save_dir


def finished_run(run_id: str):
    return MlflowClient().get_run(run_id)


def fail_to_log(*_args, **_kwargs):
    raise MlflowException("banco indisponível")


def artifact_paths(run_id: str, folder: str | None = None) -> set[str]:
    found = set()
    for artifact in MlflowClient().list_artifacts(run_id, folder):
        found |= artifact_paths(run_id, artifact.path) if artifact.is_dir else {artifact.path}
    return found


def write_fake_dataset(root: Path, patients: int = 24) -> None:
    ids = [f"p{index:03d}" for index in range(patients)]
    scores = [20 if index % 2 else 28 for index in range(patients)]
    pd.DataFrame({"patient_id": ids, "MoCA_score": scores}).to_csv(root / "label.csv", index=False)
    for patient in ids:
        (root / "images" / patient).mkdir(parents=True)
        Image.fromarray(np.full((8, 8, 3), 200, dtype=np.uint8)).save(root / "images" / patient / "clock.png")


def training_cfg(cfg: dict) -> dict:
    return {
        **cfg,
        "data": {
            **cfg["data"], "images_dir": "images", "moca_cutoff": 25, "soft_label_center": 24.5,
            "train_frac": 0.5, "val_frac": 0.25, "test_frac": 0.25, "num_workers": 0,
        },
        "augmentation": {"pad": 1, "crop_size": 8},
        "training": {"epochs": 2, "batch_size": 4, "lr": 1.0e-3},
    }


def test_telemetry_is_disabled():
    assert os.environ["MLFLOW_DISABLE_TELEMETRY"] == "true"


def test_seed_run_records_parameters_config_and_data_hash(cfg):
    with tracking.seed_run(cfg, seed=3):
        run_id = tracking.current_run_id()

    run = finished_run(run_id)
    assert run.info.status == "FINISHED"
    assert run.data.params["seed"] == "3"
    assert run.data.params["training.lr"] == "1e-05"
    assert run.data.params["data.label_csv_sha256"] == tracking.file_sha256("label.csv")
    assert run.data.tags[tracking.ORIGIN_TAG] == tracking.ORIGIN_TRAINING
    assert run.data.tags[tracking.GIT_DIRTY_TAG] in {"true", "false", tracking.UNKNOWN}
    assert {"config.yaml", "environment.txt"} <= artifact_paths(run_id)


def test_each_epoch_is_recorded_on_its_own_step(cfg):
    with tracking.seed_run(cfg, seed=0):
        run_id = tracking.current_run_id()
        for entry in HISTORY:
            tracking.log_epoch(entry)

    series = MlflowClient().get_metric_history(run_id, "train_loss")
    assert [(point.step, point.value) for point in series] == [(0, 0.9), (1, 0.7)]
    assert "epoch" not in finished_run(run_id).data.metrics


def test_result_links_the_weights_to_the_run(cfg, tmp_path):
    save_dir = write_training_files(tmp_path / "seed_1")
    with tracking.seed_run(cfg, seed=1):
        run_id = tracking.current_run_id()
        tracking.log_result(RESULT, save_dir)

    run = finished_run(run_id)
    assert run.data.metrics["test_auc"] == TEST_METRICS["auc"]
    assert run.data.params["n_train"] == "10"
    assert run.data.tags[tracking.WEIGHTS_HASH_TAG] == tracking.file_sha256(save_dir / "model.pt")
    assert {"history.json", "result.json", "weights/model.pt"} <= artifact_paths(run_id)
    assert [found.info.run_id for found in tracking.runs_with_weights(save_dir / "model.pt")] == [run_id]


def test_weights_can_stay_out_of_the_artifacts_and_keep_the_hash(cfg, tmp_path):
    save_dir = write_training_files(tmp_path / "seed_1", weights=b"sem copia")
    with tracking.seed_run(cfg, seed=1):
        run_id = tracking.current_run_id()
        tracking.log_result(RESULT, save_dir, log_weights=False)

    assert "weights/model.pt" not in artifact_paths(run_id)
    assert finished_run(run_id).data.tags[tracking.WEIGHTS_HASH_TAG] == tracking.file_sha256(save_dir / "model.pt")


def test_experiment_nests_the_seeds_and_records_mean_and_std(cfg):
    with tracking.experiment_run(cfg, [0, 1]):
        parent_id = tracking.current_run_id()
        for seed in [0, 1]:
            with tracking.seed_run(cfg, seed):
                child_id = tracking.current_run_id()
        tracking.log_summary([{"auc": 0.7}, {"auc": 0.9}])

    assert finished_run(child_id).data.tags["mlflow.parentRunId"] == parent_id
    metrics = finished_run(parent_id).data.metrics
    assert metrics["mean_test_auc"] == pytest.approx(0.8)
    assert metrics["std_test_auc"] == pytest.approx(0.1)


def test_undefined_auc_of_a_single_class_split_is_recorded(cfg, tmp_path):
    result = {**RESULT, "test_metrics": {"auc": float("nan")}}
    save_dir = write_training_files(tmp_path / "seed_1", result=result, weights=b"nan")
    with tracking.seed_run(cfg, seed=1):
        run_id = tracking.current_run_id()
        tracking.log_result(result, save_dir)

    assert math.isnan(finished_run(run_id).data.metrics["test_auc"])


def test_failure_during_training_marks_the_run_as_failed(cfg):
    with pytest.raises(RuntimeError):
        with tracking.seed_run(cfg, seed=0):
            run_id = tracking.current_run_id()
            raise RuntimeError("falha no treino")

    assert finished_run(run_id).info.status == "FAILED"


def test_tracker_failure_on_an_epoch_warns_and_lets_training_continue(cfg, monkeypatch):
    with tracking.seed_run(cfg, seed=0):
        monkeypatch.setattr(mlflow, "log_metrics", fail_to_log)
        with pytest.warns(UserWarning, match="Época 0 sem registro"):
            tracking.log_epoch(HISTORY[0])


def test_tracker_failure_on_the_result_stops_with_an_error(cfg, tmp_path, monkeypatch):
    save_dir = write_training_files(tmp_path / "seed_1", weights=b"erro")
    with pytest.raises(MlflowException):
        with tracking.seed_run(cfg, seed=0):
            monkeypatch.setattr(mlflow, "log_metrics", fail_to_log)
            tracking.log_result(RESULT, save_dir)


def test_epoch_missed_during_training_is_completed_from_the_history_at_the_end(cfg, tmp_path, monkeypatch):
    save_dir = write_training_files(tmp_path / "seed_1", weights=b"completa")
    with tracking.seed_run(cfg, seed=1):
        run_id = tracking.current_run_id()
        with monkeypatch.context() as patch:
            patch.setattr(mlflow, "log_metrics", fail_to_log)
            with pytest.warns(UserWarning, match="Época 0 sem registro"):
                tracking.log_epoch(HISTORY[0])
        tracking.log_epoch(HISTORY[1])
        tracking.log_result(RESULT, save_dir)

    series = MlflowClient().get_metric_history(run_id, "train_loss")
    assert sorted((point.step, point.value) for point in series) == [(0, 0.9), (1, 0.7)]
    assert finished_run(run_id).info.status == "FINISHED"


def test_epoch_still_missing_at_the_end_marks_the_run_as_failed(cfg, tmp_path, monkeypatch):
    log_metrics = mlflow.log_metrics

    def fail_only_on_the_first_epoch(metrics, step=None, **kwargs):
        if step == HISTORY[0]["epoch"]:
            fail_to_log()
        log_metrics(metrics, step=step, **kwargs)

    save_dir = write_training_files(tmp_path / "seed_1", weights=b"buraco")
    with pytest.raises(MlflowException), tracking.seed_run(cfg, seed=1):
        run_id = tracking.current_run_id()
        monkeypatch.setattr(mlflow, "log_metrics", fail_only_on_the_first_epoch)
        with pytest.warns(UserWarning, match="Época 0 sem registro"):
            tracking.log_epoch(HISTORY[0])
        tracking.log_result(RESULT, save_dir)

    assert finished_run(run_id).info.status == "FAILED"


def test_nothing_is_recorded_without_an_open_run(cfg, tmp_path):
    save_dir = write_training_files(tmp_path / "seed_1", weights=b"sem run")
    tracking.log_device("cpu")
    tracking.log_epoch(HISTORY[0])
    tracking.log_result(RESULT, save_dir)

    assert tracking.current_run_id() is None
    assert mlflow.active_run() is None
    assert tracking.runs_with_weights(save_dir / "model.pt") == []


def test_default_store_keeps_records_and_artifacts_under_outputs(cfg, tmp_path, monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI")
    with tracking.seed_run(cfg, seed=0):
        artifact_uri = mlflow.active_run().info.artifact_uri

    assert (tmp_path / "outputs" / "mlflow" / "mlflow.db").is_file()
    assert artifact_uri.startswith(tracking.local_artifact_uri())
    assert not (tmp_path / "mlruns").exists()


def test_training_records_every_epoch_and_the_weights(cfg, tmp_path, monkeypatch):
    write_fake_dataset(tmp_path)
    monkeypatch.setattr(run_module, "build_model", lambda _cfg: TinyModel())
    config = training_cfg(cfg)

    with tracking.seed_run(config, seed=0):
        run_id = tracking.current_run_id()
        run_single_split(config, 0, tmp_path / "saida")

    assert len(MlflowClient().get_metric_history(run_id, "val_auc")) == config["training"]["epochs"]
    assert finished_run(run_id).data.params["n_train"] == "12"
    assert finished_run(run_id).data.tags["device"]
    assert json.loads((tmp_path / "saida" / "result.json").read_text())["mlflow_run_id"] == run_id
    assert "weights/model.pt" in artifact_paths(run_id)


def test_training_records_no_patient_identifier(cfg, tmp_path, monkeypatch):
    write_fake_dataset(tmp_path)
    monkeypatch.setattr(run_module, "build_model", lambda _cfg: TinyModel())
    config = training_cfg(cfg)

    with tracking.seed_run(config, seed=0):
        run_id = tracking.current_run_id()
        run_single_split(config, 0, tmp_path / "saida")

    run = finished_run(run_id)
    records = json.dumps([run.data.params, run.data.tags, sorted(artifact_paths(run_id))])
    downloaded = Path(mlflow.artifacts.download_artifacts(run_id=run_id, dst_path=str(tmp_path / "baixados")))
    texts = [path.read_text() for path in downloaded.rglob("*") if path.is_file() and path.suffix != ".pt"]
    assert "p001" not in records
    assert not [text for text in texts if "p001" in text]


def test_history_keeps_the_finished_epochs_when_training_stops_midway(cfg, tmp_path, monkeypatch):
    write_fake_dataset(tmp_path)
    monkeypatch.setattr(run_module, "build_model", lambda _cfg: TinyModel())
    train_one_epoch = run_module.train_one_epoch
    finished_epochs = 2
    calls = []

    def stop_after_the_finished_epochs(*args, **kwargs):
        calls.append(args)
        if len(calls) > finished_epochs:
            raise RuntimeError("queda no meio do treino")
        return train_one_epoch(*args, **kwargs)

    monkeypatch.setattr(run_module, "train_one_epoch", stop_after_the_finished_epochs)
    config = training_cfg(cfg)
    config["training"]["epochs"] = finished_epochs + 1

    with pytest.raises(RuntimeError, match="queda no meio do treino"):
        run_single_split(config, 0, tmp_path / "saida")

    history = json.loads((tmp_path / "saida" / "history.json").read_text())
    assert [entry["epoch"] for entry in history] == list(range(finished_epochs))
    assert not (tmp_path / "saida" / "history.json.tmp").exists()


def test_smoke_test_path_opens_no_run(cfg, tmp_path, monkeypatch):
    write_fake_dataset(tmp_path)
    monkeypatch.setattr(run_module, "build_model", lambda _cfg: TinyModel())

    metrics = run_single_split(training_cfg(cfg), 0, tmp_path / "saida")

    assert set(metrics) == set(TEST_METRICS)
    assert mlflow.active_run() is None
    assert json.loads((tmp_path / "saida" / "result.json").read_text())["mlflow_run_id"] is None


def test_existing_trainings_are_registered_as_imported(cfg, tmp_path):
    pasta = tmp_path / "outputs" / "teste"
    write_training_files(pasta / "seed_0", weights=b"importado 0")
    write_training_files(pasta / "seed_1", weights=b"importado 1")

    sementes = registro.sementes_da_pasta(pasta)
    registro.conferir_treinos(pasta, sementes)
    registro.registrar(cfg, pasta, sementes)

    found = tracking.runs_with_weights(pasta / "seed_1" / "model.pt")
    assert sementes == [0, 1]
    assert len(found) == 1
    assert found[0].data.tags[tracking.ORIGIN_TAG] == tracking.ORIGIN_IMPORTED
    assert found[0].data.tags[tracking.GIT_COMMIT_TAG] == tracking.UNKNOWN
    assert len(MlflowClient().get_metric_history(found[0].info.run_id, "train_loss")) == len(HISTORY)
    parent = finished_run(found[0].data.tags["mlflow.parentRunId"])
    assert parent.data.metrics["mean_test_auc"] == TEST_METRICS["auc"]


def test_registering_the_same_weights_twice_is_refused(cfg, tmp_path):
    pasta = tmp_path / "outputs" / "teste"
    write_training_files(pasta / "seed_0", weights=b"duplicado")
    registro.registrar(cfg, pasta, [0])

    with pytest.raises(SystemExit, match="já estão no registro"):
        registro.conferir_treinos(pasta, [0])


@pytest.mark.parametrize(
    ("arquivo_removido", "mensagem"),
    [("model.pt", "Faltam model.pt"), ("history.json", "Faltam history.json")],
)
def test_incomplete_training_is_refused_before_registering(cfg, tmp_path, arquivo_removido, mensagem):
    pasta = tmp_path / "outputs" / "teste"
    write_training_files(pasta / "seed_0", weights=b"incompleto")
    (pasta / "seed_0" / arquivo_removido).unlink()

    with pytest.raises(SystemExit, match=mensagem):
        registro.conferir_treinos(pasta, [0])


def test_empty_folder_is_refused(cfg, tmp_path):
    with pytest.raises(SystemExit, match="Nenhum treino"):
        registro.conferir_treinos(tmp_path, [])


@pytest.mark.parametrize(
    ("config", "flat"),
    [
        ({"a": 1}, {"a": 1}),
        ({"data": {"tasks": ["clock"], "extra": {"pad": 2}}}, {"data.tasks": ["clock"], "data.extra.pad": 2}),
        ({}, {}),
    ],
)
def test_flatten_config(config, flat):
    assert tracking.flatten_config(config) == flat
