#!/usr/bin/env python
"""Registra no MLflow os treinos que já estão em outputs/<experimento>/seed_<n> (#22).

O commit e o ambiente daquele treino não são recuperáveis, por isso o registro leva a marca
registro=importado. A configuração e o hash dos dados são os informados agora.

Uso, na pasta treino:
    python scripts/registrar_treinos_existentes.py --config configs/original.yaml
    python scripts/registrar_treinos_existentes.py --config configs/convnextv2.yaml --seeds 0 1
"""

import argparse
import json
from pathlib import Path

from mci import tracking
from mci.utils import load_config

ARQUIVOS_DO_TREINO = ("history.json", "result.json", "model.pt")


def sementes_da_pasta(pasta: Path) -> list[int]:
    return sorted(int(semente.name.removeprefix("seed_")) for semente in pasta.glob("seed_*"))


def conferir_treinos(pasta: Path, sementes: list[int]) -> None:
    if not sementes:
        raise SystemExit(f"Nenhum treino encontrado em {pasta}")
    for semente in sementes:
        pasta_da_semente = pasta / f"seed_{semente}"
        faltando = [nome for nome in ARQUIVOS_DO_TREINO if not (pasta_da_semente / nome).is_file()]
        if faltando:
            raise SystemExit(f"Faltam {', '.join(faltando)} em {pasta_da_semente}")
        registros = tracking.runs_with_weights(pasta_da_semente / "model.pt")
        if registros:
            raise SystemExit(f"Os pesos de {pasta_da_semente} já estão no registro {registros[0].info.run_id}")


def registrar(cfg: dict, pasta: Path, sementes: list[int]) -> None:
    metricas_por_semente = []
    with tracking.experiment_run(cfg, sementes, imported=True):
        for semente in sementes:
            pasta_da_semente = pasta / f"seed_{semente}"
            resultado = json.loads((pasta_da_semente / "result.json").read_text())
            with tracking.seed_run(cfg, semente, imported=True):
                # As épocas entram pelo history.json dentro de log_result, como no fim de um treino.
                tracking.log_result(resultado, pasta_da_semente, cfg.get("tracking", {}).get("log_weights", True))
            metricas_por_semente.append(resultado["test_metrics"])
        tracking.log_summary(metricas_por_semente)


def main() -> None:
    leitor = argparse.ArgumentParser(description=__doc__)
    leitor.add_argument("--config", required=True)
    leitor.add_argument("--seeds", type=int, nargs="*", default=None)
    opcoes = leitor.parse_args()

    cfg = load_config(opcoes.config)
    pasta = Path(cfg["output_dir"])
    sementes = opcoes.seeds if opcoes.seeds is not None else sementes_da_pasta(pasta)
    conferir_treinos(pasta, sementes)
    registrar(cfg, pasta, sementes)
    print(f"Registrados no experimento {cfg['experiment_name']}: sementes {sementes}")


if __name__ == "__main__":
    main()
