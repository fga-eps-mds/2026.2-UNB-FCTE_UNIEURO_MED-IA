#!/usr/bin/env python
"""Mostra de qual registro do MLflow saiu um arquivo de pesos, pelo hash do arquivo.

Uso, na pasta treino:
    python scripts/localizar_pesos.py ../model.pt
"""

import argparse
from pathlib import Path

from mci.tracking import ORIGIN_TAG, runs_with_weights


def main() -> None:
    leitor = argparse.ArgumentParser(description=__doc__)
    leitor.add_argument("pesos", type=Path)
    opcoes = leitor.parse_args()

    registros = runs_with_weights(opcoes.pesos)
    if not registros:
        raise SystemExit(f"Nenhum registro com os pesos de {opcoes.pesos}")
    for registro in registros:
        print(
            f"run {registro.info.run_id}, experimento {registro.info.experiment_id}, "
            f"semente {registro.data.params.get('seed')}, origem {registro.data.tags.get(ORIGIN_TAG)}, "
            f"AUC de teste {registro.data.metrics.get('test_auc')}"
        )


if __name__ == "__main__":
    main()
