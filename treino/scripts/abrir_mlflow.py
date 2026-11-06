#!/usr/bin/env python
"""Abre a interface do MLflow com os registros locais do treino.

Uso, na pasta treino:
    python scripts/abrir_mlflow.py
    python scripts/abrir_mlflow.py --porta 5001
"""

import argparse
import subprocess
import sys

from mci.tracking import local_artifact_uri, local_store_uri

# Só esta máquina acessa: o MLflow não tem login, e os pesos derivam de dados clínicos.
ENDERECO = "127.0.0.1"
PORTA_PADRAO = 5000


def main() -> None:
    leitor = argparse.ArgumentParser(description=__doc__)
    leitor.add_argument("--porta", type=int, default=PORTA_PADRAO)
    opcoes = leitor.parse_args()

    comando = [
        sys.executable, "-m", "mlflow", "server",
        "--backend-store-uri", local_store_uri(),
        "--artifacts-destination", local_artifact_uri(),
        "--host", ENDERECO,
        "--port", str(opcoes.porta),
    ]
    try:
        subprocess.run(comando, check=True)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
