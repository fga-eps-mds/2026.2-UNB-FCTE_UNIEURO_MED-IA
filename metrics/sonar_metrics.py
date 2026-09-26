#!/usr/bin/env python3
"""
Versionamento das releases e coleta das métricas do SonarCloud, chamados pelo
workflow release.yml a cada PR fechado com merge.

Atende ao requisito do professor publicado no Discord da disciplina em 14/09:
o arquivo .json com as métricas de cada versão do produto é gerado pelo pipeline
do próprio repositório de código, com o nome

    fga-eps-mds-<repositório>-<MM-DD-YYYY-HH-MM-SS>-v<X.Y.Z>.json

Subcomandos:
    versao                  imprime tag=vX.Y.Z (formato de $GITHUB_OUTPUT)
    aguardar <sha>          espera o SonarCloud analisar o commit do merge
    coletar <tag>           grava o .json em analytics-raw-data/ e imprime arquivo=<caminho>

Usa apenas a biblioteca padrão. Variáveis de ambiente: GH_TOKEN, GITHUB_REPOSITORY,
MAJOR e MINOR ("true"/"false", vindas dos rótulos do PR).
"""

import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ORGANIZACAO_SONAR = "fga-eps-mds"

# Lista definida pelo professor (Discord, 14/09).
METRICAS = [
    "files",
    "functions",
    "complexity",
    "comment_lines_density",
    "duplicated_lines_density",
    "coverage",
    "ncloc",
    "tests",
    "test_errors",
    "test_failures",
    "test_execution_time",
    "security_rating",
]

SONAR = "https://sonarcloud.io/api"
DESTINO = Path("analytics-raw-data")


def repositorio():
    return os.environ["GITHUB_REPOSITORY"]


def chave_sonar():
    return f"{ORGANIZACAO_SONAR}_{repositorio().split('/')[1]}"


def obter_json(url, token=None):
    cabecalhos = {"Accept": "application/json"}
    if token:
        cabecalhos["Authorization"] = f"Bearer {token}"
    with urllib.request.urlopen(urllib.request.Request(url, headers=cabecalhos), timeout=30) as resposta:
        return json.load(resposta)


def versao():
    """Próxima versão a partir da maior tag vX.Y.Z já publicada no repositório."""
    releases = obter_json(
        f"https://api.github.com/repos/{repositorio()}/releases?per_page=100", os.environ.get("GH_TOKEN"))
    versoes = [tuple(map(int, m.groups()))
               for r in releases
               if (m := re.fullmatch(r"v(\d+)\.(\d+)\.(\d+)", r["tag_name"]))]
    x, y, z = max(versoes, default=(0, 0, 0))

    if os.environ.get("MAJOR") == "true":
        x, y, z = x + 1, 0, 0
    elif os.environ.get("MINOR") == "true":
        y, z = y + 1, 0
    else:
        z += 1
    print(f"tag=v{x}.{y}.{z}")


def aguardar(sha, limite_s=600, intervalo_s=20):
    """O merge dispara o workflow do SonarCloud em paralelo; sem esperar por ele,
    as métricas coletadas seriam as da versão anterior."""
    url = f"{SONAR}/project_analyses/search?project={chave_sonar()}&ps=20"
    inicio = time.monotonic()
    while time.monotonic() - inicio < limite_s:
        try:
            analises = obter_json(url).get("analyses", [])
            if any(a.get("revision") == sha for a in analises):
                print(f"Análise do commit {sha[:7]} disponível no SonarCloud.")
                return
        except (urllib.error.URLError, TimeoutError) as erro:
            print(f"Consulta ao SonarCloud falhou ({erro}); tentando de novo.")
        time.sleep(intervalo_s)
    print(f"::warning::O SonarCloud não publicou a análise de {sha[:7]} em {limite_s // 60} min; "
          "as métricas refletem a última análise disponível.")


def coletar(tag):
    """Grava a resposta de measures/component_tree, o mesmo endpoint do parser do
    repositório Analytics da disciplina e do exemplo indicado pelo professor.

    O notebook de análise da disciplina percorre `components` filtrando por arquivo,
    diretório e testes (qualifier FIL, DIR e UTS), por isso a métrica por arquivo é
    necessária. O valor agregado do projeto inteiro fica em `baseComponent`.
    A árvore é paginada (no máximo 500 nós por página); todas as páginas são reunidas."""
    parametros = {"component": chave_sonar(), "metricKeys": ",".join(METRICAS), "ps": 500}
    dados, componentes, pagina = None, [], 1
    while True:
        consulta = urllib.parse.urlencode({**parametros, "p": pagina})
        resposta = obter_json(f"{SONAR}/measures/component_tree?{consulta}")
        dados = dados or resposta
        componentes += resposta.get("components", [])
        total = resposta.get("paging", {}).get("total", 0)
        if len(componentes) >= total or not resposta.get("components"):
            break
        pagina += 1
    dados["components"] = componentes
    dados["paging"] = {"pageIndex": 1, "pageSize": len(componentes), "total": len(componentes)}

    agora = datetime.now(ZoneInfo("America/Sao_Paulo"))
    nome = f"fga-eps-mds-{repositorio().split('/')[1]}-{agora:%m-%d-%Y-%H-%M-%S}-{tag}.json"
    DESTINO.mkdir(exist_ok=True)
    caminho = DESTINO / nome
    caminho.write_text(json.dumps(dados, ensure_ascii=False), encoding="utf-8")
    print(f"arquivo={caminho}")


if __name__ == "__main__":
    comando, *argumentos = sys.argv[1:] or ["?"]
    {"versao": versao, "aguardar": aguardar, "coletar": coletar}.get(
        comando, lambda *_: sys.exit(__doc__))(*argumentos)
