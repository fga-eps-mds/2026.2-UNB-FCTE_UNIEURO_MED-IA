#!/usr/bin/env python
"""Troca só o cubo de exames saudáveis e mede como o escore do arquivo exportado reage.

Exames saudáveis do teste com escore baixo no preparo do aplicativo ficam com o relógio e as trilhas
originais. O cubo é trocado por cubos desenhados por script, com e sem erros de ligação, e pelos cubos dos
pacientes com CCL do mesmo teste. Todos os desenhos passam pelo preparo do aplicativo.

Uso, na raiz do repositório:
    uv run --with scikit-learn==1.7.2 --with pandas --with opencv-contrib-python-headless --with pyyaml \
        --with torchvision python treino/scripts/testar_troca_do_cubo.py modelo.onnx \
        --imagens images --rotulos label.csv --semente 1 --figura cubos.png
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime

RAIZ = Path(__file__).resolve().parents[2]
# Scripts rodados por caminho não enxergam a raiz nem o pacote mci, que não é instalado aqui.
sys.path[:0] = [str(RAIZ), str(RAIZ / "treino" / "src"), str(Path(__file__).resolve().parent)]

from conferir_saidas_onnx import CORTE, entrada, ler_rgb, preparo_do_app  # noqa: E402
from exportacao.exportar import ENTRADAS  # noqa: E402
from mci.dataset import TASKS, load_records, make_splits  # noqa: E402

RELOGIO, CUBO, TRILHAS = range(3)
# Base: exame saudável que o modelo já pontua como claramente normal, para o efeito do cubo aparecer sozinho.
ESCORE_MAXIMO_DA_BASE = 0.2
BASES = 8
SEMENTE_DO_SORTEIO = 7

# Quadro em que os cubos são desenhados. O preparo do aplicativo recorta a caixa do traço, então o tamanho
# do quadro não muda a entrada do modelo.
QUADRO = 600
ESPESSURA_NO_QUADRO = 3
PONTOS_POR_ARESTA = 30
# Desvio de cada passo do tremor da mão, em pixels do quadro.
TREMOR = 0.8
PASSOS_DO_RABISCO = 60
PASSO_DO_RABISCO = 25
MARGEM_DO_RABISCO = 20

FRENTE = np.array([[150, 250], [350, 250], [350, 450], [150, 450]], float)
FUNDO = FRENTE + [110, -110]


def quadrado(cantos: np.ndarray) -> list:
    return [(cantos[i], cantos[(i + 1) % 4]) for i in range(4)]


LIGACOES = [(FRENTE[i], FUNDO[i]) for i in range(4)]
CUBOS = {
    "Certo": quadrado(FRENTE) + quadrado(FUNDO) + LIGACOES,
    "Duas ligações faltando": quadrado(FRENTE) + quadrado(FUNDO) + LIGACOES[:2],
    "Sem nenhuma ligação": quadrado(FRENTE) + quadrado(FUNDO),
    "Ligações nos cantos errados": quadrado(FRENTE) + quadrado(FUNDO)
    + [(FRENTE[i], FUNDO[(i + 1) % 4]) for i in range(4)],
    "Só o quadrado da frente": quadrado(FRENTE),
}


def aresta_tremida(inicio, fim, sorteio: np.random.Generator) -> np.ndarray:
    tremor = np.cumsum(sorteio.normal(0, TREMOR, (PONTOS_POR_ARESTA, 2)), axis=0)
    return (np.linspace(inicio, fim, PONTOS_POR_ARESTA) + tremor).astype(np.int32)


def rabisco(sorteio: np.random.Generator) -> np.ndarray:
    passos = np.cumsum(sorteio.normal(0, PASSO_DO_RABISCO, (PASSOS_DO_RABISCO, 2)), axis=0) + QUADRO / 2
    return np.clip(passos, MARGEM_DO_RABISCO, QUADRO - MARGEM_DO_RABISCO).astype(np.int32)


def desenhar(tracos: list[np.ndarray]) -> np.ndarray:
    tela = np.full((QUADRO, QUADRO, 3), 255, np.uint8)
    cv2.polylines(tela, tracos, False, (0, 0, 0), ESPESSURA_NO_QUADRO, cv2.LINE_AA)
    return preparo_do_app(tela)


def main() -> None:
    leitor = argparse.ArgumentParser(description=__doc__)
    leitor.add_argument("onnx", type=Path)
    leitor.add_argument("--imagens", type=Path, required=True)
    leitor.add_argument("--rotulos", type=Path, required=True)
    leitor.add_argument("--semente", type=int, required=True)
    leitor.add_argument("--figura", type=Path, help="Grava os cubos desenhados, como o modelo os recebe.")
    opcoes = leitor.parse_args()

    sessao = onnxruntime.InferenceSession(str(opcoes.onnx), providers=["CPUExecutionProvider"])

    def escore(desenhos: list[np.ndarray]) -> float:
        saidas = sessao.run(None, {nome: entrada(desenho) for nome, desenho in zip(ENTRADAS, desenhos)})
        return float(saidas[0].ravel()[0])

    _, _, teste = make_splits(load_records(opcoes.rotulos), opcoes.semente)
    exames = [
        [preparo_do_app(ler_rgb(opcoes.imagens / registro.patient_id / f"{tarefa}.png")) for tarefa in TASKS]
        for registro in teste
    ]
    escores = np.array([escore(exame) for exame in exames])
    rotulos = np.array([registro.hard_label for registro in teste])
    sorteio = np.random.default_rng(SEMENTE_DO_SORTEIO)
    candidatos = np.flatnonzero((rotulos == 0) & (escores < ESCORE_MAXIMO_DA_BASE))
    bases = sorteio.choice(candidatos, BASES, replace=False)

    def com_o_cubo(cubo: np.ndarray) -> np.ndarray:
        return np.array([escore([exames[b][RELOGIO], cubo, exames[b][TRILHAS]]) for b in bases])

    cubos = {nome: desenhar([aresta_tremida(a, b, sorteio) for a, b in arestas]) for nome, arestas in CUBOS.items()}
    cubos["Rabisco"] = desenhar([rabisco(sorteio)])
    if opcoes.figura:
        cv2.imwrite(str(opcoes.figura), np.hstack(list(cubos.values())))

    print(f"{len(candidatos)} exames saudáveis com escore abaixo de {ESCORE_MAXIMO_DA_BASE}; {BASES} sorteados")
    print(f"{'Cubo original do paciente':30s} {resumo(escores[bases])}")
    for nome, cubo in cubos.items():
        print(f"{nome:30s} {resumo(com_o_cubo(cubo))}")

    cubos_com_ccl = np.concatenate([com_o_cubo(exames[c][CUBO]) for c in np.flatnonzero(rotulos == 1)])
    quartis = np.percentile(cubos_com_ccl, [25, 50, 75])
    print(f"\nCubos de pacientes com CCL ({len(cubos_com_ccl)} trocas): mediana {quartis[1]:.2f}, "
          f"quartis {quartis[0]:.2f} e {quartis[2]:.2f}, acima de {CORTE} em {np.mean(cubos_com_ccl > CORTE):.0%}")


def resumo(valores: np.ndarray) -> str:
    return f"média {valores.mean():.2f}, de {valores.min():.2f} a {valores.max():.2f}"


if __name__ == "__main__":
    main()
