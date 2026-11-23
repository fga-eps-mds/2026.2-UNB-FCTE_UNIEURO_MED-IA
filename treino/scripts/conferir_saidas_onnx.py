#!/usr/bin/env python
"""Confere se as três saídas do arquivo exportado fazem sentido no conjunto de teste de uma semente.

Mede o escore com as imagens do conjunto e com as mesmas imagens redesenhadas como o aplicativo prepara
a entrada (traçado central, caixa dos pontos esticada até 224 px e traço de 2,5 px), confere que a soma
das contribuições com uma constante dá o logit do escore e mede quanto de cada mapa cai sobre a tinta. O mapa tem
sinal, então as medidas sobre a tinta usam o tamanho do efeito de cada célula.

Uso, na raiz do repositório:
    uv run --with scikit-learn==1.7.2 --with pandas --with opencv-contrib-python-headless \
        --with torchvision python treino/scripts/conferir_saidas_onnx.py modelo.onnx \
        --imagens images --rotulos label.csv --semente 1
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
# Scripts rodados por caminho não enxergam a raiz nem o pacote mci, que não é instalado aqui.
sys.path[:0] = [str(RAIZ), str(RAIZ / "treino" / "src")]

from exportacao.exportar import ENTRADAS, SAIDAS  # noqa: E402
from mci.dataset import TASKS, load_records, make_splits  # noqa: E402
from mci.utils import compute_metrics  # noqa: E402

LADO = 256
# Mesmo preparo de src/features/inferencia/entrada-modelo.ts e preparo-desenho.ts do aplicativo.
CAIXA, BORDA, ESPESSURA = 224, 16, 2.5
SUPERAMOSTRAGEM = 4
LADO_DO_MAPA = 8
TINTA = 128
CORTE = 0.5


def ler_rgb(caminho: Path) -> np.ndarray:
    imagem = cv2.cvtColor(cv2.imread(str(caminho)), cv2.COLOR_BGR2RGB)
    return cv2.resize(imagem, (LADO, LADO), interpolation=cv2.INTER_LINEAR)


def tracado_central(imagem: np.ndarray) -> np.ndarray:
    """Pontos do esqueleto da tinta, o mais perto que uma imagem chega dos pontos que o app grava."""
    tinta = (cv2.cvtColor(imagem, cv2.COLOR_RGB2GRAY) < TINTA).astype(np.uint8) * 255
    esqueleto = cv2.ximgproc.thinning(tinta)
    ys, xs = np.nonzero(esqueleto)
    return np.stack([xs, ys], axis=1).astype(np.float32)


def preparo_do_app(imagem: np.ndarray) -> np.ndarray:
    """Redesenha o traçado como o app: cada eixo esticado até a caixa de 224 px, traço preto de 2,5 px."""
    pontos = tracado_central(imagem)
    if len(pontos) == 0:
        return np.full((LADO, LADO, 3), 255, np.uint8)
    minimo = pontos.min(axis=0)
    extensao = np.maximum(pontos.max(axis=0) - minimo, 1)
    ajustados = BORDA + (pontos - minimo) * (CAIXA - 1) / extensao
    s = SUPERAMOSTRAGEM
    tela = np.full((LADO * s, LADO * s, 3), 255, np.uint8)
    raio = max(1, round(ESPESSURA * s / 2))
    # O esqueleto não guarda a ordem dos pontos; um disco por ponto refaz um traço de espessura constante.
    for x, y in np.round(ajustados * s).astype(int):
        cv2.circle(tela, (int(x), int(y)), raio, (0, 0, 0), -1, cv2.LINE_AA)
    return cv2.resize(tela, (LADO, LADO), interpolation=cv2.INTER_AREA)


def entrada(imagem: np.ndarray) -> np.ndarray:
    return (imagem.astype(np.float32) / 255).transpose(2, 0, 1)[None]


def tinta_por_celula(imagem: np.ndarray) -> np.ndarray:
    """Fração de pixels com tinta em cada célula da grade de 8 × 8 do mapa."""
    tinta = cv2.cvtColor(imagem, cv2.COLOR_RGB2GRAY) < TINTA
    lado = LADO // LADO_DO_MAPA
    return tinta.reshape(LADO_DO_MAPA, lado, LADO_DO_MAPA, lado).mean(axis=(1, 3))


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    # Postos médios nos empates: muitas células não têm tinta nenhuma.
    return float(pd.Series(a.ravel()).corr(pd.Series(b.ravel()), method="spearman"))


def entropia_normalizada(mapa: np.ndarray) -> float:
    """De 0 a 1: 1 é um mapa uniforme, que não aponta região nenhuma."""
    p = mapa.ravel() / max(mapa.sum(), 1e-8)
    return float(-(p * np.log(p + 1e-12)).sum() / np.log(p.size))


def avaliar(sessao: onnxruntime.InferenceSession, desenhos: list[np.ndarray]) -> list[np.ndarray]:
    return sessao.run(list(SAIDAS), {nome: entrada(desenho) for nome, desenho in zip(ENTRADAS, desenhos)})


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def main() -> None:
    leitor = argparse.ArgumentParser(description=__doc__)
    leitor.add_argument("onnx", type=Path)
    leitor.add_argument("--imagens", type=Path, required=True)
    leitor.add_argument("--rotulos", type=Path, required=True)
    leitor.add_argument("--semente", type=int, required=True)
    opcoes = leitor.parse_args()

    _, _, teste = make_splits(load_records(opcoes.rotulos), opcoes.semente)
    rotulos = np.array([registro.hard_label for registro in teste])
    sessao = onnxruntime.InferenceSession(str(opcoes.onnx), providers=["CPUExecutionProvider"])

    escores, escores_app, constantes, contribuicoes = [], [], [], []
    no_traco, no_fundo, correlacoes, entropias = [], [], [], []
    for registro in teste:
        desenhos = [ler_rgb(opcoes.imagens / registro.patient_id / f"{tarefa}.png") for tarefa in TASKS]
        probabilidade, contribuicao, mapas = avaliar(sessao, desenhos)
        escores.append(float(probabilidade[0]))
        contribuicoes.append(contribuicao[0])
        constantes.append(float(logit(probabilidade)[0] - contribuicao[0].sum()))
        for desenho, mapa in zip(desenhos, np.abs(mapas[0])):
            tinta = tinta_por_celula(desenho)
            no_traco.append(float(mapa[tinta > 0].mean()))
            if (tinta == 0).any():
                no_fundo.append(float(mapa[tinta == 0].mean()))
            correlacoes.append(spearman(mapa, tinta))
            entropias.append(entropia_normalizada(mapa))
        escores_app.append(float(avaliar(sessao, [preparo_do_app(d) for d in desenhos])[0][0]))

    escores, escores_app = np.array(escores), np.array(escores_app)
    contribuicoes, constantes = np.array(contribuicoes), np.array(constantes)
    mesma_classe = float(((escores >= CORTE) == (escores_app >= CORTE)).mean())

    print(f"Pacientes de teste da semente {opcoes.semente}: {len(teste)}, {int(rotulos.sum())} com CCL")
    print(f"Imagens do conjunto: {compute_metrics(rotulos, escores)}")
    print(f"Preparo do app: {compute_metrics(rotulos, escores_app)}")
    print(f"Mesma classe nos dois preparos: {mesma_classe:.1%}; "
          f"diferença média de escore {np.abs(escores - escores_app).mean():.3f}")
    print(f"Escore médio com CCL {escores[rotulos == 1].mean():.3f}, sem CCL {escores[rotulos == 0].mean():.3f}")
    print(f"Constante do logit (soma das contribuições): média {constantes.mean():.4f}, "
          f"desvio {constantes.std():.2e}")
    for indice, tarefa in enumerate(ENTRADAS):
        print(f"Contribuição do {tarefa}: média {contribuicoes[:, indice].mean():+.3f}, "
              f"desvio {contribuicoes[:, indice].std():.3f}")
    print(f"Efeito médio do mapa, em módulo: {np.mean(no_traco):.4f} nas células com tinta, "
          f"{np.mean(no_fundo):.4f} nas sem tinta")
    print(f"Spearman entre o efeito e a tinta de cada célula: {np.nanmean(correlacoes):.3f}")
    print(f"Entropia normalizada do mapa (1 = uniforme): {np.mean(entropias):.3f}")


if __name__ == "__main__":
    main()
