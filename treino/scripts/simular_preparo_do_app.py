#!/usr/bin/env python
"""Redesenha as imagens de teste como o aplicativo grava e compara preparos para o modelo (IA #21).

O conjunto de dados já vem normalizado: cada desenho foi esticado para caber num quadrado de cerca de
224 px, centralizado numa imagem de 256 x 256, com traço preto de cerca de 2,4 px. O aplicativo grava o
desenho no tamanho do quadro da tela, com traço de 4 pt, e o paciente desenha em qualquer posição e
tamanho. Este script extrai o traçado central de cada imagem de teste, redesenha como o aplicativo,
aplica cada preparo e mede o modelo de cada semente.

Uso, na raiz do repositório, com a pasta de cada semente em `--pesos` (seed_0/model.pt, ...):
    uv run --with scikit-learn==1.7.2 --with pandas --with opencv-python-headless --with pyyaml \
        --with torchvision python treino/scripts/simular_preparo_do_app.py --pesos outputs/original \
        --imagens images --rotulos label.csv
"""

import argparse
import json
import sys
import zlib
from pathlib import Path

import cv2
import numpy as np
import torch

RAIZ = Path(__file__).resolve().parents[2]
# Scripts rodados por caminho não enxergam o pacote mci, que não é instalado aqui.
sys.path[:0] = [str(RAIZ / "treino" / "src")]

from mci.dataset import build_transforms, load_records, make_splits  # noqa: E402
from mci.model import MultiInputConvAtt  # noqa: E402
from mci.utils import compute_metrics  # noqa: E402

TAREFAS = ("clock", "copy", "trail")

# Quadro do aplicativo na tela de 1280 x 800 dp: largura menos as margens de 40, a coluna lateral de
# 260 e o espaço de 20 (drawing-task.styles.ts). A altura é uma estimativa.
QUADRO_LARGURA, QUADRO_ALTURA = 920, 560
ESPESSURA_APP = 4.0
COR_APP = (15, 23, 42)  # #0F172A, em RGB

# Normalização observada no conjunto: tinta de 13 a 242 px, traço preto de 2,3 a 2,5 px.
LADO, CAIXA, BORDA = 256, 224, 16
ESPESSURA_CONJUNTO = 2.5
SUPERAMOSTRAGEM = 4

PREPAROS = ("Imagem do conjunto", "Quadro esticado", "Recorte central",
            "Contorno da PNG", "Traçado vetorial")


def afinar(mascara: np.ndarray) -> np.ndarray:
    """Esqueleto de um pixel de largura (Zhang e Suen, 1984)."""
    img = mascara.astype(np.uint8).copy()
    mudou = True
    while mudou:
        mudou = False
        for passo in (0, 1):
            p = np.pad(img, 1)
            p2, p3, p4, p5 = p[:-2, 1:-1], p[:-2, 2:], p[1:-1, 2:], p[2:, 2:]
            p6, p7, p8, p9 = p[2:, 1:-1], p[2:, :-2], p[1:-1, :-2], p[:-2, :-2]
            vizinhos = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
            seq = [p2, p3, p4, p5, p6, p7, p8, p9, p2]
            transicoes = sum(((seq[i] == 0) & (seq[i + 1] == 1)).astype(np.uint8) for i in range(8))
            if passo == 0:
                condicao = (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
            else:
                condicao = (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)
            remover = (img == 1) & (vizinhos >= 2) & (vizinhos <= 6) & (transicoes == 1) & condicao
            if remover.any():
                img[remover] = 0
                mudou = True
    return img.astype(bool)


def tracado(imagem: np.ndarray) -> np.ndarray:
    """Segmentos do traçado central, em pixels do conjunto. Ponto isolado vira segmento de tamanho zero."""
    ys, xs = np.nonzero(afinar(cv2.cvtColor(imagem, cv2.COLOR_RGB2GRAY) < 128))
    pontos = set(zip(xs.tolist(), ys.tolist()))
    segmentos = []
    for x, y in pontos:
        vizinhos = [(x + dx, y + dy) for dx, dy in ((1, 0), (0, 1), (1, 1), (-1, 1)) if (x + dx, y + dy) in pontos]
        isolado = not any((x + a, y + b) in pontos for a in (-1, 0, 1) for b in (-1, 0, 1) if a or b)
        segmentos += [((x, y), v) for v in vizinhos] + ([((x, y), (x, y))] if isolado else [])
    return np.array(segmentos, dtype=np.float32).reshape(-1, 2, 2)


def desenhar(segmentos: np.ndarray, tamanho: tuple[int, int], espessura: float, cor) -> np.ndarray:
    """Desenha os segmentos, já na escala do destino, com antialias por superamostragem."""
    largura, altura = tamanho
    s = SUPERAMOSTRAGEM
    tela = np.full((altura * s, largura * s, 3), 255, np.uint8)
    t = max(1, round(espessura * s))
    for a, b in np.round(segmentos * s).astype(int):
        cv2.line(tela, tuple(a), tuple(b), cor, t, cv2.LINE_8)
    return cv2.resize(tela, (largura, altura), interpolation=cv2.INTER_AREA)


def no_quadro(segmentos: np.ndarray, chave: str, fracao: float) -> np.ndarray:
    """Segmentos posicionados no quadro do app: o desenho ocupa `fracao` da altura, em posição variável."""
    minimo, maximo = segmentos.reshape(-1, 2).min(axis=0), segmentos.reshape(-1, 2).max(axis=0)
    escala = fracao * QUADRO_ALTURA / max(maximo - minimo)
    folga = np.array([QUADRO_LARGURA, QUADRO_ALTURA]) - (maximo - minimo) * escala
    origem = folga * np.random.default_rng(zlib.crc32(chave.encode())).uniform(0.2, 0.8, size=2)
    return (segmentos - minimo) * escala + origem


def esticar_quadro(quadro: np.ndarray) -> np.ndarray:
    return cv2.resize(quadro, (LADO, LADO), interpolation=cv2.INTER_AREA)


def recorte_central(quadro: np.ndarray) -> np.ndarray:
    x0 = (QUADRO_LARGURA - QUADRO_ALTURA) // 2
    return cv2.resize(quadro[:, x0 : x0 + QUADRO_ALTURA], (LADO, LADO), interpolation=cv2.INTER_AREA)


def contorno_da_png(quadro: np.ndarray) -> np.ndarray:
    """Recorta a PNG pelo contorno da tinta, estica para 224 x 224 e põe a borda de 16 px."""
    ys, xs = np.nonzero(cv2.cvtColor(quadro, cv2.COLOR_RGB2GRAY) < 250)
    caixa = cv2.resize(quadro[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1], (CAIXA, CAIXA),
                       interpolation=cv2.INTER_AREA)
    return cv2.copyMakeBorder(caixa, BORDA, BORDA, BORDA, BORDA, cv2.BORDER_CONSTANT, value=(255, 255, 255))


def tracado_vetorial(segmentos_no_quadro: np.ndarray) -> np.ndarray:
    """Redesenha os pontos do traçado direto em 256 x 256, como o conjunto: caixa de 224 esticada,
    borda de 16 e traço preto fino. É o preparo recomendado para o aplicativo."""
    pontos = segmentos_no_quadro.reshape(-1, 2)
    minimo, maximo = pontos.min(axis=0), pontos.max(axis=0)
    escala = (CAIXA - 1) / np.maximum(maximo - minimo, 1e-6)
    return desenhar((segmentos_no_quadro - minimo) * escala + BORDA, (LADO, LADO), ESPESSURA_CONJUNTO, (0, 0, 0))


def preparar(original: np.ndarray, segmentos: np.ndarray, chave: str, fracao: float) -> dict[str, np.ndarray]:
    no_app = no_quadro(segmentos, chave, fracao)
    quadro = desenhar(no_app, (QUADRO_LARGURA, QUADRO_ALTURA), ESPESSURA_APP, COR_APP)
    return dict(zip(PREPAROS, (original, esticar_quadro(quadro), recorte_central(quadro),
                               contorno_da_png(quadro), tracado_vetorial(no_app))))


def main() -> None:
    global QUADRO_LARGURA, QUADRO_ALTURA
    leitor = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    leitor.add_argument("--pesos", type=Path, required=True, help="pasta com seed_N/model.pt de cada semente")
    leitor.add_argument("--imagens", type=Path, required=True)
    leitor.add_argument("--rotulos", type=Path, required=True, help="label.csv do conjunto de dados")
    leitor.add_argument("--sementes", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    leitor.add_argument("--fracoes", type=float, nargs="+", default=[0.4, 0.6, 0.8],
                        help="altura do desenho em relação ao quadro")
    leitor.add_argument("--quadro", type=int, nargs=2, metavar=("LARGURA", "ALTURA"),
                        default=[QUADRO_LARGURA, QUADRO_ALTURA], help="tamanho do quadro do app, em pontos")
    leitor.add_argument("--saida", type=Path, help="grava os resultados em JSON")
    opcoes = leitor.parse_args()
    QUADRO_LARGURA, QUADRO_ALTURA = opcoes.quadro

    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, transformar = build_transforms()
    registros = load_records(opcoes.rotulos)
    tracados: dict[str, np.ndarray] = {}
    resultados = {}
    for fracao in opcoes.fracoes:
        metricas = {p: [] for p in PREPAROS}
        for semente in opcoes.sementes:
            modelo = MultiInputConvAtt(pretrained=False).to(dispositivo).eval()
            modelo.load_state_dict(torch.load(opcoes.pesos / f"seed_{semente}" / "model.pt",
                                              map_location=dispositivo, weights_only=True))
            _, _, teste = make_splits(registros, semente)
            escores = {p: [] for p in PREPAROS}
            for registro in teste:
                lotes = [[] for _ in TAREFAS]
                for i, tarefa in enumerate(TAREFAS):
                    caminho = opcoes.imagens / registro.patient_id / f"{tarefa}.png"
                    original = cv2.cvtColor(cv2.imread(str(caminho), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                    segmentos = tracados.setdefault(str(caminho), tracado(original))
                    imagens = preparar(original, segmentos, f"{registro.patient_id}/{tarefa}", fracao)
                    lotes[i] = torch.stack([transformar(imagens[p]) for p in PREPAROS]).to(dispositivo)
                with torch.no_grad():
                    p_ccl = torch.softmax(modelo(lotes), dim=1)[:, 1].cpu().numpy()
                for p, valor in zip(PREPAROS, p_ccl):
                    escores[p].append(float(valor))
            rotulos = np.array([registro.hard_label for registro in teste])
            base = np.array(escores[PREPAROS[0]])
            for p in PREPAROS:
                e = np.array(escores[p])
                metricas[p].append({**compute_metrics(rotulos, e),
                                    "mesma_classe": float(((e >= 0.5) == (base >= 0.5)).mean()),
                                    "diferenca_media": float(np.abs(e - base).mean())})
        resultados[fracao] = {p: {k: (float(np.mean([m[k] for m in ms])), float(np.std([m[k] for m in ms])))
                                  for k in ms[0]} for p, ms in metricas.items()}
        print(f"\nDesenho com {fracao:.0%} da altura do quadro, média e desvio em {len(opcoes.sementes)} sementes:")
        for p, r in resultados[fracao].items():
            print(f"  {p:20s} " + "  ".join(f"{k} {m:.3f} ± {d:.3f}" for k, (m, d) in r.items()), flush=True)
    if opcoes.saida:
        opcoes.saida.write_text(json.dumps(resultados, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
