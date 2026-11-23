#!/usr/bin/env python
"""Compara métodos de mapa por desenho pela fidelidade ao modelo, no conjunto de teste de uma semente.

A referência é a oclusão no próprio desenho: apagar a tinta de uma célula da grade de 8 × 8 e medir quanto a
contribuição do desenho ao logit muda. Um método fiel ordena as células como essa referência e, quando as
células que ele mais culpa são apagadas, a contribuição cai mais que ao apagar células ao acaso.

Uso, na raiz do repositório:
    uv run --with scikit-learn==1.7.2 --with pandas --with opencv-contrib-python-headless --with pyyaml \
        --with torchvision python treino/scripts/comparar_mapas.py model.pt \
        --imagens treino/data/images --rotulos treino/data/label.csv --semente 1
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

RAIZ = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(RAIZ), str(RAIZ / "treino" / "src"), str(Path(__file__).resolve().parent)]

from conferir_saidas_onnx import LADO, LADO_DO_MAPA, ler_rgb, preparo_do_app, tinta_por_celula  # noqa: E402
from exportacao.modelo import carregar_modelo  # noqa: E402
from mci.attention_rollout import compute_rollout  # noqa: E402
from mci.dataset import TASKS, load_records, make_splits  # noqa: E402

CELULA = LADO // LADO_DO_MAPA
# Células apagadas em sequência na curva de apagamento; os desenhos têm de 10 a 40 células com tinta.
PASSOS_DE_APAGAMENTO = 5
# Lote pequeno: com 16 imagens de 256 px, as ativações da VGG16 passam de 1 GB.
LOTE = 4
SORTEIOS = 3
# A curva de apagamento custa uma passada da VGG16 por passo; os demais métodos ficam só na correlação.
COM_CURVA = ("oclusao_imagem", "gradcam", "oclusao_tokens", "rollout")


class Torre:
    """Uma torre do modelo aberta em partes, para trocar os tokens antes da atenção."""

    def __init__(self, modelo, indice: int):
        self.torre = modelo.towers[indice]
        dimensao = self.torre.cls_token.shape[-1]
        pesos_da_ccl = modelo.classifier.weight[1] - modelo.classifier.weight[0]
        self.parcela = pesos_da_ccl[indice * dimensao:(indice + 1) * dimensao]
        self.media, self.desvio = modelo.media, modelo.desvio

    def ativacoes(self, imagens: torch.Tensor) -> torch.Tensor:
        return self.torre.backbone((imagens - self.media) / self.desvio)

    def tokens(self, ativacoes: torch.Tensor) -> torch.Tensor:
        return self.torre.proj(ativacoes).flatten(2).transpose(1, 2)

    def atencao(self, tokens: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        tokens = torch.cat([self.torre.cls_token.expand(len(tokens), -1, -1), tokens], dim=1)
        pesos_por_camada = []
        for bloco in self.torre.blocks:
            tokens, pesos = bloco(tokens)
            pesos_por_camada.append(pesos)
        return tokens[:, 0], pesos_por_camada

    def contribuicao_dos_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        resumo, _ = self.atencao(tokens)
        return resumo @ self.parcela

    def contribuicao(self, imagens: torch.Tensor) -> torch.Tensor:
        saidas = [self.contribuicao_dos_tokens(self.tokens(self.ativacoes(lote))) for lote in imagens.split(LOTE)]
        return torch.cat(saidas)


def apagar(imagem: torch.Tensor, celulas) -> torch.Tensor:
    """Pinta de branco as células da grade, como se aquela parte não tivesse sido desenhada."""
    copia = imagem.clone()
    for celula in celulas:
        linha, coluna = divmod(int(celula), LADO_DO_MAPA)
        copia[:, linha * CELULA:(linha + 1) * CELULA, coluna * CELULA:(coluna + 1) * CELULA] = 1.0
    return copia


def mapa_rollout(torre: Torre, imagem: torch.Tensor) -> np.ndarray:
    _, pesos = torre.atencao(torre.tokens(torre.ativacoes(imagem[None])))
    return compute_rollout(pesos)[0, 0, 1:].reshape(LADO_DO_MAPA, LADO_DO_MAPA).numpy()


def mapas_por_gradiente(torre: Torre, imagem: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Grad-CAM com sinal na última camada da VGG16 e gradiente × ativação em cada posição."""
    with torch.enable_grad():
        ativacoes = torre.ativacoes(imagem[None]).detach().requires_grad_(True)
        contribuicao = torre.contribuicao_dos_tokens(torre.tokens(ativacoes))
        (gradiente,) = torch.autograd.grad(contribuicao.sum(), ativacoes)
    gradcam = (gradiente.mean(dim=(2, 3), keepdim=True) * ativacoes).sum(dim=1)[0]
    gradiente_vezes_ativacao = (gradiente * ativacoes).sum(dim=1)[0]
    return gradcam.detach().numpy(), gradiente_vezes_ativacao.detach().numpy()


def mapas_por_oclusao_dos_tokens(torre: Torre, tokens: torch.Tensor, tokens_do_branco: torch.Tensor, base: float):
    """Quanto a contribuição cai quando cada token é coberto com o do papel em branco, ou é removido da atenção."""
    posicoes = tokens.shape[1]
    trocados = tokens.repeat(posicoes, 1, 1)
    indices = torch.arange(posicoes)
    trocados[indices, indices] = tokens_do_branco[0]
    fica = ~torch.eye(posicoes, dtype=torch.bool)
    sem_o_token = torch.stack([tokens[0][fica[j]] for j in range(posicoes)])
    branco = base - torre.contribuicao_dos_tokens(trocados)
    fora = base - torre.contribuicao_dos_tokens(sem_o_token)
    forma = (LADO_DO_MAPA, LADO_DO_MAPA)
    return branco.numpy().reshape(forma), fora.numpy().reshape(forma)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return float("nan")
    return float(pd.Series(a).corr(pd.Series(b), method="spearman"))


def queda_ao_apagar(torre: Torre, imagem: torch.Tensor, ordem: np.ndarray, base: float) -> float:
    """Média da queda da contribuição ao apagar, uma a uma, as células na ordem dada (AOPC)."""
    passos = ordem[:PASSOS_DE_APAGAMENTO]
    apagadas = torch.stack([apagar(imagem, passos[:k + 1]) for k in range(len(passos))])
    return float((base - torre.contribuicao(apagadas)).mean())


def main() -> None:
    leitor = argparse.ArgumentParser(description=__doc__)
    leitor.add_argument("pesos", type=Path)
    leitor.add_argument("--imagens", type=Path, required=True)
    leitor.add_argument("--rotulos", type=Path, required=True)
    leitor.add_argument("--semente", type=int, required=True)
    leitor.add_argument("--preparo-do-app", action="store_true")
    leitor.add_argument("--limite", type=int, default=None, help="Quantos pacientes de teste usar.")
    opcoes = leitor.parse_args()

    torch.manual_seed(0)
    sorteio = np.random.default_rng(0)
    modelo = carregar_modelo(opcoes.pesos)
    torres = [Torre(modelo, indice) for indice in range(len(TASKS))]
    branco = torch.ones(1, 3, LADO, LADO)
    with torch.no_grad():
        tokens_do_branco = [torre.tokens(torre.ativacoes(branco)) for torre in torres]

    _, _, teste = make_splits(load_records(opcoes.rotulos), opcoes.semente)
    teste = teste[:opcoes.limite]
    sufixo = "-app" if opcoes.preparo_do_app else ""
    destino = RAIZ / "saidas-locais" / f"comparacao-mapas-semente{opcoes.semente}{sufixo}.csv"
    linhas = []
    inicio = time.time()
    for numero, registro in enumerate(teste, start=1):
        for tarefa, torre, branco_da_torre in zip(TASKS, torres, tokens_do_branco):
            rgb = ler_rgb(opcoes.imagens / registro.patient_id / f"{tarefa}.png")
            if opcoes.preparo_do_app:
                rgb = preparo_do_app(rgb)
            tinta = tinta_por_celula(rgb).ravel() > 0
            celulas = np.flatnonzero(tinta)
            imagem = torch.from_numpy(rgb.astype(np.float32) / 255).permute(2, 0, 1)
            with torch.no_grad():
                tokens = torre.tokens(torre.ativacoes(imagem[None]))
                base = float(torre.contribuicao_dos_tokens(tokens))
                referencia = base - torre.contribuicao(torch.stack([apagar(imagem, [c]) for c in celulas])).numpy()
                mapas = {"rollout": mapa_rollout(torre, imagem)}
                mapas["oclusao_tokens"], mapas["token_removido"] = mapas_por_oclusao_dos_tokens(
                    torre, tokens, branco_da_torre, base
                )
            mapas["gradcam"], mapas["grad_x_ativacao"] = mapas_por_gradiente(torre, imagem)
            mapas["oclusao_imagem"] = np.zeros(LADO_DO_MAPA * LADO_DO_MAPA)
            mapas["oclusao_imagem"][celulas] = referencia

            linha = {"paciente": registro.patient_id, "tarefa": tarefa, "rotulo": registro.hard_label,
                     "celulas": len(celulas), "contribuicao": base}
            with torch.no_grad():
                for nome, mapa in mapas.items():
                    valores = mapa.ravel()[celulas]
                    linha[f"spearman_{nome}"] = spearman(valores, referencia)
                    linha[f"na_tinta_{nome}"] = float(np.abs(mapa.ravel()[tinta]).mean() /
                                                      max(np.abs(mapa.ravel()).mean(), 1e-12))
                    if nome not in COM_CURVA:
                        continue
                    linha[f"queda_{nome}"] = queda_ao_apagar(torre, imagem, celulas[np.argsort(-valores)], base)
                    # Sentido contrário: apagar o que o método diz que puxou para sem CCL deve subir a contribuição.
                    linha[f"subida_{nome}"] = -queda_ao_apagar(torre, imagem, celulas[np.argsort(valores)], base)
                linha["queda_acaso"] = np.mean([
                    queda_ao_apagar(torre, imagem, sorteio.permutation(celulas), base) for _ in range(SORTEIOS)
                ])
            linhas.append(linha)
        # A tabela vai para o disco a cada paciente, para uma execução interrompida não perder o que já mediu.
        pd.DataFrame(linhas).to_csv(destino, index=False)
        print(f"{numero}/{len(teste)} pacientes, {time.time() - inicio:.0f} s", flush=True)

    tabela = pd.DataFrame(linhas)
    print(f"\nDesenhos: {len(tabela)}; tabela em {destino}")
    print(f"Queda ao apagar 5 células ao acaso: {tabela['queda_acaso'].mean():+.3f}")
    for nome in ("oclusao_imagem", "gradcam", "grad_x_ativacao", "oclusao_tokens", "token_removido", "rollout"):
        curva = (f"queda {tabela[f'queda_{nome}'].mean():+.3f}  subida {tabela[f'subida_{nome}'].mean():+.3f}  "
                 if nome in COM_CURVA else "")
        print(f"{nome:16s} Spearman com a oclusão na imagem {tabela[f'spearman_{nome}'].mean():+.3f}  {curva}"
              f"tinta/média {tabela[f'na_tinta_{nome}'].mean():.2f}")


if __name__ == "__main__":
    main()
