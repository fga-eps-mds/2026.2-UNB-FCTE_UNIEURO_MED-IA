from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

LADO = 256
# O último pooling vem do treino (`vgg16().features`). O arquivo de pesos não o registra, porque ele não tem parâmetros.
VGG16 = (64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M")
MEDIA_IMAGENET = (0.485, 0.456, 0.406)
DESVIO_IMAGENET = (0.229, 0.224, 0.225)

# Os atributos em inglês repetem os nomes gravados no arquivo de pesos do treino.


class Extrator(nn.Module):
    def __init__(self, blocos: tuple):
        super().__init__()
        camadas: list[nn.Module] = []
        canais = 3
        for bloco in blocos:
            if bloco == "M":
                camadas.append(nn.MaxPool2d(2))
            else:
                camadas += [nn.Conv2d(canais, bloco, 3, padding=1), nn.ReLU()]
                canais = bloco
        self.features = nn.Sequential(*camadas)
        self.canais = canais

    def forward(self, imagem):
        return self.features(imagem)


class Atencao(nn.Module):
    def __init__(self, dimensao: int, cabecas: int):
        super().__init__()
        self.cabecas = cabecas
        self.qkv = nn.Linear(dimensao, 3 * dimensao)
        self.proj = nn.Linear(dimensao, dimensao)

    def forward(self, tokens):
        lote, quantidade, dimensao = tokens.shape
        qkv = self.qkv(tokens).reshape(lote, quantidade, 3, self.cabecas, dimensao // self.cabecas)
        consulta, chave, valor = qkv.permute(2, 0, 3, 1, 4)
        saida = F.scaled_dot_product_attention(consulta, chave, valor)
        return self.proj(saida.transpose(1, 2).reshape(lote, quantidade, dimensao))


class Bloco(nn.Module):
    def __init__(self, dimensao: int, cabecas: int):
        super().__init__()
        self.attn = Atencao(dimensao, cabecas)
        self.norm1 = nn.LayerNorm(dimensao)
        # A posição 2 era o dropout do treino, sem efeito na avaliação.
        self.ff = nn.Sequential(
            nn.Linear(dimensao, 4 * dimensao), nn.GELU(), nn.Identity(), nn.Linear(4 * dimensao, dimensao)
        )
        self.norm2 = nn.LayerNorm(dimensao)

    def forward(self, tokens):
        tokens = self.norm1(tokens + self.attn(tokens))
        return self.norm2(tokens + self.ff(tokens))


class Torre(nn.Module):
    def __init__(self, blocos: tuple, dimensao: int, camadas: int, cabecas: int):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dimensao))
        self.backbone = Extrator(blocos)
        self.proj = nn.Conv2d(self.backbone.canais, dimensao, 1)
        self.blocks = nn.ModuleList(Bloco(dimensao, cabecas) for _ in range(camadas))

    def forward(self, imagem):
        tokens = self.proj(self.backbone(imagem)).flatten(2).transpose(1, 2)
        tokens = torch.cat([self.cls_token.expand(len(tokens), -1, -1), tokens], dim=1)
        for bloco in self.blocks:
            tokens = bloco(tokens)
        return tokens[:, 0]


class ModeloBase(nn.Module):
    """Modelo do treino registrado em docs/resultados-modelo-base.md.

    Recebe os três desenhos com valores de 0 a 1 e devolve a probabilidade de CCL.
    """

    def __init__(
        self, blocos: tuple = VGG16, dimensao: int = 128, camadas: int = 3, cabecas: int = 1
    ):
        super().__init__()
        self.towers = nn.ModuleList(Torre(blocos, dimensao, camadas, cabecas) for _ in range(3))
        self.classifier = nn.Linear(3 * dimensao, 2)
        self.register_buffer("media", torch.tensor(MEDIA_IMAGENET).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("desvio", torch.tensor(DESVIO_IMAGENET).view(1, 3, 1, 1), persistent=False)

    def forward(self, relogio, cubo, trilhas):
        resumos = [
            torre((imagem - self.media) / self.desvio)
            for torre, imagem in zip(self.towers, (relogio, cubo, trilhas))
        ]
        return self.classifier(torch.cat(resumos, dim=1)).softmax(dim=1)[:, 1]


def carregar_modelo(caminho: Path) -> ModeloBase:
    """Monta o modelo com os pesos do treino e recusa arquivo de outra arquitetura."""
    modelo = ModeloBase()
    modelo.load_state_dict(torch.load(caminho, map_location="cpu", weights_only=True))
    return modelo.eval()
