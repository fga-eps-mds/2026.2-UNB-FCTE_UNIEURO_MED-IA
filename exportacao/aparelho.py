"""Prepara e serve na rede local a página que roda o modelo exportado no navegador de um aparelho.

Uso, na raiz do repositório:
    uv run python -m exportacao.aparelho modelo.onnx --imagens images
"""

import argparse
import functools
import json
import shutil
import socket
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from exportacao.conferir import TOLERANCIA
from exportacao.exportar import ENTRADAS
from exportacao.imagens import ARQUIVOS, amostras_da_pasta
from exportacao.medir import medir_execucao
from exportacao.modelo import LADO

PAGINA = Path(__file__).with_name("aparelho") / "index.html"
PORTA = 8000


def preparar(onnx: Path, imagens: Path, quantidade: int, destino: Path) -> dict:
    """Copia a página, o modelo e os desenhos e grava os escores do computador para comparação.

    Os pacientes são renomeados para não levar o identificador do conjunto de dados à página.
    """
    destino.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(PAGINA, destino / "index.html")
    shutil.copyfile(onnx, destino / "modelo.onnx")

    pacientes = sorted(caminho for caminho in imagens.iterdir() if caminho.is_dir())[:quantidade]
    amostras = amostras_da_pasta(imagens, quantidade, LADO)
    medicao = medir_execucao(onnx, amostras)

    referencia = {
        "entradas": list(ENTRADAS),
        "arquivos": list(ARQUIVOS),
        "lado": LADO,
        "tolerancia": TOLERANCIA,
        "segundos_por_avaliacao_no_computador": medicao.segundos_por_avaliacao,
        "pacientes": [],
    }
    for numero, (paciente, escore) in enumerate(zip(pacientes, medicao.escores), start=1):
        nome = f"paciente-{numero:03d}"
        (destino / nome).mkdir(exist_ok=True)
        for arquivo in ARQUIVOS:
            shutil.copyfile(paciente / arquivo, destino / nome / arquivo)
        referencia["pacientes"].append({"nome": nome, "escore": escore})

    (destino / "referencia.json").write_text(json.dumps(referencia, indent=2), encoding="utf-8")
    return referencia


def endereco_na_rede() -> str:
    """Endereço deste computador na rede local, o que o aparelho usa para abrir a página."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as conexao:
        try:
            conexao.connect(("10.255.255.255", 1))
            return conexao.getsockname()[0]
        except OSError:
            return "127.0.0.1"


def pasta_servida(pasta: Path) -> Path:
    """Tudo o que estiver na pasta fica aberto a qualquer aparelho da rede, então ela precisa ser
    uma subpasta do diretório atual, nunca o próprio diretório nem um caminho fora dele.
    """
    base = Path.cwd().resolve()
    destino = pasta.resolve()
    if destino == base or not destino.is_relative_to(base):
        raise ValueError(f"A pasta {pasta} precisa ficar dentro de {base}.")
    return destino


def criar_servidor(pasta: Path, porta: int) -> ThreadingHTTPServer:
    """Servidor que entrega a pasta preparada a qualquer aparelho da rede local."""
    manipulador = functools.partial(SimpleHTTPRequestHandler, directory=str(pasta))
    return ThreadingHTTPServer(("0.0.0.0", porta), manipulador)


def servir(pasta: Path, porta: int) -> None:
    with criar_servidor(pasta, porta) as servidor:
        print(f"Abra no navegador do aparelho, na mesma rede Wi-Fi: http://{endereco_na_rede()}:{porta}")
        print("Ctrl+C encerra.")
        try:
            servidor.serve_forever()
        except KeyboardInterrupt:
            pass


def main(argumentos: list[str] | None = None) -> None:
    leitor = argparse.ArgumentParser(description="Roda o modelo exportado no navegador de um aparelho.")
    leitor.add_argument("onnx", type=Path, help="arquivo .onnx gerado por exportacao.gerar")
    leitor.add_argument("--imagens", type=Path, required=True,
                        help="pasta com uma subpasta de desenhos por paciente")
    leitor.add_argument("--quantidade", type=int, default=10, help="avaliações feitas no aparelho")
    leitor.add_argument("--pasta", type=Path, default=Path("aparelho-teste"),
                        help="pasta temporária com o que o aparelho baixa")
    leitor.add_argument("--porta", type=int, default=PORTA)
    opcoes = leitor.parse_args(argumentos)

    pasta = pasta_servida(opcoes.pasta)
    referencia = preparar(opcoes.onnx, opcoes.imagens, opcoes.quantidade, pasta)
    print(f"{len(referencia['pacientes'])} avaliações preparadas em {pasta}")
    servir(pasta, opcoes.porta)


if __name__ == "__main__":
    main()
