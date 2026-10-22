import json
import socket
import threading
from urllib.request import urlopen

import pytest
from PIL import Image

from exportacao import aparelho
from exportacao.conferir import TOLERANCIA
from exportacao.exportar import exportar_onnx
from exportacao.imagens import ARQUIVOS


@pytest.fixture
def pasta_de_imagens(tmp_path):
    for paciente in ("918273", "102938", "564738"):
        (tmp_path / "imagens" / paciente).mkdir(parents=True)
        for arquivo in ARQUIVOS:
            Image.new("RGB", (8, 8), (200, 200, 200)).save(tmp_path / "imagens" / paciente / arquivo)
    (tmp_path / "imagens" / ".DS_Store").write_text("")
    return tmp_path / "imagens"


@pytest.fixture
def onnx_pequeno(monkeypatch, modelo, lado, tmp_path):
    monkeypatch.setattr(aparelho, "LADO", lado)
    return exportar_onnx(modelo, tmp_path / "modelo.onnx", lado)


def test_prepara_pagina_modelo_desenhos_e_escores(onnx_pequeno, pasta_de_imagens, tmp_path):
    destino = tmp_path / "aparelho"

    referencia = aparelho.preparar(onnx_pequeno, pasta_de_imagens, 2, destino)

    assert (destino / "index.html").read_text(encoding="utf-8").startswith("<!doctype html>")
    assert (destino / "modelo.onnx").stat().st_size == onnx_pequeno.stat().st_size
    assert [paciente["nome"] for paciente in referencia["pacientes"]] == ["paciente-001", "paciente-002"]
    for paciente in referencia["pacientes"]:
        assert 0 <= paciente["escore"] <= 1
        assert all((destino / paciente["nome"] / arquivo).exists() for arquivo in ARQUIVOS)
    assert referencia["tolerancia"] == TOLERANCIA
    assert json.loads((destino / "referencia.json").read_text(encoding="utf-8")) == referencia


def test_nao_leva_o_identificador_do_paciente_para_a_pagina(onnx_pequeno, pasta_de_imagens, tmp_path):
    destino = tmp_path / "aparelho"

    aparelho.preparar(onnx_pequeno, pasta_de_imagens, 3, destino)

    nomes = {caminho.name for caminho in destino.iterdir()}
    assert not nomes & {"918273", "102938", "564738"}
    assert "918273" not in (destino / "referencia.json").read_text(encoding="utf-8")


def test_main_prepara_e_serve_na_porta_informada(onnx_pequeno, pasta_de_imagens, monkeypatch, tmp_path):
    servidos = []
    monkeypatch.setattr(aparelho, "servir", lambda pasta, porta: servidos.append((pasta, porta)))
    destino = tmp_path / "aparelho"

    aparelho.main([str(onnx_pequeno), "--imagens", str(pasta_de_imagens), "--pasta", str(destino),
                  "--porta", "8123"])

    assert servidos == [(destino, 8123)]
    assert (destino / "referencia.json").exists()


def test_servidor_entrega_os_arquivos_preparados(onnx_pequeno, pasta_de_imagens, tmp_path):
    destino = tmp_path / "aparelho"
    aparelho.preparar(onnx_pequeno, pasta_de_imagens, 1, destino)
    servidor = aparelho.criar_servidor(destino, 0)
    porta = servidor.server_address[1]
    linha = threading.Thread(target=servidor.serve_forever)
    linha.start()
    try:
        for arquivo in ("index.html", "referencia.json", "modelo.onnx", "paciente-001/clock.png"):
            with urlopen(f"http://127.0.0.1:{porta}/{arquivo}") as resposta:
                assert resposta.read() == (destino / arquivo).read_bytes()
    finally:
        servidor.shutdown()
        servidor.server_close()
        linha.join()


def test_endereco_na_rede_pertence_a_este_computador():
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as conexao:
        conexao.bind((aparelho.endereco_na_rede(), 0))
