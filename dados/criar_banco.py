from pathlib import Path
import sqlite3

DIRETORIO = Path(__file__).resolve().parent
ARQUIVO_BANCO = DIRETORIO / "med.db"
ARQUIVO_ESQUEMA = DIRETORIO / "schema.sql"
VERSAO_ESQUEMA = 1


def criar_banco() -> None:
    esquema = ARQUIVO_ESQUEMA.read_text(encoding="utf-8-sig")

    with sqlite3.connect(ARQUIVO_BANCO) as conexao:
        conexao.execute("PRAGMA foreign_keys = ON")
        conexao.executescript(esquema)
        conexao.execute(f"PRAGMA user_version = {VERSAO_ESQUEMA}")

    print(f"Banco criado/atualizado em: {ARQUIVO_BANCO}")


if __name__ == "__main__":
    criar_banco()
