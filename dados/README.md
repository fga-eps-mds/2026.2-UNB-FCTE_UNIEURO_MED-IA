# Base de dados provisória

Este diretório contém o esquema inicial da base SQLite do projeto MED e o script Python que cria o arquivo local `med.db`.

## Requisitos

- Python 3.10 ou superior
- Nenhuma dependência externa: o script usa `sqlite3`, incluído na biblioteca padrão do Python.

## Criar o banco

Na raiz do repositório, execute:

```bash
python dados/criar_banco.py
```

No Windows, caso `python` não esteja disponível no terminal:

```powershell
py dados/criar_banco.py
```

O arquivo `dados/med.db` será criado. Executar o script novamente é seguro: a tabela é criada com `IF NOT EXISTS`, sem apagar os registros existentes.

## Tabela inicial

A tabela `profissional` contém os dados necessários para o cadastro e o login:

- `id`: identificador inteiro local;
- `nome` e `email`;
- `crm_numero` e `uf_crm`: partes do CRM completo preenchido no cadastro, por exemplo `12345/DF`;
- `senha_hash`: hash da senha, nunca a senha em texto puro;
- `criado_em` e `atualizado_em`: datas em formato ISO 8601;
- `ativo`: indicador `0` ou `1` para permitir desativação de conta.

O app deve separar e validar o CRM completo antes de salvar suas duas partes. O script cria apenas a estrutura; não cria contas fictícias nem define a lógica de autenticação.

## Observação sobre o aplicativo

Este banco é um artefato provisório criado por Python. O Expo/React Native não executa este script. Para usar a base no aplicativo, a equipe ainda precisará decidir como distribuir ou copiar o arquivo `.db` e acessá-lo com uma biblioteca SQLite compatível com o Expo. O script Python pode também servir como referência para o esquema.
