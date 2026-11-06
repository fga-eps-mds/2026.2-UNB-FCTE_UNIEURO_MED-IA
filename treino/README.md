# Treino do modelo base

Código de treino de [Thales Germano Vargas Lima](https://github.com/thalesgvl) e [Vitor Carvalho Pereira](https://github.com/vcpVitor), usado para
gerar os pesos registrados em [resultados do modelo base](../docs/resultados-modelo-base.md)
e o teste com ConvNeXt V2 da [Ata 10](https://github.com/fga-eps-mds/2026.2-UNB-FCTE_UNIEURO_MED-DOCS/blob/main/docs/atas-reunioes/Ata-10-EPS-2026-10-19-PO.md).
O pacote foi recebido em 30/10/2026. Os guias [TRAINING.md](TRAINING.md),
[RESULTS.md](RESULTS.md) e [IMPROVEMENTS.md](IMPROVEMENTS.md) estão em inglês, como vieram.

| Pasta ou arquivo | Conteúdo |
|---|---|
| `src/mci` | Modelo, dados, perda, laço de treino e mapas de atenção |
| `configs` | `original` (artigo), `improved` e `convnextv2` |
| `scripts` | Treino e visualização, mais os scripts de conferência e do MLflow abaixo |
| `tests` | Testes do registro no MLflow, que rodam no ambiente do treino |
| `requirements.txt` | Dependências do treino, separadas das do repositório |

O treino usa versões diferentes das do `pyproject.toml` da raiz, por isso tem ambiente próprio,
como explicado no [TRAINING.md](TRAINING.md).

Os scripts abaixo foram acrescentados depois e rodam no ambiente da raiz. O resultado deles está em
[exportação do modelo](../docs/exportacao-do-modelo.md):

- `scripts/conferir_reconstrucao.py` compara o modelo do treino com `exportacao/modelo.py`.
- `scripts/simular_imagem_do_app.py` mede o efeito do preparo da imagem do aplicativo no escore.

## Registro dos treinos no MLflow

`scripts/run_experiment.py` e `scripts/train.py` registram cada treino no MLflow sem nenhuma opção a mais
([#22](https://github.com/fga-eps-mds/2026.2-UNB-FCTE_UNIEURO_MED-IA/issues/22)). O `smoke_test.py` não
registra nada. Os comandos desta seção rodam na pasta `treino`, com o ambiente do [TRAINING.md](TRAINING.md).

Para ver os registros, rode o comando abaixo e abra <http://127.0.0.1:5000>. A interface só aceita acesso da
própria máquina.

```bash
python scripts/abrir_mlflow.py
```

**Tabela 1:** O que cada registro guarda

| Registro | Conteúdo |
|---|---|
| Experimento (`5-seeds`) | Configuração, hash do `label.csv`, sementes, `mean_test_*` e `std_test_*` |
| Semente (`seed-<n>`), dentro do experimento | Configuração, hash do `label.csv`, tamanho dos conjuntos, métricas por época, `test_*`, uso de CPU, memória e GPU |
| Arquivos da semente | `config.yaml`, `environment.txt` (versão de cada pacote), `history.json`, `result.json` e `weights/model.pt` |
| Marcas da semente | Commit, `git.dirty` (se havia alteração sem commit), `device`, `weights_sha256` e `registro` (`treino` ou `importado`) |

Para comparar configurações, use `mean_test_auc` e `std_test_auc`, porque a variação entre sementes mostra se a
diferença passa do acaso da divisão dos dados. Para ver sobreajuste, compare `train_loss` com `val_auc` por
época. A melhora de um modelo só conta com o registro dele no MLflow, com o mesmo `data.label_csv_sha256` do
modelo comparado.

Para saber de qual registro saiu um arquivo de pesos, inclusive depois de renomeado ou copiado:

```bash
python scripts/localizar_pesos.py caminho/model.pt
```

O `result.json` de cada semente também traz o `mlflow_run_id`.

O `history.json` é gravado a cada época, então um treino que cai no meio mantém as épocas já feitas. Se o
MLflow não receber alguma época, o treino avisa e continua, e no fim da semente as épocas que faltam são enviadas
de novo a partir do `history.json`. Se ainda faltar alguma, o registro termina como `FAILED`, nunca como
concluído com buraco.

Treinos feitos antes do MLflow entram a partir de `outputs/<experimento>/seed_<n>`. O script recusa pesos já
registrados e arquivos faltando. O commit desses treinos fica como `desconhecido`, e a configuração e o hash dos
dados são os da hora do registro, por isso o `label.csv` e a configuração precisam ser os usados no treino.

```bash
python scripts/registrar_treinos_existentes.py --config configs/original.yaml
```

### Onde os registros ficam

Banco e arquivos ficam em `outputs/mlflow`, que o `.gitignore` cobre. Cada máquina tem o próprio registro, porque
o lugar compartilhado ainda não foi decidido na #22 e os pesos só podem ir para ele depois de combinar a
publicação, já que o conjunto de dados não declara licença. Quando houver um servidor, a variável
`MLFLOW_TRACKING_URI` aponta o treino para ele.

Cada cópia dos pesos da VGG16 ocupa cerca de 180 MB. Para guardar só o hash, sem a cópia, acrescente à configuração:

```yaml
tracking:
  log_weights: false
```

Sem a cópia, um novo treino com a mesma semente sobrescreve o `model.pt` de `outputs`, e o registro antigo fica
só com o hash.

### Dados clínicos e rede

O registro guarda só números, configuração, hash e pesos. Identificador de paciente, desenho e nota do MoCA ficam
de fora, e não se usa `autolog` nem `log_input`, que podem guardar amostras da entrada. A telemetria de uso do
MLflow fica desligada em `src/mci/tracking.py`, e os mapas de `visualize.py` não entram no registro porque o nome
do arquivo traz o identificador do paciente.

### Testes

```bash
python -m pytest
```

Os testes usam dados inventados e um banco temporário. Eles ficam fora do pipeline do repositório, que usa o
ambiente da raiz, sem o MLflow.

No Windows, crie o ambiente em `treino/.venv` ou outro caminho curto. O MLflow tem arquivos de nome longo, e o
limite de 260 caracteres do caminho quebra a criação do banco quando a pasta é muito funda.

## Histórico de Versões

| Versão | Descrição | Autor(es) | Data | Revisor(es) | Data de Revisão |
|---|---|---|---|---|---|
| 1.0 | Inclusão do código de treino e dos scripts de conferência | [Artur Mendonça Arruda](https://github.com/ArtyMend07) | 30/10/2026 | A definir | A definir |
| 1.1 | Registro dos treinos no MLflow | [Artur Mendonça Arruda](https://github.com/ArtyMend07) | 02/11/2026 | A definir | A definir |
