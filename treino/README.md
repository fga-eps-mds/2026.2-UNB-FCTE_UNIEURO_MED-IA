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
| `scripts` | Treino e visualização, mais os dois scripts de conferência abaixo |
| `requirements.txt` | Dependências do treino, separadas das do repositório |

O treino usa versões diferentes das do `pyproject.toml` da raiz, por isso tem ambiente próprio,
como explicado no [TRAINING.md](TRAINING.md).

Os scripts abaixo foram acrescentados depois e rodam no ambiente da raiz. O resultado deles está em
[exportação do modelo](../docs/exportacao-do-modelo.md):

- `scripts/conferir_reconstrucao.py` compara o modelo do treino com `exportacao/modelo.py`.
- `scripts/simular_imagem_do_app.py` mede o efeito do preparo da imagem do aplicativo no escore.

## Histórico de Versões

| Versão | Descrição | Autor(es) | Data | Revisor(es) | Data de Revisão |
|---|---|---|---|---|---|
| 1.0 | Inclusão do código de treino e dos scripts de conferência | [Artur Mendonça Arruda](https://github.com/ArtyMend07) | 30/10/2026 | A definir | A definir |
