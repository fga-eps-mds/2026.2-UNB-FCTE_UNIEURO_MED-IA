# Guia de Contribuição

Este repositório contém o modelo de pontuação dos testes de desenho do MED. As diretrizes abaixo se aplicam a qualquer alteração feita aqui.

Este guia trata de processo técnico. As expectativas de comportamento, respeito e inclusão valem igualmente para os três repositórios do projeto e estão no [Código de Conduta](https://github.com/fga-eps-mds/2026.2-UNB-FCTE_UNIEURO_MED-DOCS/blob/main/CODE_OF_CONDUCT.md), mantido no repositório de documentação.

## Padrão de Branch

A `develop` é a **branch padrão** do repositório e a branch de integração: quem clona cai nela, e é dela que toda branch de trabalho sai. A `main` é a linha de release, ela recebe a `develop` apenas nas entregas de release major (R1, R2 e R3) e é o que o parceiro vê.

- `feat/nome-da-funcionalidade`
- `fix/nome-da-correcao`
- `chore/nome-da-tarefa`
- `docs/nome-do-documento`

Push direto em `main` e em `develop` não é permitido. Toda alteração passa por Pull Request.

## Padrão de Commits

Este repositório adota o padrão [Conventional Commits](https://www.conventionalcommits.org/pt-br/v1.0.0/):

```
<tipo>(escopo opcional): descrição curta em português
```

| Tipo | Quando usar |
|------|-------------|
| `feat` | Nova etapa de pipeline, modelo ou rotina de treino |
| `fix` | Correção de comportamento que não funcionava |
| `refactor` | Mesma lógica, estrutura refeita |
| `test` | Criação ou ajuste de testes |
| `chore` | Pipeline, dependências, configuração |
| `docs` | Documentação do repositório |

Exemplos:

```
feat(treino): adiciona validação estratificada por escolaridade
fix(preprocess): corrige normalização das coordenadas do traçado
chore(deps): fixa versão da biblioteca de exportação do modelo
```

Commits atômicos: uma alteração lógica por commit.

## Antes de abrir o Pull Request

Confirme localmente que os testes passam. Resultado de modelo só conta como resultado se o experimento tiver run registrado no MLflow.

## Pull Requests

- Todo PR deve estar vinculado a uma Issue. Use `Closes #numero` na descrição.
- O PR aponta para `develop`, não para `main`. Como a `develop` é a branch padrão, o `Closes #numero` fecha a issue automaticamente no merge.
- Solicite revisão de no mínimo 1 colega antes do merge.
- O quality gate do SonarCloud precisa passar. PR com gate reprovado não é mergeado.
- PRs sem Issue vinculada não serão aceitos.

### Como revisar

- Não aprove um Pull Request sem ter lido e compreendido o que está sendo alterado.
- Mantenha o foco construtivo: aponte o problema concreto e, quando possível, sugira o caminho.
- Sem contexto para avaliar, peça a revisão de quem tem, em vez de aprovar por omissão.

## Dados clínicos

O conjunto de desenhos anotados é dado de paciente. Será recusado qualquer PR que:

- versione dado bruto de paciente no repositório;
- inclua identificação de paciente em log, notebook, saída de célula ou artefato de experimento;
- adicione dado ao repositório sem registrar procedência e critério de anotação.

## Restrição de execução

O modelo roda embarcado no aplicativo Android, sem internet. Alterações que dependam de serviço de inferência remoto, ou que adotem arquitetura sem caminho viável de exportação para execução no dispositivo, precisam ser discutidas na Issue antes de virarem PR.

## Histórico de Versões

| Versão | Descrição | Autor(es) | Data | Revisor(es) | Data de Revisão |
|---|---|---|---|---|---|
| 1.0 | Criação do Guia de Contribuição do repositório de IA | [Artur Mendonça Arruda](https://github.com/ArtyMend07) | 19/09/2026 | | |
