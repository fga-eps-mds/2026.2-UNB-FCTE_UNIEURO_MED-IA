# Exportação do modelo para o tablet

Proposta de formato de exportação do modelo de pontuação e registro da exportação
feita para a issue #6. O formato ainda precisa do aceite do time do aplicativo, e
parte das atividades da issue continua aberta, como descrito em
[Situação da issue](#situação-da-issue).

## Formato proposto

A proposta é exportar o modelo do PyTorch para **ONNX** e executá-lo no tablet com o
**ONNX Runtime**.

O [documento de arquitetura](https://github.com/fga-eps-mds/2026.2-UNB-FCTE_UNIEURO_MED-DOCS/blob/main/docs/produto/arquitetura.md)
deixa o runtime como pendência e cita dois candidatos, ExecuTorch e ONNX Runtime. A
Tabela 1 compara os dois pelo que a documentação de cada biblioteca declara.

**Tabela 1:** Candidatos a runtime no aplicativo

| Critério | ONNX Runtime | ExecuTorch |
|----------|--------------|------------|
| Biblioteca para React Native | [`onnxruntime-react-native`](https://github.com/microsoft/onnxruntime/tree/main/js/react_native) | [`react-native-executorch`](https://github.com/software-mansion/react-native-executorch) |
| Arquivo do modelo | `.onnx` ou `.ort` | `.pte` |
| Versão do Android | Não declara mínimo | Android 13 ou superior |
| Expo | *Config plugin* e `npx expo prebuild` | Expo SDK 55 ou superior, com *development build* |
| Expo Go | Não funciona, por ter módulo nativo | Não funciona, por ter módulo nativo |
| Conferência dos escores fora do tablet | O mesmo runtime roda no computador, pelo pacote `onnxruntime` do Python | Precisa do runtime do ExecuTorch instalado à parte |

**Fonte:** documentação das duas bibliotecas, consultada em 09/10/2026.

Três motivos sustentam a proposta:

1. A conferência dos escores usa no computador o mesmo runtime que vai rodar no
   tablet, o que reduz o risco de o escore mudar no aparelho. A repetição no tablet
   continua necessária.
2. O modelo do tablet ainda não foi informado. O ExecuTorch só roda a partir do
   Android 13, e essa exigência não pode ser confirmada agora.
3. O modelo treinado foi exportado para esse formato, e o arquivo devolveu os
   mesmos escores do original em 138 pacientes, como descrito abaixo.

Um ponto precisa ser testado no aplicativo antes do aceite. Há um relato aberto de
que o *config plugin* do `onnxruntime-react-native` 1.23.2 não foi reconhecido pelo
`npx expo prebuild`
([issue 548 dos exemplos do ONNX Runtime](https://github.com/microsoft/onnxruntime-inference-examples/issues/548)).
O aplicativo usa Expo SDK 57 e React Native 0.86, e essa combinação não foi testada
aqui.

## Exportação do modelo treinado

O arquivo de pesos do treino registrado em
[resultados do modelo base](resultados-modelo-base.md) guarda só os valores dos
parâmetros, sem a definição do modelo, e o script de treino ainda não está neste
repositório. A definição em [`exportacao/modelo.py`](../exportacao/modelo.py) foi
reconstruída a partir dos nomes e das formas dos 197 tensores do arquivo. O
carregamento exige que todos os nomes e formas coincidam, e a contagem fecha em
46.126.658 parâmetros.

Cada desenho passa por uma torre própria, formada pela parte convolucional da VGG16,
por uma convolução que reduz as características a 128 dimensões e por três blocos de
autoatenção com um token de classe. Os três tokens de classe são concatenados e
classificados em duas saídas. O arquivo exportado devolve a segunda, que é a
probabilidade de CCL.

Os nomes dos parâmetros não informam como algumas operações são encadeadas. Essas
escolhas foram comparadas em 150 pacientes do
[conjunto de dados](https://github.com/cccnlab/MCI-multiple-drawings), pela perda
em relação ao rótulo usado no treino. A Tabela 2 separa o que os dados decidiram do
que só o script de treino confirma. O artigo não informa a ativação nem o corte da VGG16.

**Tabela 2:** Escolhas da reconstrução que os pesos não determinam

| Escolha | Valor adotado | Situação |
|---------|---------------|----------|
| Normalização da imagem | Média e desvio do ImageNet | Decidida pelos dados. Sem ela a acurácia cai de 0,90 para 0,68 |
| Posição da normalização no bloco | Depois da atenção e depois da camada densa | Decidida pelos dados. A outra ordem dá acurácia de 0,30 a 0,60 |
| Saída que representa o CCL | Segunda saída do classificador | Decidida pelos dados, com AUC de 0,93 |
| Leitura da torre | Token de classe | Decidida pelos dados. A média dos tokens eleva a perda de 0,39 para 0,60 |
| Último *pooling* da VGG16 | Não aplicado | A confirmar. A perda fica em 0,386 sem ele e 0,400 com ele, e o escore muda em média 0,061 |
| Função de ativação da camada densa | GELU | A confirmar. A perda fica em 0,386 com GELU e 0,390 com ReLU, e o escore muda em média 0,021 |
| Cabeças de atenção | Uma | Informada pelo [artigo](https://pmc.ncbi.nlm.nih.gov/articles/PMC9361513/), que também dá as 128 dimensões e as 512 da camada densa. Os dados não distinguem |

**Fonte:** [Artur Mendonça Arruda](https://github.com/ArtyMend07), 2026. Os 150
pacientes incluem pacientes de treino, por isso a acurácia e o AUC desta tabela
servem para comparar as escolhas e não medem o desempenho do modelo.

A exportação e a conferência rodam com um comando. As dependências são instaladas
com o [uv](https://docs.astral.sh/uv/), a partir do `pyproject.toml` e do `uv.lock`.

```bash
uv sync
uv run python -m exportacao.gerar model.pt modelo.onnx --imagens images --quantidade 138
```

O comando grava o `.onnx`, calcula o escore de cada paciente com o modelo original
e com o arquivo exportado e informa a maior diferença. A pasta de `--imagens` tem
uma subpasta por paciente, com `clock.png`, `copy.png` e `trail.png`, como no
conjunto de dados. Sem essa opção a conferência usa imagens aleatórias.

**Tabela 3:** Resultado da exportação do modelo treinado

| Medida | Resultado |
|--------|-----------|
| Parâmetros | 46.126.658 |
| Tamanho do arquivo `.onnx` | 185,2 MB |
| Maior diferença de escore em 138 pacientes | 4,77 × 10⁻⁷ |
| Tempo por avaliação no computador | 0,52 s |

**Fonte:** [Artur Mendonça Arruda](https://github.com/ArtyMend07), 2026. Computador
com processador Intel Core i7-1165G7 e 8 GB de memória, sem placa de vídeo dedicada.
Versões: PyTorch 2.14.1, ONNX 1.23.2, ONNX Runtime 1.31.0, operadores ONNX na versão 20.

A conferência é aprovada quando a maior diferença de escore fica em até 0,0001. O
exemplo da arquitetura mostra a probabilidade com duas casas decimais, e essa
tolerância é cem vezes menor que a última casa exibida.

Limites deste resultado:

- A conferência compara o arquivo exportado com o modelo reconstruído. Enquanto os
  dois itens a confirmar da Tabela 2 não forem conferidos no script de treino, o
  escore pode diferir do que o treino calculava.
- Os 138 pacientes são os primeiros da pasta do conjunto de dados, em ordem de
  nome. O conjunto de teste da semente usada no treino depende do script.
- O tempo foi medido em um computador. Ele não substitui a medição no tablet.

Os testes automatizados de [`exportacao`](../exportacao) cobrem o modelo, a leitura
das imagens, a exportação e a conferência com um modelo pequeno:

```bash
uv run pytest --cov
```

## Contrato do arquivo exportado

O contrato entre o aplicativo e o modelo está na seção 6.3 da arquitetura. A Tabela 4
traz o que o arquivo exportado espera e devolve.

**Tabela 4:** Entradas e saída do arquivo exportado

| Item | Valor |
|------|-------|
| Entradas | `relogio`, `cubo` e `trilhas`, nessa ordem |
| Formato de cada entrada | `float32` com forma 1 × 3 × 256 × 256, ou seja, uma avaliação por chamada e canais RGB antes da altura e da largura |
| Valores de cada entrada | De 0 a 1, o valor do pixel dividido por 255 |
| Fundo e traço | Como nas imagens do conjunto de dados |
| Normalização | Feita dentro do arquivo exportado. O aplicativo não normaliza |
| Saída | `probabilidade_ccl`, um número entre 0 e 1 |

**Fonte:** [`exportacao/exportar.py`](../exportacao/exportar.py) e
[`exportacao/modelo.py`](../exportacao/modelo.py)

Pela arquitetura, o arquivo fica em `assets/modelo` no aplicativo e é carregado pelo
módulo `inferencia`. O `onnxruntime-react-native` abre o modelo a partir de um
caminho de arquivo, com `InferenceSession.create`, e a documentação lista o
carregamento do conteúdo em memória como não suportado. O passo a passo dentro do
aplicativo só pode ser escrito depois do teste com o Expo SDK 57.

## Pontos que precisam de decisão

Estes pontos apareceram ao comparar o modelo treinado com a arquitetura. Nenhum é
resolvido pela exportação.

1. **Classes da saída.** A arquitetura espera a probabilidade de três classes:
   normal, CCL e demência. O modelo treinado devolve só a probabilidade de CCL,
   porque o conjunto de dados separa os pacientes pelo MoCA menor que 25.
2. **Terceira tarefa.** O conjunto de dados usa o teste de trilhas. A arquitetura
   fala em figura geométrica e registra como pendência a escolha entre hexágono e
   pentágono. Um modelo treinado com trilhas não avalia outra figura.
3. **Mapas de calor.** A arquitetura pede um mapa por desenho e recomenda que o
   modelo já os devolva como saída, porque o runtime do tablet pode não calcular
   gradiente. O arquivo exportado devolve só a probabilidade.
4. **Tamanho no APK.** O arquivo tem 185,2 MB sem compressão. Ele também é
   derivado do conjunto de dados, então segue a mesma restrição dos pesos: só é
   publicado depois da autorização do *Product Owner*. O `.gitignore` impede que um
   arquivo `.onnx` seja versionado por engano.
5. **Tablet.** O modelo do aparelho e a versão do Android ainda não foram informados.
6. **Limite de tempo.** A issue pede o tempo de cálculo dentro do limite combinado
   com o parceiro, e esse limite ainda não foi combinado.

## Situação da issue

| Atividade ou critério | Situação |
|-----------------------|----------|
| Escolher o formato de exportação junto com o time do aplicativo | Proposta feita neste documento. Falta o teste no aplicativo e o aceite do time |
| Conferir que o modelo exportado dá os mesmos escores que o original | Feita com o modelo treinado em 138 pacientes. Falta confirmar no script de treino os dois itens da Tabela 2 |
| Medir o tempo de cálculo e o uso de memória no tablet do parceiro | Depende do tablet |
| Instruções de como o aplicativo carrega o modelo | Contrato do arquivo registrado na Tabela 4. O passo a passo depende do teste no aplicativo |

## Histórico de Versões

| Versão | Descrição | Autor(es) | Data | Revisor(es) | Data de Revisão |
|---|---|---|---|---|---|
| 1.0 | Proposta de formato de exportação e registro da exportação do modelo treinado | [Artur Mendonça Arruda](https://github.com/ArtyMend07) | 12/10/2026 | A definir | A definir |
