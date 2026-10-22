# Exportação do modelo para o tablet

Proposta de formato de exportação do modelo de pontuação e registro da exportação
feita para a issue #6. O formato ainda precisa do aceite do time do aplicativo, e a
situação de cada atividade da issue está em [Situação da issue](#situação-da-issue).

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

A exportação e a conferência rodam com o comando do passo 4 de
[Como reproduzir](#como-reproduzir). O comando grava o `.onnx`, calcula o escore de cada paciente com o modelo original
e com o arquivo exportado e informa a maior diferença. A pasta de `--imagens` tem
uma subpasta por paciente, com `clock.png`, `copy.png` e `trail.png`, como no
conjunto de dados. Sem essa opção a conferência usa imagens aleatórias.

**Tabela 3:** Resultado da exportação do modelo treinado

| Medida | Resultado |
|--------|-----------|
| Parâmetros | 46.126.658 |
| Tamanho do arquivo `.onnx` | 185,2 MB |
| Maior diferença de escore em 138 pacientes | 4,77 × 10⁻⁷ |
| Tempo por avaliação no computador | 0,74 s |
| Memória ocupada pela sessão do ONNX Runtime no computador | 333 MB |

**Fonte:** [Artur Mendonça Arruda](https://github.com/ArtyMend07), 2026. Computador
com processador Intel Core i7-1165G7 e 8 GB de memória, sem placa de vídeo dedicada.
Versões: PyTorch 2.14.1, ONNX 1.23.2, ONNX Runtime 1.31.0, operadores ONNX na versão 20.
Medição de 22/10/2026. A medição de 12/10/2026 deu 0,52 s por avaliação no mesmo
computador.

A memória é o aumento da memória residente do processo entre antes de abrir a sessão e
depois das 138 avaliações, medido com o `psutil` em
[`exportacao/medir.py`](../exportacao/medir.py).

A conferência é aprovada quando a maior diferença de escore fica em até 0,0001. O
exemplo da arquitetura mostra a probabilidade com duas casas decimais, e essa
tolerância é cem vezes menor que a última casa exibida.

Limites deste resultado:

- A conferência compara o arquivo exportado com o modelo reconstruído. Enquanto os
  dois itens a confirmar da Tabela 2 não forem conferidos no script de treino, o
  escore pode diferir do que o treino calculava.
- Os 138 pacientes são os primeiros da pasta do conjunto de dados, em ordem de
  nome. O conjunto de teste da semente usada no treino depende do script.
- O tempo foi medido em um computador. Ele não substitui a medição no tablet, e a
  medição em aparelho está em [Execução em um celular Android](#execução-em-um-celular-android).

Os testes automatizados em [`tests`](../tests) cobrem o modelo, a leitura
das imagens, a exportação, a conferência, a medição e a preparação da página de
conferência no aparelho com um modelo pequeno:

```bash
uv run pytest --cov
```

## Como reproduzir

1. Instalar as dependências na raiz do repositório, com o
   [uv](https://docs.astral.sh/uv/), a partir do `pyproject.toml` e do `uv.lock`:

   ```bash
   uv sync
   ```

2. Baixar o `images.zip` do
   [conjunto de dados](https://github.com/cccnlab/MCI-multiple-drawings) e extrair a
   pasta `images`, que tem uma subpasta por paciente.
3. Obter o arquivo de pesos `model.pt` com o responsável pelo treino registrado em
   [resultados do modelo base](resultados-modelo-base.md). O arquivo não está neste
   repositório.
4. Exportar e conferir no computador, o que gera os números da Tabela 3:

   ```bash
   uv run python -m exportacao.gerar model.pt modelo.onnx --imagens images --quantidade 138
   ```

5. Conferir em um aparelho Android, o que gera os números da Tabela 5, como descrito
   em [Execução em um celular Android](#execução-em-um-celular-android):

   ```bash
   uv run python -m exportacao.aparelho modelo.onnx --imagens images
   ```

Sem o `model.pt`, os testes automatizados e o passo 4 sem `--imagens` ainda conferem a
exportação, com um modelo pequeno e com imagens aleatórias.

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

## Execução em um celular Android

O arquivo exportado também foi rodado no navegador de um celular Android, com o
[`onnxruntime-web`](https://www.npmjs.com/package/onnxruntime-web) 1.30.0. O computador
serve uma página na rede Wi-Fi local, o aparelho abre a página, baixa o modelo e dez
avaliações do conjunto de dados, calcula o escore de cada uma e compara com o escore
que o ONNX Runtime do computador calculou para os mesmos desenhos.

O comando do passo 5 de [Como reproduzir](#como-reproduzir) mostra o endereço que o aparelho deve abrir. O aparelho e o computador
precisam estar na mesma rede, e o firewall do computador precisa permitir conexões do
Python na porta 8000. Os desenhos vão para a pasta `aparelho-teste`, que o `.gitignore`
exclui, com os pacientes renomeados para `paciente-001`, `paciente-002` e assim por
diante.

**Tabela 5:** Execução do modelo exportado no navegador de um celular Android

| Medida | Resultado |
|--------|-----------|
| Aparelho | Celular com Android 10 e 8 núcleos |
| Abertura da sessão | 1,9 s |
| Primeira avaliação | 12,32 s |
| Tempo médio por avaliação, sem a primeira | 15,30 s |
| Maior diferença de escore em 10 avaliações | 1,49 × 10⁻⁷ |

**Fonte:** [Artur Mendonça Arruda](https://github.com/ArtyMend07), 2026. Página
[`exportacao/aparelho/index.html`](../exportacao/aparelho/index.html), ONNX Runtime Web 1.30.0.
Medição de 22/10/2026. O download dos 185,2 MB pela rede local levou 80,4 s e não
entra na tabela.

Limites deste resultado:

- O navegador executa o modelo em WebAssembly, e o aplicativo vai usar o
  `onnxruntime-react-native`. O tempo dentro do aplicativo não foi medido.
- A página rodou o modelo em uma única *thread*.
- A memória no celular não foi medida.
- O celular não é o tablet do parceiro. A medição no tablet continua pendente.

## Carregamento no aplicativo

Pela arquitetura, o arquivo fica em `assets/modelo` no aplicativo e é carregado pelo
módulo `inferencia`. O passo a passo abaixo segue o
[README do `onnxruntime-react-native`](https://www.npmjs.com/package/onnxruntime-react-native)
e o contrato da Tabela 4, e ainda não foi executado no aplicativo.

1. Instalar a biblioteca:

   ```bash
   npm install onnxruntime-react-native
   ```

2. Copiar o `modelo.onnx` gerado por `exportacao.gerar` para `assets/modelo`.

3. Abrir a sessão com `InferenceSession.create`, a partir do caminho do arquivo no
   aparelho. O README informa que o carregamento a partir de um `ArrayBuffer` não é
   suportado:

   ```ts
   import { InferenceSession } from "onnxruntime-react-native";

   const sessao = await InferenceSession.create(caminhoDoModelo);
   ```

4. Converter cada desenho em um `Float32Array` de 3 × 256 × 256 com os canais
   separados, primeiro todo o vermelho, depois o verde e depois o azul, cada pixel
   dividido por 255. É a conversão da função `lerDesenho` da página de conferência,
   que deu os mesmos escores do computador na Tabela 5.

5. Rodar `sessao.run` com as entradas `relogio`, `cubo` e `trilhas`, no formato da
   Tabela 4, e ler a saída `probabilidade_ccl`.

6. Conferir no aparelho com os mesmos desenhos da página de conferência. O escore de cada
   avaliação deve ficar a até 0,0001 do escore do computador gravado em
   `aparelho-teste/referencia.json`.

O relato aberto sobre o *config plugin* com `npx expo prebuild`, citado em
[Formato proposto](#formato-proposto), é um ponto a observar ao gerar o build do
aplicativo com a biblioteca.

## Pontos que precisam de decisão

Estes pontos apareceram ao comparar o modelo treinado com a arquitetura. Nenhum é
resolvido pela exportação.

1. **Classes da saída.** A arquitetura espera a probabilidade de três classes:
   normal, CCL e demência. O modelo treinado devolve só a probabilidade de CCL,
   porque o conjunto de dados separa os pacientes pelo MoCA menor que 25.
2. **Mapas de calor.** A arquitetura pede um mapa por desenho e recomenda que o
   modelo já os devolva como saída, porque o runtime do tablet pode não calcular
   gradiente. O arquivo exportado devolve só a probabilidade.
3. **Tamanho no APK.** O arquivo tem 185,2 MB sem compressão. O `.gitignore` impede
   que um arquivo `.onnx` seja versionado por engano.
4. **Publicação dos pesos.** Na [Ata 09](https://github.com/fga-eps-mds/2026.2-UNB-FCTE_UNIEURO_MED-DOCS/blob/main/docs/atas-reunioes/Ata-09-EPS-2026-10-05-PO.md) o *Product Owner* não se opôs à
   publicação dos pesos do modelo treinado pela equipe, e a decisão ficou com a equipe.
5. **Tablet.** O modelo do aparelho e a versão do Android ainda não foram informados.
6. **Limite de tempo.** A issue pede o tempo de cálculo dentro do limite combinado
   com o parceiro, e esse limite ainda não foi combinado.

## Situação da issue

| Atividade ou critério | Situação |
|-----------------------|----------|
| Escolher o formato de exportação junto com o time do aplicativo | ONNX com ONNX Runtime, proposto neste documento. Falta o aceite do time do aplicativo |
| Conferir que o modelo exportado dá os mesmos escores que o original | Feita com o modelo treinado em 138 pacientes no computador (Tabela 3) e em 10 no celular (Tabela 5). Os dois itens a confirmar da Tabela 2 dependem do script de treino |
| Medir o tempo de cálculo e o uso de memória no tablet do parceiro | Tempo e memória medidos no computador (Tabela 3) e tempo medido em um celular Android (Tabela 5). A medição no tablet continua pendente |
| Instruções de como o aplicativo carrega o modelo | Contrato na Tabela 4 e passo a passo em [Carregamento no aplicativo](#carregamento-no-aplicativo) |

## Histórico de Versões

| Versão | Descrição | Autor(es) | Data | Revisor(es) | Data de Revisão |
|---|---|---|---|---|---|
| 1.0 | Proposta de formato de exportação e registro da exportação do modelo treinado | [Artur Mendonça Arruda](https://github.com/ArtyMend07) | 12/10/2026 | A definir | A definir |
| 1.1 | Medição de memória no computador, execução em celular Android, passo a passo de carregamento no aplicativo e instruções de reprodução | [Artur Mendonça Arruda](https://github.com/ArtyMend07) | 22/10/2026 | A definir | A definir |
| 1.2 | Testes na pasta `tests`, atualização dos pontos de decisão conforme a Ata 09 e comandos concentrados em Como reproduzir | [Artur Mendonça Arruda](https://github.com/ArtyMend07) | 22/10/2026 | A definir | A definir |
