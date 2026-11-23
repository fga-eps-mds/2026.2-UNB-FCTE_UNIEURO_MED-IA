# Exportação do modelo para o tablet

Proposta de formato de exportação do modelo de pontuação e registro da exportação
feita para a issue #6. O time do aplicativo aceitou o formato em 22/10/2026, e a
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

Um ponto ainda precisa ser testado no aplicativo. Há um relato aberto de
que o *config plugin* do `onnxruntime-react-native` 1.23.2 não foi reconhecido pelo
`npx expo prebuild`
([issue 548 dos exemplos do ONNX Runtime](https://github.com/microsoft/onnxruntime-inference-examples/issues/548)).
O aplicativo usa Expo SDK 57 e React Native 0.86, e essa combinação não foi testada
aqui.

## Exportação do modelo treinado

O arquivo de pesos do treino registrado em
[resultados do modelo base](resultados-modelo-base.md) guarda só os valores dos
parâmetros, sem a definição do modelo. A definição em
[`exportacao/modelo.py`](../exportacao/modelo.py) foi reconstruída a partir dos nomes e das
formas dos 197 tensores do arquivo, antes de o código de treino chegar a este repositório. O
carregamento exige que todos os nomes e formas coincidam, e a contagem fecha em
46.126.658 parâmetros. O código de treino está agora em [`treino`](../treino/README.md), e a
conferência entre os dois está em
[Conferência com o código de treino](#conferência-com-o-código-de-treino).

Cada desenho passa por uma torre própria, formada pela parte convolucional da VGG16,
por uma convolução que reduz as características a 128 dimensões e por três blocos de
autoatenção com um token de classe. Os três tokens de classe são concatenados e
classificados em duas saídas. O arquivo exportado devolve a segunda, que é a
probabilidade de CCL.

Os nomes dos parâmetros não informam como algumas operações são encadeadas. Essas
escolhas foram comparadas em 150 pacientes do
[conjunto de dados](https://github.com/cccnlab/MCI-multiple-drawings), pela perda
em relação ao rótulo usado no treino. O artigo não informa a ativação nem o corte da VGG16.
A Tabela 2 registra o valor de cada escolha e o que o código de treino confirmou depois.

**Tabela 2:** Escolhas da reconstrução que os pesos não determinam

| Escolha | Valor adotado | Situação |
|---------|---------------|----------|
| Normalização da imagem | Média e desvio do ImageNet | Confirmada no treino. Sem ela a acurácia cai de 0,90 para 0,68 |
| Posição da normalização no bloco | Depois da atenção e depois da camada densa | Confirmada no treino. A outra ordem dá acurácia de 0,30 a 0,60 |
| Saída que representa o CCL | Segunda saída do classificador | Confirmada no treino. O AUC nos 150 pacientes é de 0,93 |
| Leitura da torre | Token de classe | Confirmada no treino. A média dos tokens eleva a perda de 0,39 para 0,60 |
| Último *pooling* da VGG16 | Aplicado | Corrigido na versão 1.4. O treino usa o *pooling*. A perda nos 150 pacientes apontava o contrário, 0,386 sem ele e 0,400 com ele |
| Função de ativação da camada densa | GELU | Confirmada no treino. A perda fica em 0,386 com GELU e 0,390 com ReLU, e o escore muda em média 0,021 |
| Cabeças de atenção | Uma | Informada pelo [artigo](https://pmc.ncbi.nlm.nih.gov/articles/PMC9361513/), que também dá as 128 dimensões e as 512 da camada densa. Confirmada no treino |

**Fonte:** [Artur Mendonça Arruda](https://github.com/ArtyMend07), 2026. Os 150
pacientes incluem pacientes de treino, por isso a acurácia e o AUC desta tabela
servem para comparar as escolhas e não medem o desempenho do modelo. As perdas e os escores
da tabela foram medidos antes da versão 1.4, sem o último *pooling*, exceto na linha dele.

A exportação e a conferência rodam com o comando do passo 4 de
[Como reproduzir](#como-reproduzir). O comando grava o `.onnx`, calcula as três saídas da Tabela 4 para cada
paciente com o modelo original e com o arquivo exportado e informa a maior diferença em cada saída. A pasta de `--imagens` tem
uma subpasta por paciente, com `clock.png`, `copy.png` e `trail.png`, como no
conjunto de dados. Sem essa opção a conferência usa imagens aleatórias.

**Tabela 3:** Resultado da exportação do modelo treinado

| Medida | Resultado |
|--------|-----------|
| Parâmetros | 46.126.658 |
| Tamanho do arquivo `.onnx` | 185,2 MB |
| Maior diferença de escore em 138 pacientes | 4,47 × 10⁻⁷ |
| Tempo por avaliação no computador | 0,51 s |
| Memória ocupada pela sessão do ONNX Runtime no computador | 393 MB |

**Fonte:** [Artur Mendonça Arruda](https://github.com/ArtyMend07), 2026. Computador
com processador Intel Core i7-1165G7 e 8 GB de memória, sem placa de vídeo dedicada.
Versões: PyTorch 2.14.1, ONNX 1.23.2, ONNX Runtime 1.31.0, operadores ONNX na versão 20.
Medição de 30/10/2026, com o último *pooling*. Uma segunda execução, com todos os 918 pacientes do
conjunto, deu a mesma diferença máxima de 4,47 × 10⁻⁷.

A memória variou muito entre execuções do mesmo arquivo: 393 MB com 138 pacientes e 124 MB com 918. Por
isso esse número serve só como ordem de grandeza e não permite comparar versões do modelo.

A memória é o aumento da memória residente do processo entre antes de abrir a sessão e
depois das 138 avaliações, medido com o `psutil` em
[`exportacao/medir.py`](../exportacao/medir.py).

A conferência é aprovada quando a maior diferença em cada saída fica em até 0,0001. O
exemplo da arquitetura mostra a probabilidade com duas casas decimais, e essa
tolerância é cem vezes menor que a última casa exibida.

Limites deste resultado:

- A conferência compara o arquivo exportado com o modelo reconstruído. A comparação com o
  treino está em [Conferência com o código de treino](#conferência-com-o-código-de-treino).
- Os 138 pacientes são os primeiros da pasta do conjunto de dados, em ordem de
  nome, e incluem pacientes de treino.
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

6. Comparar o modelo do treino com a reconstrução, com e sem o último *pooling*, o que gera os
   números da Tabela 6 e os da Tabela 7, menos os tempos. O `label.csv` vem do mesmo conjunto de
   dados:

   ```bash
   uv run --with scikit-learn==1.7.2 --with pandas --with opencv-python-headless --with pyyaml --with torchvision python treino/scripts/conferir_reconstrucao.py model.pt --imagens images --rotulos label.csv --semente 1
   ```

7. Medir o efeito do preparo da imagem no escore, o que gera os números da Tabela 8:

   ```bash
   uv run --with scikit-learn==1.7.2 --with pandas --with opencv-python-headless --with pyyaml --with torchvision python treino/scripts/simular_imagem_do_app.py modelo.onnx --imagens images --rotulos label.csv --semente 1
   ```

8. Conferir as três saídas do arquivo exportado no conjunto de teste, o que gera os números da Tabela 9:

   ```bash
   uv run --with scikit-learn==1.7.2 --with pandas --with opencv-contrib-python-headless --with pyyaml --with torchvision python treino/scripts/conferir_saidas_onnx.py modelo.onnx --imagens images --rotulos label.csv --semente 1
   ```

9. Comparar o mapa com outros métodos de explicação, o que gera os números da Tabela 10. O script
   grava uma linha por desenho em `saidas-locais/`, a cada paciente, e `--limite` define quantos
   pacientes de teste entram:

   ```bash
   uv run --with scikit-learn==1.7.2 --with pandas --with opencv-contrib-python-headless --with pyyaml --with torchvision python treino/scripts/comparar_mapas.py model.pt --imagens images --rotulos label.csv --semente 1 --limite 20
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
| Saídas | `probabilidade_ccl`, `contribuicoes` e `mapas`, nessa ordem |
| `probabilidade_ccl` | `float32` com forma 1, um número entre 0 e 1 |
| `contribuicoes` | `float32` com forma 1 × 3, uma por desenho na ordem das entradas, na escala do logit |
| `mapas` | `float32` com forma 1 × 3 × 8 × 8, um mapa por desenho na ordem das entradas, na escala do logit e com sinal |

**Fonte:** [`exportacao/exportar.py`](../exportacao/exportar.py) e
[`exportacao/modelo.py`](../exportacao/modelo.py)

### Contribuição e mapa de cada desenho

A probabilidade de CCL é a sigmoide da diferença entre os dois logits do classificador. O classificador é uma
camada linear sobre o resumo de cada torre, então essa diferença é a soma das três contribuições com uma
constante do modelo, a diferença entre os vieses das duas classes. Uma contribuição positiva puxa o escore
para CCL e uma negativa puxa para o normal. A contribuição mede quanto cada desenho pesou dentro do escore
único. Ela não substitui um escore de cada desenho sozinho, porque o modelo foi treinado só com os três
desenhos juntos.

Cada mapa é a grade de 8 × 8 da última camada da VGG16, então cada célula cobre 32 × 32 px da imagem de
256 × 256 que entrou no modelo. O mapa é calculado por oclusão das regiões (Zeiler e Fergus, 2014): o valor
de uma célula é quanto a contribuição do desenho cai quando aquela região é coberta com papel em branco. A
oclusão é feita nos tokens: o token da região é trocado pelo token que a mesma posição tem numa folha toda
branca, e o resumo da torre é calculado de novo. Um valor positivo diz que a região puxou o escore para CCL,
um negativo, que puxou para o normal, e uma região sem traço dá zero. O mapa fica na mesma escala do logit
que a contribuição.

Como roda o próprio modelo com a região coberta, o mapa mede o efeito da região no próprio escore, sem aproximação. Ele não
usa gradiente e roda dentro do arquivo exportado como o resto do modelo: a VGG16 passa uma vez por desenho e
os blocos de atenção passam 64 vezes, uma por região, em lote. O token da folha branca fica guardado no
modelo e é recalculado sempre que os pesos são carregados.

A versão 1.6 deste documento usava o *attention rollout* (Abnar e Zuidema, 2020). A Tabela 9 mostrou que
esse mapa quase não diferia entre a tinta e o fundo, e a comparação da Tabela 10 mostrou que ele não
acompanha o efeito de apagar a região, por isso ele foi trocado.

Para sobrepor o mapa ao desenho, o aplicativo amplia a grade sobre a mesma área que usou para montar a imagem
de entrada. Se a imagem de entrada sai de um recorte do desenho, o mapa vale para esse recorte.

Nenhuma avaliação clínica dos mapas nem das contribuições está registrada.

### Conferência das três saídas no conjunto de teste

O script [`treino/scripts/conferir_saidas_onnx.py`](../treino/scripts/conferir_saidas_onnx.py) roda o arquivo
exportado nos 138 pacientes de teste da semente 1 e confere se cada saída faz sentido. O escore é medido com as
imagens do conjunto e com as mesmas imagens redesenhadas como o aplicativo prepara a entrada: o traçado
central de cada desenho, a caixa dos pontos esticada até 224 px e o traço preto de 2,5 px.

**Tabela 9:** Saídas do arquivo exportado nos 138 pacientes de teste da semente 1

| Medida | Resultado |
|--------|-----------|
| AUC e acurácia com as imagens do conjunto | 0,800 e 0,775, os mesmos do treino na Tabela 6 |
| AUC e acurácia com o preparo do aplicativo | 0,758 e 0,725 |
| Pacientes na mesma classe nos dois preparos | 89,1 %, com diferença média de escore de 0,109 |
| Escore médio com e sem CCL | 0,605 e 0,261 |
| Logit do escore menos a soma das contribuições | 0,0299 em todos os pacientes, com desvio de 6,8 × 10⁻⁷ |
| Efeito médio do mapa, em módulo, nas células com tinta e sem tinta | 0,0325 e 0,0198 |
| Correlação de Spearman entre o efeito e a tinta de cada célula | 0,23 |
| Entropia do mapa, de 0 a 1, em que 1 é uniforme | 0,90 |

**Fonte:** [Artur Mendonça Arruda](https://github.com/ArtyMend07), 2026. Arquivo `vgg16-semente1-v1.7`, o mesmo
do APK de 22/11/2026, com o `model.pt` da semente 1. As três últimas linhas usam as imagens do conjunto. Medição
de 23/11/2026. Com o *attention rollout* da versão 1.6, as três últimas linhas davam 0,409 e 0,402 (valor
normalizado de 0 a 1), 0,03 e 0,96.

A contribuição se comporta como a conta da seção anterior prevê: a diferença entre o logit e a soma das três
contribuições é a mesma constante em todos os pacientes, então a parte de cada desenho no escore é exata. O
escore e as contribuições são os mesmos da versão 1.6, porque só o cálculo do mapa mudou.

O mapa por oclusão das regiões tem efeito maior nas células com tinta do que no fundo, ao contrário do
*attention rollout*, que quase não diferia entre os dois. As células sem tinta não ficam em zero porque a
região que cada célula da VGG16 enxerga passa dos 32 × 32 px dela e alcança o traço vizinho.

### Comparação do mapa com outros métodos

O script [`treino/scripts/comparar_mapas.py`](../treino/scripts/comparar_mapas.py) compara cinco métodos de
mapa com uma referência, a oclusão na própria imagem: apagar a tinta de uma célula e medir quanto a
contribuição do desenho cai. Essa referência é a mais fiel ao modelo, mas pede uma passada da VGG16 por
célula com tinta e não cabe no tablet. Cada método é medido de duas formas:

- a correlação de Spearman entre o mapa e a referência, nas células com tinta;
- a queda média da contribuição ao apagar, uma a uma, as cinco células que o método mais culpa (AOPC). Um
  método fiel faz a contribuição cair mais do que apagar cinco células ao acaso.

**Tabela 10:** Métodos de mapa frente à oclusão na imagem, em 60 desenhos de teste da semente 1

| Método | Spearman, conjunto | Spearman, preparo do app | Queda, conjunto | Queda, preparo do app |
|--------|:------------------:|:------------------------:|:---------------:|:---------------------:|
| Oclusão na imagem (referência) | 1,000 | 1,000 | +0,452 | +0,419 |
| Oclusão das regiões nos tokens (arquivo exportado) | 0,302 | 0,302 | +0,039 | +0,074 |
| Token da região retirado da atenção | 0,295 | 0,298 | — | — |
| Grad-CAM | 0,292 | 0,297 | +0,021 | +0,069 |
| Gradiente vezes ativação | 0,294 | 0,300 | — | — |
| *Attention rollout* (versão 1.6) | 0,061 | 0,088 | −0,312 | −0,176 |
| Cinco células ao acaso | — | — | −0,282 | −0,245 |

**Fonte:** [Artur Mendonça Arruda](https://github.com/ArtyMend07), 2026. Os 20 primeiros pacientes de teste da
semente 1, com os três desenhos de cada um, e o `model.pt` da semente 1. A curva de apagamento só foi medida nos
métodos com valor na tabela. Medição de 22/11/2026.

A Tabela 10 leva a três conclusões:

- O *attention rollout* não acompanha a referência, e apagar as células que ele mais destaca faz a contribuição
  cair menos do que apagar células ao acaso. Por isso ele saiu do arquivo exportado.
- A oclusão das regiões nos tokens e o Grad-CAM ficam praticamente empatados, com a oclusão um pouco à frente
  nas duas medidas. A oclusão foi escolhida porque roda dentro do arquivo exportado sem gradiente, e o runtime
  do tablet não calcula gradiente.
- Nenhum método barato chega perto da referência: a correlação fica em torno de 0,3. Com 60 desenhos, a
  diferença entre os métodos do meio da tabela não é conclusiva.

Nenhuma avaliação clínica dos mapas está registrada. Mostrar o mapa ao médico como a região que pesou no escore
ainda depende de o parceiro clínico conferir se as regiões fazem sentido.

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
diante. Como tudo o que estiver nessa pasta fica acessível a qualquer aparelho da rede, a
opção `--pasta` só aceita uma subpasta do diretório em que o comando roda. A página confere
o `onnxruntime-web` baixado do CDN pelo atributo `integrity`, e o navegador recusa o script
se o arquivo publicado mudar.

**Tabela 5:** Execução do modelo exportado no navegador de um celular Android

| Medida | Resultado |
|--------|-----------|
| Aparelho | Celular com Android 10 e 8 núcleos |
| Abertura da sessão | 1,4 s |
| Primeira avaliação | 10,10 s |
| Tempo médio por avaliação, sem a primeira | 11,94 s |
| Maior diferença de escore em 10 avaliações | 1,79 × 10⁻⁷ |

**Fonte:** [Artur Mendonça Arruda](https://github.com/ArtyMend07), 2026. Página
[`exportacao/aparelho/index.html`](../exportacao/aparelho/index.html), ONNX Runtime Web 1.30.0.
Medição de 30/10/2026, com o último *pooling*. A medição de 22/10/2026, sem ele, deu 15,30 s por
avaliação. O download dos 185,2 MB pela rede local levou 77,7 s e não entra na tabela. Das 10
avaliações, uma usa imagens geradas pela função `gerarImagemDesenho` do aplicativo, com traços criados
por script e recortadas para 256 × 256, e as outras nove usam desenhos do conjunto de dados.

Limites deste resultado:

- O navegador executa o modelo em WebAssembly, e o aplicativo vai usar o
  `onnxruntime-react-native`. O tempo dentro do aplicativo não foi medido.
- A página rodou o modelo em uma única *thread*.
- A memória no celular não foi medida.
- O celular não é o tablet do parceiro. O tempo no tablet está em [Execução no tablet](#execução-no-tablet).

## Execução no tablet

O APK de 22/11/2026 foi instalado no tablet e rodou exames completos, com o arquivo `vgg16-semente1-v1.7` e o
[`onnxruntime-react-native`](https://www.npmjs.com/package/onnxruntime-react-native) 1.24.3 só na CPU.

**Tabela 11:** Execução do modelo no aplicativo, no tablet

| Medida | Resultado |
|--------|-----------|
| Aparelho | Samsung Galaxy Tab A11+ |
| Tempo por avaliação, com as três saídas | Cerca de 1,5 s |

**Fonte:** [Artur Mendonça Arruda](https://github.com/ArtyMend07), 2026. Tempo observado no aplicativo, sem
medição repetida nem registro por avaliação.

A avaliação no tablet é bem mais rápida que os 11,94 s do celular da Tabela 5. A página do celular rodava o
modelo em WebAssembly, numa única *thread*, e o aplicativo usa o ONNX Runtime nativo. O tempo do tablet já inclui
o mapa, que roda os blocos de atenção 64 vezes por desenho. A memória no tablet não foi medida.

## Conferência com o código de treino

Com o código de treino, a classe original e a reconstrução foram carregadas com os mesmos pesos e
comparadas no conjunto de teste da semente 1. O `model.pt` usado na exportação é dessa semente: ele
reproduz a linha dela nas tabelas de [resultados do modelo base](resultados-modelo-base.md), e as
outras sementes não batem. O comando é o do passo 6 de [Como reproduzir](#como-reproduzir).

**Tabela 6:** Modelo do treino e reconstrução, nos 138 pacientes de teste da semente 1

| Medida | Treino registrado | Classe do treino | Reconstrução |
|--------|-------------------|------------------|--------------|
| Acurácia | 0,775 | 0,775 | 0,775 |
| F1-score | 0,617 | 0,617 | 0,617 |
| AUC | 0,800 | 0,800 | 0,800 |

**Fonte:** [Artur Mendonça Arruda](https://github.com/ArtyMend07), 2026. A maior diferença de escore
entre a classe do treino e a reconstrução é de 2,38 × 10⁻⁷. Medição de 30/10/2026.

A confirmação mostrou que faltava o último *pooling* na reconstrução. Sem ele, cada torre lê 257
*tokens* em vez de 65, e os pesos continuam carregando, porque o *pooling* não tem parâmetros. A
Tabela 7 mostra o efeito do erro. Nesse conjunto, as métricas sem o *pooling* ficaram um pouco maiores.

**Tabela 7:** Efeito do último *pooling* nos 138 pacientes de teste da semente 1

| Medida | Com o *pooling* (treino) | Sem o *pooling* (até a versão 1.3) |
|--------|--------------------------|------------------------------------|
| Acurácia | 0,775 | 0,797 |
| F1-score | 0,617 | 0,632 |
| AUC | 0,800 | 0,810 |
| Tempo por avaliação no computador | 0,51 s | 0,63 s |
| Diferença de escore para o treino, média e maior | 0 | 0,051 e 0,209 |
| Pacientes que mudam de classe | 0 | 7 de 138 |

**Fonte:** [Artur Mendonça Arruda](https://github.com/ArtyMend07), 2026. O passo 6 de
[Como reproduzir](#como-reproduzir) gera as métricas, a diferença de escore e a mudança de classe. Tempos medidos em
sequência no computador da Tabela 3, em 138 pacientes do início da pasta. Medição de 30/10/2026.

O arquivo mantém o *pooling* mesmo com as métricas menores, por três motivos:

1. Os pesos foram treinados com ele. Sem o *pooling*, a atenção recebe uma entrada que não viu no
   treino, e a Tabela 6 deixa de valer.
2. A diferença de acurácia corresponde a 3 de 138 pacientes, e o desvio-padrão entre os treinos é de
   0,021. Um único conjunto de teste não sustenta essa melhora.
3. Escolher a arquitetura pelo resultado no conjunto de teste contaminaria a medida de desempenho. Uma
   comparação justa exigiria treinar sem o *pooling* e avaliar nas cinco sementes.

A perda nos 150 pacientes da Tabela 2 apontou a opção errada, com uma diferença pequena e em um grupo que
inclui pacientes de treino. Uma escolha desse tipo só ficou fechada com o código de treino.

## Imagem gerada pelo aplicativo

A função `gerarImagemDesenho` do aplicativo foi executada fora do aparelho, com o Skia em WebAssembly
(`canvaskit-wasm`) e traços criados por script. O PNG saiu com fundo branco e traço `#0F172A`, e cresceu
de 1168 × 520 para 1200 × 542 quando um traço passou da borda. Isso confere a função, mas não confere um
desenho feito à mão no tablet.

A função grava o tamanho do quadro, e o modelo pede 256 × 256. Para medir o efeito desse preparo, as
imagens do conjunto de teste da semente 1 foram colocadas em um quadro simulado de 1168 × 520 e reduzidas
de cada forma. O tamanho do quadro é uma suposição, feita a partir das posições dos círculos das trilhas no
aplicativo, e o tamanho real depende do aparelho. O comando é o do passo 7 de
[Como reproduzir](#como-reproduzir).

**Tabela 8:** Efeito do preparo da imagem no escore, nos 138 pacientes de teste da semente 1

| Preparo | Acurácia | F1-score | AUC | Diferença média de escore | Mudam de classe |
|---------|----------|----------|-----|---------------------------|-----------------|
| Imagem do conjunto, sem mudança | 0,775 | 0,617 | 0,800 | 0 | 0 de 138 |
| Recorte central do quadro | 0,797 | 0,533 | 0,767 | 0,110 | 23 de 138 |
| Quadro inteiro esticado | 0,435 | 0,458 | 0,650 | 0,311 | 69 de 138 |
| Recorte central com limiar, só preto e branco | 0,449 | 0,500 | 0,760 | 0,376 | 71 de 138 |
| Traço 1 pixel mais grosso | 0,775 | 0,635 | 0,770 | 0,098 | 10 de 138 |
| Traço na cor do aplicativo | 0,732 | 0,619 | 0,803 | 0,062 | 16 de 138 |

**Fonte:** [Artur Mendonça Arruda](https://github.com/ArtyMend07), 2026. Simulação feita sobre as imagens do
conjunto. Medição de 30/10/2026.

A Tabela 8 leva a três conclusões:

- Esticar o quadro inteiro para 256 × 256 deforma o desenho e derruba a acurácia. O PNG precisa ficar
  quadrado antes de ser reduzido. O recorte central só preserva o desenho que couber nesse quadrado, e um
  traço fora dele seria perdido. A outra saída é um quadro quadrado nas tarefas, que não foi testada.
- Transformar a imagem em preto e branco puro também derruba a acurácia. As imagens do conjunto têm tons
  de cinza nas bordas do traço, e o limiar deslocou os escores.
- Mudanças pequenas de cor, espessura ou reamostragem já trocam a classe de vários pacientes. O AUC varia
  menos que a acurácia, o que indica um deslocamento do escore em relação ao corte de 0,5.

Esse último ponto é um risco para a [#12 do aplicativo](https://github.com/fga-eps-mds/2026.2-UNB-FCTE_UNIEURO_MED-APP/issues/12),
porque a imagem pode ter o formato certo e ainda assim dar um escore deslocado. A simulação não substitui
desenhos feitos no tablet, e o desempenho do modelo em imagens do aplicativo não foi medido.

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
   que deu os mesmos escores do computador na Tabela 5. Essa função e a `ler_desenho` do
   computador esticam a imagem que não for quadrada, o que não acontece com as imagens do
   conjunto. O PNG do aplicativo precisa ficar quadrado antes dessa etapa, como explicado em
   [Imagem gerada pelo aplicativo](#imagem-gerada-pelo-aplicativo).

5. Rodar `sessao.run` com as entradas `relogio`, `cubo` e `trilhas`, no formato da
   Tabela 4, e ler as saídas `probabilidade_ccl`, `contribuicoes` e `mapas` pelo nome.

6. Conferir no aparelho com os mesmos desenhos da página de conferência. O escore de cada
   avaliação deve ficar a até 0,0001 do escore do computador gravado em
   `aparelho-teste/referencia.json`.

O relato aberto sobre o *config plugin* com `npx expo prebuild`, citado em
[Formato proposto](#formato-proposto), é um ponto a observar ao gerar o build do
aplicativo com a biblioteca. Na versão 1.24.3, o pacote traz um `unimodule.json`, e a ligação
automática do Expo 57 o trata como módulo do Expo e o deixa de fora do Android. O aplicativo
contorna isso declarando o pacote no `react-native.config.js` do projeto.

## Pontos que precisam de decisão

Estes pontos apareceram ao comparar o modelo treinado com a arquitetura. Nenhum é
resolvido pela exportação.

1. **Classes da saída.** A arquitetura espera a probabilidade de três classes:
   normal, CCL e demência. O modelo treinado devolve só a probabilidade de CCL,
   porque o conjunto de dados separa os pacientes pelo MoCA menor que 25.
2. **Mapas de calor.** A arquitetura pede um mapa por desenho e recomenda que o
   modelo já os devolva como saída, porque o runtime do tablet pode não calcular
   gradiente. O arquivo exportado devolve os mapas na saída `mapas`, sem gradiente, como explicado em
   [Contribuição e mapa de cada desenho](#contribuição-e-mapa-de-cada-desenho). Falta registrar as saídas
   novas na seção 6.3 da arquitetura e validar com o parceiro clínico se as regiões destacadas fazem
   sentido para o médico.
3. **Tamanho no APK.** O arquivo tem 185,2 MB sem compressão. O `.gitignore` impede
   que um arquivo `.onnx` seja versionado por engano.
4. **Publicação dos pesos.** Na [Ata 09](https://github.com/fga-eps-mds/2026.2-UNB-FCTE_UNIEURO_MED-DOCS/blob/main/docs/atas-reunioes/Ata-09-EPS-2026-10-05-PO.md) o *Product Owner* não se opôs à
   publicação dos pesos do modelo treinado pela equipe, e a decisão ficou com a equipe.
5. **Tablet.** O aparelho é o Samsung Galaxy Tab A11+ (Tabela 11). A versão do Android não foi registrada.
6. **Limite de tempo.** A issue pede o tempo de cálculo dentro do limite combinado
   com o parceiro, e esse limite ainda não foi combinado.
7. **Modelo da R3.** Na [Ata 10](https://github.com/fga-eps-mds/2026.2-UNB-FCTE_UNIEURO_MED-DOCS/blob/main/docs/atas-reunioes/Ata-10-EPS-2026-10-19-PO.md)
   o *Product Owner* decidiu trocar a VGG16 pela ConvNeXt V2. O
   [teste de 20/10/2026](https://github.com/fga-eps-mds/2026.2-UNB-FCTE_UNIEURO_MED-DOCS/blob/main/docs/produto/modelo-ia-experimentos.md)
   não mostrou melhora, e este arquivo exporta a VGG16. A escolha do modelo que vai para o aplicativo não
   está registrada.
8. **Imagem do aplicativo.** A sensibilidade da Tabela 8 pede uma decisão sobre como
   aproximar a imagem do aplicativo das imagens do treino, ou sobre treinar com imagens parecidas com as dele.

## Situação da issue

| Atividade ou critério | Situação |
|-----------------------|----------|
| Escolher o formato de exportação junto com o time do aplicativo | ONNX com ONNX Runtime, proposto neste documento e aceito pelo time do aplicativo em 22/10/2026 |
| Conferir que o modelo exportado dá os mesmos escores que o original | Feita com o modelo treinado em 138 pacientes no computador (Tabela 3), confirmada com o código de treino (Tabela 6) e repetida em 10 avaliações no celular (Tabela 5) |
| Medir o tempo de cálculo e o uso de memória no tablet do parceiro | Tempo e memória medidos no computador (Tabela 3), tempo medido em um celular Android (Tabela 5) e tempo observado no tablet (Tabela 11). A memória no tablet continua sem medição |
| Instruções de como o aplicativo carrega o modelo | Contrato na Tabela 4 e passo a passo em [Carregamento no aplicativo](#carregamento-no-aplicativo) |

## Histórico de Versões

| Versão | Descrição | Autor(es) | Data | Revisor(es) | Data de Revisão |
|---|---|---|---|---|---|
| 1.0 | Proposta de formato de exportação e registro da exportação do modelo treinado | [Artur Mendonça Arruda](https://github.com/ArtyMend07) | 12/10/2026 | A definir | A definir |
| 1.1 | Medição de memória no computador, execução em celular Android, passo a passo de carregamento no aplicativo e instruções de reprodução | [Artur Mendonça Arruda](https://github.com/ArtyMend07) | 22/10/2026 | A definir | A definir |
| 1.2 | Testes na pasta `tests`, atualização dos pontos de decisão conforme a Ata 09, comandos concentrados em Como reproduzir e registro do aceite do formato | [Artur Mendonça Arruda](https://github.com/ArtyMend07) | 22/10/2026 | A definir | A definir |
| 1.3 | Restrição da pasta servida na rede e conferência de integridade do `onnxruntime-web` na página de conferência | [Artur Mendonça Arruda](https://github.com/ArtyMend07) | 28/10/2026 | A definir | A definir |
| 1.4 | Correção do último *pooling* da VGG16, conferência com o código de treino e efeito do preparo da imagem do aplicativo | [Artur Mendonça Arruda](https://github.com/ArtyMend07) | 30/10/2026 | A definir | A definir |
| 1.5 | Comparação sem o último *pooling* no script de conferência, para reproduzir a Tabela 7 | [Artur Mendonça Arruda](https://github.com/ArtyMend07) | 01/11/2026 | A definir | A definir |
| 1.6 | Contribuição e mapa de cada desenho como saídas do arquivo exportado e conferência das três saídas | [Artur Mendonça Arruda](https://github.com/ArtyMend07) | 17/11/2026 | A definir | A definir |
| 1.7 | Conferência das três saídas no conjunto de teste, com o preparo do aplicativo, e avaliação dos mapas | [Artur Mendonça Arruda](https://github.com/ArtyMend07) | 21/11/2026 | A definir | A definir |
| 1.8 | Mapa de cada desenho por oclusão das regiões, no lugar do *attention rollout*, e comparação com outros métodos | [Artur Mendonça Arruda](https://github.com/ArtyMend07) | 21/11/2026 | A definir | A definir |
| 1.9 | Conferência das saídas com o arquivo do APK, comparação dos mapas na Tabela 10 e tempo no tablet | [Artur Mendonça Arruda](https://github.com/ArtyMend07) | 23/11/2026 | A definir | A definir |
