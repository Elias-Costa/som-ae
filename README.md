# SOM-AE

Implementacao do projeto de Iniciação Cientifica desenvolvido por Elias C. Rodrigues, sob orientação do Prof. Dr. Leonardo Nogueira Matos, na Universidade Federal de Sergipe (UFS), no contexto do PIBIC entre 2023 e 2025.

O repositório contém o protótipo de pesquisa descrito no artigo **"Mapas Auto-Organizaveis e Autoencoders"**, publicado nos anais da **ERBASE 2025**. A proposta combina **Self-Organizing Maps (SOM)** com **autoencoders convolucionais**, substituindo cada neurônio da grade por um autoencoder especializado e acoplando uma MLP para classificação.

## Artigo

- PDF no repositório: [artigo_publicado.pdf](./artigo_publicado.pdf)
- Título: *Mapas Auto-Organizaveis e Autoencoders*
- Autores: Elias C. Rodrigues e Leonardo N. Matos
- Evento: ERBASE 2025

## Visão Geral

O objetivo do projeto é investigar uma arquitetura híbrida, hierárquica e interpretável para classificação de imagens. Em vez de usar um SOM tradicional, cada célula da grade passa a ser um autoencoder convolucional próprio. Com isso:

- o SOM preserva a organização topológica entre os neurônios;
- cada autoencoder aprende a reconstruir padrões visuais semelhantes;
- a camada latente de cada autoencoder funciona como extrator de características;
- uma MLP associada a cada célula realiza a classificação de forma descentralizada.

## Arquitetura do Modelo

Fluxo geral:

1. Uma grade bidimensional do SOM é criada.
2. Cada posição da grade contém um `AutoencoderConv`.
3. Para cada imagem de entrada, encontra-se a BMU (*Best Matching Unit*) com base no erro de reconstrução.
4. A BMU é ajustada por backpropagation para reconstruir melhor aquela amostra.
5. Os autoencoders vizinhos são atualizados para aproximar seus pesos dos pesos da BMU, respeitando a topologia do SOM.
6. Em uma segunda etapa, a representação latente produzida pela BMU alimenta uma MLP responsável pela classificação.

### Autoencoder convolucional

Cada autoencoder possui:

- 4 camadas convolucionais na etapa de codificação;
- operações de `MaxPool2d` para redução espacial;
- 4 camadas de convolução transposta na etapa de decodificação;
- ativações `ReLU` e `Sigmoid` na saída de reconstrução;
- uma cabeça MLP para classificação.

Para imagens MNIST, a camada latente utilizada pela classificação tem dimensão **32 x 3 x 3**, seguida por uma MLP com camadas **128 -> 64 -> 10**.

### Treinamento

- **Treinamento não supervisionado do SOM com autoencoders**
  - 100 imagens do MNIST;
  - 1000 épocas;
  - taxa de aprendizado inicial `0.2`;
  - raio de vizinhançaa inicial `2.0`.
- **Treinamento supervisionado das MLPs**
  - 500 imagens adicionais, distintas das usadas no SOM;
  - 1000 épocas;
  - 10 épocas internas de treino da MLP da BMU por amostra;
  - mesma parametrização inicial de taxa de aprendizado e raio.

## Estrutura do Repositorio

- `main.py`: script principal com interface de linha de comando para treinar, avaliar, salvar e carregar checkpoints.
- `SOM.py`: implementação da classe `SelfOrganizingMap`, incluindo busca da BMU, treinamento topológico, etapa supervisionada e cálculo de acurácia.
- `AE_CNN.py`: implementação do autoencoder convolucional e da MLP de classificação associada a cada célula da grade.
- `requirements.txt`: dependências mínimas do projeto.
- `artigo_publicado.pdf`: versão do artigo publicado nos anais da ERBASE 2025.

## Requisitos

O código foi escrito em Python e depende principalmente de:

- `torch`
- `torchvision`
- `numpy`
- `matplotlib`

Instalação mínima:

```bash
pip install -r requirements.txt
```

Se desejar instalar uma versão específica do PyTorch com suporte a GPU, o ideal é seguir as instrucoes oficiais do framework.

## Como Executar

O script principal agora permite treinar o modelo do zero, salvar checkpoints e retomar execuções posteriores.

```bash
python main.py --skip-plot
```

Ao executar:

- o MNIST é baixado automaticamente para a pasta `./data`;
- o script seleciona subconjuntos balanceados por classe;
- o SOM-AE é treinado com os hiperparâmetros padrão do artigo;
- um checkpoint é salvo por padrão em `checkpoints/som_ae_mnist.pt`;
- a acurácia no conjunto de teste é calculada ao final, a menos que isso seja desabilitado por argumento.

### Exemplos de uso

Treinar do zero com a configuração do artigo:

```bash
python main.py --skip-plot
```

Treinar um experimento menor para teste rapido:

```bash
python main.py --som-images 20 --classifier-images 40 --som-epochs 1 --classifier-epochs 1 --skip-plot
```

Apenas avaliar um checkpoint já salvo:

```bash
python main.py --load-checkpoint checkpoints/som_ae_mnist.pt --skip-som-training --skip-classifier-training --skip-plot
```

Retomar treinamento a partir de um checkpoint:

```bash
python main.py --load-checkpoint checkpoints/som_ae_mnist.pt --classifier-epochs 200 --skip-plot
```

### Argumentos principais

- `--device {auto,cpu,cuda}`: escolhe o dispositivo de execução.
- `--seed`: define a semente aleatória.
- `--grid-height` e `--grid-width`: definem o tamanho da grade do SOM.
- `--som-images` e `--classifier-images`: definem o número de imagens usadas em cada fase.
- `--som-epochs` e `--classifier-epochs`: controlam a duração do treinamento.
- `--load-checkpoint`: carrega um checkpoint existente.
- `--save-checkpoint`: escolhe onde salvar o checkpoint gerado.
- `--data-root`: escolhe onde armazenar o MNIST localmente.
- `--skip-som-training`: pula a etapa nao supervisionada.
- `--skip-classifier-training`: pula a etapa supervisionada.
- `--skip-eval`: pula o cálculo de acurácia.
- `--skip-plot`: evita abrir a janela do `matplotlib`.

## Resultados Reportados no Artigo

- a grade utilizada nos experimentos principais foi de **7 x 7**;
- o modelo foi treinado com poucos dados, priorizando interpretabilidade e organização topológica;
- a acurácia obtida no conjunto de teste do MNIST foi de **74,81%** sobre **10.000 imagens**.

O trabalho destaca que o resultado ficou abaixo de arquiteturas mais modernas, mas foi considerado promissor dado o baixo número de amostras utilizadas no treinamento e a proposta de interpretabilidade do modelo.

## Principais Contribuições

- Integração direta entre SOM e autoencoders convolucionais em uma única arquitetura.
- Organização topológica de especialistas locais, em vez de um classificador centralizado único.
- Uso da representação latente dos autoencoders como base para classificação supervisionada.
- Ênfase em interpretabilidade visual, por meio da inspeção da grade de autoencoders após o treinamento.

## Limitações Atuais do Repositorio

Este repositório preserva o código do protótipo de pesquisa. Por isso, alguns pontos de engenharia ainda não foram empacotados como um projeto de reprodução completa:

- não há checkpoint publicado junto com o código;
- a configuração do experimento continua centralizada no `main.py`;
- a reprodução estrita ainda pode variar conforme versão do PyTorch, hardware e tempo de treinamento;
- o treinamento é computacionalmente custoso, especialmente com grades maiores ou mais dados.

## Citação

Se este repositório ou a ideia do trabalho forem úteis para sua pesquisa, considere citar:

```bibtex
@inproceedings{rodrigues2025somae,
  title     = {Mapas Auto-Organizaveis e Autoencoders},
  author    = {Rodrigues, Elias C. and Matos, Leonardo N.},
  booktitle = {Anais da ERBASE 2025},
  year      = {2025}
}
```

## Agradecimentos

Este trabalho foi desenvolvido com apoio do **Programa Institucional de Bolsas de Iniciacao Cientifica (PIBIC)** da **Universidade Federal de Sergipe (UFS)**, sob coordenacao da **COPES**.
