# GoogLeNet para a classificação de sneakers

Uma Rede Neural Convolucional (CNN, do inglês *Convolutional Neural Network*) que implementa os blocos *inceptions* como proposto pela primeira vez no artigo [*Going deeper with convolution*](https://arxiv.org/pdf/1409.4842v1) para realizar a classificação de *sneakers*. A base de dados utilizada para treinar o modelo está disponível gratuitamente no [Kaggle](https://www.kaggle.com/datasets/nikolasgegenava/sneakers-classification)

## Alunos integrantes da equipe

* Carlos Emanuel Silva e Melo Oliveira
* Giovanni Bogliolo Sirihal Duarte
* Gustavo Andrade Alves
* Vítor Nunes Calhau

## Professor responsável

* Leonardo Vilela Cardoso 

## Instruções de Uso

Antes de iniciar qualquer um dos passos abaixo, clone o repositório:

```	
git clone https://github.com/ICEI-PUC-Minas-PPLES-Topicos/pmg-es-2025-1-tes-sneakers-classification.git
```

### Treinamento do modelo no Google Colaboratory

1. Criar uma conta no Kaggle utilizando [este link](https://www.kaggle.com/account/login?phase=startRegisterTab&returnUrl=%2F);
2. Criar nova chave de API em : `Configurações` -> `API`-> `Criar Novo Toke`;
3. Após a criação da chave, um arquivo nomeado *kaggle.json* será baixado;
4. Abrir o *jupyter notebook*, disponível na pasta [códigos](./códigos/googlenet.ipynb), no [*Google Colaboratory*](https://colab.google/);
5. Com o *notebook* aberto, alterar o tipo de ambiente de execução: `Ambiente de execução` -> `Alterar o tipo de ambiente de execução`-> `GPUs: T4`;
6. Executar as células individualmente, ou `Ambiente de execução` -> `Executar tudo` ou `Ctrl + F9`.

### Executar a inferência utilizando o modelo treinado

1. Criar uma ambiente virtual:
	
	```
	python3 -m venv .venv
	```

2. Ativar o ambiente virtual:
	
	```
	source .venv/bin/activate
	```
3. Instalar as dependências necessárias para executar a inferência:

	```
	pip install -r requierments.txt
	```
4. Executar o script *inferencia.py*:
	
	```
	python3 inferencia.py
	```
5. Fazer uma requisição para o modelo:

	```
	curl -X POST -F "image=@*caminho para a imagem*" https://localhost:5000/predict
	```

**OBS: o resultado tanto do treino quanto da inferência pode ser visto no vídeo disponibilizado [aqui](./vídeo)**
