# Denoising Autoencoder and MNIST Classifier  
## Autoencoder Denoising e Classificador MNIST

[![Python](https://img.shields.io/badge/python-3.7%2B-blue)](https://www.python.org/)  
[![TensorFlow](https://img.shields.io/badge/tensorflow-2.x-orange)](https://www.tensorflow.org/)  
[![GitHub issues](https://img.shields.io/github/issues/GabriGuerra/auto-encoder)](https://github.com/seuusuario/autoencoder-mnist/issues)  

---

## Portuguese / Português

Este repositório contém um notebook que explora o uso de um **Autoencoder Denoising (dAE)** para pré-treinamento de um classificador de imagens no dataset MNIST.

### Objetivo

- Treinar um Autoencoder Denoising para aprender representações robustas a partir de imagens ruidosas.  
- Utilizar o encoder pré-treinado para inicializar uma rede neural profunda (DNN) para classificação dos dígitos.  
- Avaliar o impacto do pré-treinamento na performance do classificador, especialmente com poucos dados rotulados.

### Conteúdo

- Carregamento e pré-processamento do dataset MNIST, usando a biblioteca Keras do TensorFlow.  
- Aplicação de ruído nas imagens para treinar o dAE.  
- Construção, treinamento e avaliação do Autoencoder.  
- Visualização das imagens originais, ruidosas e reconstruídas.  
- Construção e treinamento da rede neural classificador usando pesos do encoder.  
- Avaliação da acurácia final no conjunto de teste.  
- Discussão crítica dos resultados obtidos.

### Como usar

1. Clone ou baixe este repositório.  
2. Certifique-se de ter instalado Python 3.7+, TensorFlow 2.x, NumPy e Matplotlib.  
3. Ative seu ambiente virtual e instale as dependências com:

pip install -r requirements.txt


4. Abra o arquivo `denoising_autoencoder_mnist.ipynb` no Jupyter Notebook ou PyCharm com suporte a notebooks.  
5. Execute as células na ordem para reproduzir os resultados.

---

## English

This repository contains a notebook that explores the use of a **Denoising Autoencoder (dAE)** to pre-train an image classifier on the MNIST dataset.

### Objective

- Train a Denoising Autoencoder to learn robust representations from noisy images.  
- Use the pretrained encoder to initialize a deep neural network (DNN) for digit classification.  
- Evaluate the impact of pretraining on the classifier’s performance, especially with limited labeled data.

### Contents

- Loading and preprocessing the MNIST dataset using TensorFlow's Keras API.  
- Adding noise to images for dAE training.  
- Building, training, and evaluating the Autoencoder.  
- Visualizing original, noisy, and reconstructed images.  
- Building and training the classifier network using encoder weights.  
- Evaluating final accuracy on the test set.  
- Critical discussion of the obtained results.

### How to use

1. Clone or download this repository.  
2. Ensure you have Python 3.7+, TensorFlow 2.x, NumPy, and Matplotlib installed.  
3. Activate your virtual environment and install dependencies with:

pip install -r requirements.txt


4. Open `denoising_autoencoder_mnist.ipynb` in Jupyter Notebook or a Jupyter-compatible IDE like PyCharm.  
5. Run the cells sequentially to reproduce the results.

---

## Requirements

- Python 3.7 or higher  
- TensorFlow 2.x  
- NumPy  
- Matplotlib  

---

## Acknowledgements

Project developed as part of the course *Redes Neurais e Deep Learning* at  
**Universidade Tecnológica Federal do Paraná (UTFPR)**  


---

## Contact

Created by Gabriel. Feel free to open issues or pull requests.

---





