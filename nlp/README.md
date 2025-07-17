# Natural Language Processing Examples

[![ELMO](https://img.shields.io/badge/ELMO-Embeddings-blue.svg)](https://allennlp.org/elmo)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.4+-orange.svg)](https://tensorflow.org/)
[![GluonNLP](https://img.shields.io/badge/GluonNLP-0.10+-green.svg)](https://nlp.gluon.ai/)
[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![Archived](https://img.shields.io/badge/status-archived-red.svg)](https://github.com/julsimon/dlnotebooks)

> **⚠️ This folder is part of an archived repository.**

Examples demonstrating various Natural Language Processing techniques, from word embeddings to contextual representations using ELMO.

## 📁 Contents

- **ELMO TensorFlow.ipynb**: ELMO embeddings implementation with TensorFlow
- **ELMO with GluonNLP.ipynb**: ELMO embeddings using GluonNLP
- **Finding similar words and analogies with word embeddings.ipynb**: Word similarity and analogy examples
- **elmo_intro.txt**: Introduction text for ELMO examples
- **cosine.png**: Visualization of cosine similarity
- **normalized vectors.png**: Visualization of normalized word vectors

## 🚀 Quick Start

### Prerequisites

```bash
# For TensorFlow ELMO
pip install tensorflow tensorflow-hub

# For GluonNLP ELMO
pip install gluonnlp mxnet

# For word embeddings
pip install gensim numpy matplotlib
```

### Running the Examples

1. Navigate to this directory
2. Start Jupyter: `jupyter notebook`
3. Choose an example:
   - **ELMO TensorFlow**: `ELMO TensorFlow.ipynb`
   - **ELMO GluonNLP**: `ELMO with GluonNLP.ipynb`
   - **Word Embeddings**: `Finding similar words and analogies with word embeddings.ipynb`

## 📖 What You'll Learn

- Contextual word embeddings with ELMO
- Word similarity and analogy tasks
- Vector space representations of words
- Cosine similarity calculations
- Using pre-trained language models
- Text preprocessing and tokenization

## 🔗 Resources

- [ELMO Paper](https://arxiv.org/abs/1802.05365)
- [AllenNLP ELMO](https://allennlp.org/elmo)
- [GluonNLP Documentation](https://nlp.gluon.ai/)
- [Word2Vec Tutorial](https://www.tensorflow.org/tutorials/text/word2vec)
- [Gensim Word2Vec](https://radimrehurek.com/gensim/models/word2vec.html) 