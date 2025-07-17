# Deep Learning Notebooks Collection

[![Archived](https://img.shields.io/badge/status-archived-red.svg)](https://github.com/julsimon/dlnotebooks)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![Python](https://img.shields.io/badge/Python-3.6+-blue.svg)](https://python.org/)

> **⚠️ This repository is archived and no longer actively maintained.**

A comprehensive collection of Jupyter and Zeppelin notebooks demonstrating various deep learning frameworks, techniques, and AWS SageMaker integrations. These notebooks were originally used in articles published on [Medium](https://medium.com/@julsimon).

## 📚 Overview

This repository contains practical examples and tutorials covering:

- **Deep Learning Frameworks**: TensorFlow/Keras, PyTorch, MXNet, GluonCV
- **AutoML**: AutoGluon
- **Graph Neural Networks**: DGL (Deep Graph Library)
- **NLP**: ELMO, word embeddings, sentiment analysis
- **Computer Vision**: Classification, detection, segmentation
- **AWS SageMaker**: Training, inference, custom containers
- **Traditional ML**: Scikit-learn, Spark ML

## 🗂️ Repository Structure

```
dlnotebooks/
├── autogluon/          # AutoGluon examples
├── dgl/               # Deep Graph Library tutorials
├── gluoncv/           # Computer vision with GluonCV
├── keras/             # Keras/TensorFlow tutorials
├── ktrain/            # Ktrain NLP examples
├── mxnet/             # MXNet and Gluon tutorials
├── nlp/               # Natural Language Processing
├── pytorch/           # PyTorch tutorials
├── sagemaker/         # AWS SageMaker examples
├── scikit/            # Scikit-learn tutorials
└── spark/             # Apache Spark ML examples
```

## 🚀 Quick Start

### Prerequisites

- Python 3.6+
- Jupyter Notebook or JupyterLab
- Required packages (see individual folder READMEs for specific dependencies)

### Installation

1. Clone this repository:
```bash
git clone https://github.com/julsimon/dlnotebooks.git
cd dlnotebooks
```

2. Install Jupyter:
```bash
pip install jupyter
```

3. Navigate to the specific framework folder and follow the instructions in its README.

## 📖 Contents by Category

### 🤖 AutoML
- **AutoGluon**: Automated machine learning on Boston Housing dataset

### 🕸️ Graph Neural Networks
- **DGL**: Karate Club community detection example

### 👁️ Computer Vision
- **GluonCV**: Classification, detection, and segmentation models
- **Keras**: MNIST, Fashion MNIST, custom CNN implementations
- **MXNet**: Image classification, GANs, pre-trained models

### 🧠 Natural Language Processing
- **ELMO**: Contextual word embeddings
- **Word Embeddings**: Similarity and analogy examples
- **Ktrain**: BERT-based sentiment analysis

### ☁️ AWS SageMaker
- **Image Classification**: Transfer learning, custom algorithms
- **Factorization Machines**: MovieLens recommendation system
- **DeepAR**: Time series forecasting
- **XGBoost**: Gradient boosting examples

### 🔬 Traditional Machine Learning
- **Scikit-learn**: Linear/logistic regression, decision trees, clustering, PCA
- **Spark ML**: Spam classification, clustering with SageMaker integration

## 📝 Usage

Each subfolder contains:
- Jupyter notebooks with detailed explanations
- Supporting data files (where applicable)
- Docker configurations (for SageMaker examples)
- README files with specific setup instructions

## 🤝 Contributing

**Note**: This repository is archived and no longer accepting contributions. The code is provided as-is for educational and reference purposes.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Related Links

- [Original Medium Articles](https://medium.com/@julsimon)
- [Jupyter Documentation](https://jupyter.org/documentation)
- [AWS SageMaker Documentation](https://docs.aws.amazon.com/sagemaker/)

## 📊 Repository Stats

![GitHub stars](https://img.shields.io/github/stars/julsimon/dlnotebooks?style=social)
![GitHub forks](https://img.shields.io/github/forks/julsimon/dlnotebooks?style=social)
![GitHub issues](https://img.shields.io/github/issues/julsimon/dlnotebooks)
![GitHub pull requests](https://img.shields.io/github/issues-pr/julsimon/dlnotebooks)

---

**Disclaimer**: This repository is archived and may contain outdated code or dependencies. Use at your own risk and consider updating frameworks and libraries for production use.
