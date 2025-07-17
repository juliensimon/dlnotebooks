# Keras/TensorFlow Examples

[![Keras](https://img.shields.io/badge/Keras-2.4+-red.svg)](https://keras.io/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.4+-orange.svg)](https://tensorflow.org/)
[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![SageMaker](https://img.shields.io/badge/AWS-SageMaker-yellow.svg)](https://aws.amazon.com/sagemaker/)
[![Archived](https://img.shields.io/badge/status-archived-red.svg)](https://github.com/julsimon/dlnotebooks)

> **⚠️ This folder is part of an archived repository.**

Comprehensive examples demonstrating Keras and TensorFlow integration with AWS SageMaker, from basic MNIST tutorials to advanced deployment scenarios.

## 📁 Contents

### Basic Tutorials
- **00-mnist.ipynb**: Basic MNIST classification with Keras
- **00-mnist-cnn.ipynb**: MNIST with Convolutional Neural Networks
- **00a-LSTM.ipynb**: Long Short-Term Memory networks example

### Custom Containers
- **01-custom-container/**: Docker-based SageMaker training
  - `Dockerfile.cpu` & `Dockerfile.gpu`: Container configurations
  - `mnist_cnn.py`: Training script
  - `notebook.ipynb`: SageMaker integration example

### Fashion MNIST Series
- **02-fashion-mnist/**: Fashion MNIST with custom containers
- **03-fashion-mnist-sagemaker/**: SageMaker integration
- **04-fashion-mnist-sagemaker-advanced/**: Advanced SageMaker features
- **05-keras-blog-post/**: Blog post examples
- **06-keras-workshop/**: Workshop materials
- **07-keras-fmnist-tf20/**: TensorFlow 2.0 specific examples

### Additional Resources
- **digits/**: Sample digit images (0-9)
- **reinvent2017Workshop/**: Workshop materials with visualizations

## 🚀 Quick Start

### Prerequisites

```bash
pip install tensorflow keras jupyter
```

### Running the Examples

1. **Beginner**: Start with `00-mnist.ipynb`
2. **CNN**: Try `00-mnist-cnn.ipynb`
3. **SageMaker**: Explore `03-fashion-mnist-sagemaker/`
4. **Advanced**: Check `04-fashion-mnist-sagemaker-advanced/`

## 📖 What You'll Learn

- Basic neural network implementation with Keras
- Convolutional Neural Networks (CNNs)
- Recurrent Neural Networks (LSTMs)
- AWS SageMaker integration
- Custom Docker containers for training
- Model deployment and inference
- Transfer learning techniques

## 🔗 Resources

- [Keras Documentation](https://keras.io/)
- [TensorFlow Documentation](https://tensorflow.org/)
- [AWS SageMaker Documentation](https://docs.aws.amazon.com/sagemaker/)
- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
- [Fashion MNIST Dataset](https://github.com/zalandoresearch/fashion-mnist) 