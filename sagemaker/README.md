# AWS SageMaker Examples

[![SageMaker](https://img.shields.io/badge/AWS-SageMaker-yellow.svg)](https://aws.amazon.com/sagemaker/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.5+-green.svg)](https://xgboost.readthedocs.io/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.4+-orange.svg)](https://tensorflow.org/)
[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![DeepLens](https://img.shields.io/badge/AWS-DeepLens-purple.svg)](https://aws.amazon.com/deeplens/)
[![Archived](https://img.shields.io/badge/status-archived-red.svg)](https://github.com/julsimon/dlnotebooks)

> **⚠️ This folder is part of an archived repository.**

Comprehensive examples demonstrating AWS SageMaker capabilities, from image classification to time series forecasting and custom algorithms.

## 📁 Contents

### Image Classification
- **01-Image-classification-transfer-learning-cifar10.ipynb**: Transfer learning on CIFAR-10
- **02-Image-classification-transfer-learning-cifar10-train-from-scratch.ipynb**: Training from scratch
- **05-Image-classification-two-models.ipynb**: Multi-model classification
- **06-Image-classification-deeplens.ipynb**: AWS DeepLens integration
- **08-Image-classification-advanced.ipynb**: Advanced image classification techniques

### Machine Learning Algorithms
- **03-Factorization-Machines-Movielens.ipynb**: Recommendation system with MovieLens
- **04-DeepAR-temperatures.ipynb**: Time series forecasting with DeepAR
- **09-XGBoost-script-mode.ipynb**: XGBoost gradient boosting

### TensorFlow Integration
- **07-Tensorflow-script-mode.ipynb**: TensorFlow script mode training

### Supporting Files
- **caltech256_labels.txt**: Caltech-256 dataset labels
- **caltech256_lambda.py**: Lambda function for Caltech-256
- **xgb.py**: XGBoost training script

## 🚀 Quick Start

### Prerequisites

- AWS Account with SageMaker access
- SageMaker Studio or Jupyter notebook instance
- Required Python packages (see individual notebooks)

### Running the Examples

1. **Beginner**: Start with `01-Image-classification-transfer-learning-cifar10.ipynb`
2. **Recommendation Systems**: Try `03-Factorization-Machines-Movielens.ipynb`
3. **Time Series**: Explore `04-DeepAR-temperatures.ipynb`
4. **Advanced**: Check `08-Image-classification-advanced.ipynb`

## 📖 What You'll Learn

- AWS SageMaker fundamentals
- Transfer learning for image classification
- Factorization machines for recommendations
- Time series forecasting with DeepAR
- XGBoost gradient boosting
- Custom algorithm development
- Model deployment and inference
- AWS DeepLens integration

## 🔗 Resources

- [AWS SageMaker Documentation](https://docs.aws.amazon.com/sagemaker/)
- [SageMaker Built-in Algorithms](https://docs.aws.amazon.com/sagemaker/latest/dg/algos.html)
- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
- [MovieLens Dataset](https://grouplens.org/datasets/movielens/)
- [Caltech-256 Dataset](http://www.vision.caltech.edu/Image_Datasets/Caltech256/) 