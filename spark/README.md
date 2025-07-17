# Apache Spark ML Examples

[![Apache Spark](https://img.shields.io/badge/Apache-Spark-orange.svg)](https://spark.apache.org/)
[![SageMaker](https://img.shields.io/badge/AWS-SageMaker-yellow.svg)](https://aws.amazon.com/sagemaker/)
[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://python.org/)
[![Scala](https://img.shields.io/badge/Scala-2.12+-red.svg)](https://scala-lang.org/)
[![Zeppelin](https://img.shields.io/badge/Apache-Zeppelin-blue.svg)](https://zeppelin.apache.org/)
[![Archived](https://img.shields.io/badge/status-archived-red.svg)](https://github.com/julsimon/dlnotebooks)

> **⚠️ This folder is part of an archived repository.**

Examples demonstrating Apache Spark ML integration with AWS SageMaker, including both Python and Scala implementations for machine learning workflows.

## 📁 Contents

### SageMaker Spark Integration
- **sagemaker-spark/**: SageMaker Spark SDK examples
  - **standalone/**: Standalone Spark applications
    - `python-kmeans-mnist.py`: K-means clustering on MNIST
    - `python-xgboost-mnist.py`: XGBoost classification on MNIST
    - `scala-kmeans-mnist.scala`: Scala K-means implementation
    - `scala-pca-kmeans-pipeline.scala`: PCA + K-means pipeline
  - **zeppelin/**: Apache Zeppelin notebooks
    - `Classifying MNIST with the built-in XGBoost algorithm in Amazon SageMaker (Python).json`
    - `Clustering MNIST with a Pipeline_ PCA + K-Means (Scala).json`
    - `Clustering MNIST with the built-in K-Means algorithm in Amazon SageMaker (Python).json`
    - `Clustering MNIST with the built-in K-Means algorithm in Amazon SageMaker (Scala).json`

### Spam Classification
- **spam-classifier/**: Email spam classification example
  - `01 - Spam classifier.ipynb`: Jupyter notebook implementation
  - `02 - Spam classifier with SageMaker Spark SDK.json`: Zeppelin notebook
  - `ham/`: Ham (non-spam) email samples
  - `spam/`: Spam email samples

## 🚀 Quick Start

### Prerequisites

```bash
# For Python Spark
pip install pyspark sagemaker-pyspark

# For Scala (requires sbt)
# See individual examples for Scala setup
```

### Running the Examples

1. **Python Spark**: Start with `sagemaker-spark/standalone/python-kmeans-mnist.py`
2. **Scala Spark**: Try `sagemaker-spark/standalone/scala-kmeans-mnist.scala`
3. **Spam Classification**: Explore `spam-classifier/01 - Spam classifier.ipynb`

## 📖 What You'll Learn

- Apache Spark ML fundamentals
- SageMaker Spark SDK integration
- K-means clustering with Spark
- XGBoost classification
- Principal Component Analysis (PCA)
- Email spam classification
- Scala and Python implementations
- Zeppelin notebook workflows

## 🔗 Resources

- [Apache Spark Documentation](https://spark.apache.org/docs/latest/)
- [Spark ML Guide](https://spark.apache.org/docs/latest/ml-guide.html)
- [SageMaker Spark SDK](https://github.com/aws/sagemaker-spark)
- [Apache Zeppelin](https://zeppelin.apache.org/)
- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/) 