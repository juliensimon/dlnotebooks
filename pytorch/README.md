# PyTorch Examples

[![PyTorch](https://img.shields.io/badge/PyTorch-1.8+-red.svg)](https://pytorch.org/)
[![Detectron2](https://img.shields.io/badge/Detectron2-Facebook-blue.svg)](https://github.com/facebookresearch/detectron2)
[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![Docker](https://img.shields.io/badge/Docker-Container-blue.svg)](https://docker.com/)
[![Archived](https://img.shields.io/badge/status-archived-red.svg)](https://github.com/julsimon/dlnotebooks)

> **⚠️ This folder is part of an archived repository.**

Examples demonstrating PyTorch for deep learning, including custom containers for SageMaker and computer vision applications with Detectron2.

## 📁 Contents

### Custom Containers
- **01-custom-container/**: PyTorch custom container for SageMaker
  - `Dockerfile`: Container configuration
  - `mnist_cnn.py`: PyTorch CNN training script
  - `notebook.ipynb`: SageMaker integration example
  - `data/`: Training and validation datasets

### Computer Vision
- **Detectron2_Tutorial.ipynb**: Object detection and instance segmentation with Facebook's Detectron2

## 🚀 Quick Start

### Prerequisites

```bash
pip install torch torchvision jupyter
```

### Running the Examples

1. **PyTorch Basics**: Start with `01-custom-container/notebook.ipynb`
2. **Computer Vision**: Try `Detectron2_Tutorial.ipynb`

## 📖 What You'll Learn

- PyTorch fundamentals and CNN implementation
- Custom Docker containers for SageMaker
- Object detection with Detectron2
- Instance segmentation techniques
- Model training and deployment workflows

## 🔗 Resources

- [PyTorch Documentation](https://pytorch.org/docs/)
- [Detectron2 GitHub](https://github.com/facebookresearch/detectron2)
- [AWS SageMaker PyTorch](https://docs.aws.amazon.com/sagemaker/latest/dg/pytorch.html)
- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/) 