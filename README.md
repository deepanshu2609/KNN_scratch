# KNN_scratch
This repository contains a Python implementation of the k-Nearest Neighbors (k-NN) algorithm from scratch. k-NN is a simple yet powerful machine learning algorithm used for classification and regression tasks. This implementation demonstrates how the k-NN algorithm works, step-by-step, without relying on external libraries like scikit-learn. 

## Prerequisites

Ensure the following are installed:

- Python 3.6 or later versions
- Packages including `numpy`, `pandas`, `matplotlib`, `sklearn`

## KNN (K-Nearest Neighbors)

K-Nearest Neighbors is a simple yet powerful classification and regression algorithm that operates on the principle of finding the 'k' nearest data points to a given query point and making predictions based on the majority class or average value of those neighbors.

- here we have chosen the value of k less than sqrt(number of samples).
- We have used the Euclidean, Manhattan and cosine distance matrix.
- The dataset used for the KNN class contains Resnet and Vit embeddings.
- By default, it will take the distance matrix as euclidean and embedding as resnet, if not specified.

### Usage

To test the knn class script you can directly use it as,

`$ bash eval.sh /pathtodatafolder 1.py  `

This will return you an evaluation matrix containing accuracy, F1 score , precision and recall.
